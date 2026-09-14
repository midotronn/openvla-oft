"""t3r_pi0.py — training-free multi-camera token reduction for openpi pi0.5 (PyTorch).

Patches PI0Pytorch.embed_prefix (+ sample_actions/denoise_step for RoPE-position preservation)
to reduce per-camera SigLIP image tokens BEFORE the PaliGemma prefix.

*** POSITION PRESERVATION (critical) ***
pi0 inference sets position_ids = cumsum(pad_masks)-1 and the suffix offset = sum(pad_masks).
So physically DROPPING tokens re-indexes every surviving token contiguously -> shifts all image
& language tokens off the absolute RoPE positions the model was trained on -> catastrophic SR
collapse (verified: any drop -> 0% while none -> 75%). Fix (T3R_POSFIX=1, default): each method
returns the ORIGINAL indices of the tokens it keeps; we rebuild position_ids so surviving tokens
stay at their trained absolute positions (camera ci occupies [ci*256, ci*256+256)), and the action
suffix starts at the original post-prefix position (n_cams*256 + n_lang). Same fix used on CogACT.

Camera order (AlohaInputs): 0 = cam_high (head/3rd-person), 1 = left wrist, 2 = right wrist.

Methods (matched per-camera budget):
  t3r    : language-guided SigLIP salience (cosine projected-patch vs mean instruction) -> hard drop
  adp    : VLA-ADP text->vision QK importance at Gemma layer 0 (GQA-aware) -> hard drop
  fastv  : FastV received-attention (all->vision) QK at Gemma layer 0 -> hard drop
  team   : TeamVLA similarity-sampled anchors + soft bipartite MERGE (info-preserving)
  none   : identity (== baseline)

Runtime config: /workspace/t3r_runtime.json  {"method":..,"cam_keep":"0.25,1,1","sel":"lang"}
(read each call -> switch config without server restart). Falls back to env T3R_METHOD/T3R_CAM_KEEP/
T3R_KEEP/T3R_SEL/T3R_POSFIX.
"""
import os, math, types, json
import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

_LOG = {"cfg": None}
_RUNTIME = "/workspace/t3r_runtime.json"


def _runtime_cfg():
    try:
        with open(_RUNTIME) as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_cam_keep(n, s):
    s = (s or "").strip()
    if s:
        vals = [float(x) for x in s.split(",")]
        if len(vals) < n:
            vals = vals + [1.0] * (n - len(vals))
        return vals[:n]
    return [1.0] * n


# ---------------------------------------------------------------- selection signals
def _t3r_salience(img_emb, lang_emb, lang_masks, lang_query, sel):
    B, N, D = img_emb.shape
    if sel == "norm":
        return img_emb.float().norm(dim=-1)
    if sel == "random":
        return torch.rand(B, N, device=img_emb.device)
    if sel == "meanlang":
        kn = F.normalize(img_emb.float(), dim=-1)
        qn = F.normalize(lang_query.float(), dim=-1)
        return (kn * qn).sum(-1)
    # default "lang": max cosine similarity to ANY instruction token (T3R/TeamVLA similarity sampling)
    v_n = F.normalize(img_emb.float(), dim=-1)                   # [B,V,D]
    t_n = F.normalize(lang_emb.float(), dim=-1)                  # [B,L,D]
    sim = torch.bmm(v_n, t_n.transpose(1, 2))                   # [B,V,L]
    mask = lang_masks[:, None, :].to(torch.bool)
    sim = sim.masked_fill(~mask, -1e4)
    return sim.max(dim=-1)[0]                                    # [B,V]


def _layer0_qk_scores(model, img_emb, lang_emb):
    """layer-0 attention scores [nh, Tq, V], queries = all prefix tokens over vision keys.
    Real Gemma layer-0 q/k projections (post input_layernorm), no RoPE (matches CogACT). GQA-aware."""
    layer0 = model.paligemma_with_expert.paligemma.language_model.layers[0]
    attn = layer0.self_attn
    ln = layer0.input_layernorm
    hd = attn.head_dim
    nh = attn.q_proj.weight.shape[0] // hd
    nkv = attn.k_proj.weight.shape[0] // hd
    qtokens = torch.cat([img_emb, lang_emb], dim=1)
    vf = img_emb.float()
    qf = qtokens.float()

    def _ln(x):
        out = ln(x, cond=None)
        return out[0] if isinstance(out, tuple) else out

    q = F.linear(_ln(qf), attn.q_proj.weight.float(), getattr(attn.q_proj, "bias", None))
    k = F.linear(_ln(vf), attn.k_proj.weight.float(), getattr(attn.k_proj, "bias", None))
    B, Tq, _ = q.shape
    V = k.shape[1]
    q = q.view(B, Tq, nh, hd).permute(0, 2, 1, 3)
    k = k.view(B, V, nkv, hd).permute(0, 2, 1, 3)
    if nkv != nh:
        k = k.repeat_interleave(nh // nkv, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
    return scores[0]                                            # [nh, Tq, V]


def _adp_importance(model, img_emb, lang_emb):
    scores = _layer0_qk_scores(model, img_emb, lang_emb)
    Tv = img_emb.shape[1]
    return scores[:, Tv:, :].mean(dim=(0, 1))                   # text -> vision (ADP)


def _fastv_importance(model, img_emb, lang_emb, lang_masks=None):
    """FastV-native signal: the attention the image tokens RECEIVE FROM THE LAST prompt token,
    averaged over heads (paper mechanism). In pi0's [image|language] prefix the last valid
    LANGUAGE token is the 'last token'. (Layer index is a minor deviation: FastV paper uses the
    attention from layer K-1 at K=2; we use the model's real layer-0 QK, no RoPE, as in the CogACT port.)"""
    scores = _layer0_qk_scores(model, img_emb, lang_emb)        # [nh, Tq, V], Tq = Tv + L
    Tv = img_emb.shape[1]
    if lang_masks is not None:
        valid = lang_masks[0].to(torch.bool)
        n_valid = int(valid.sum().item())
        last_lang = Tv + max(0, n_valid - 1)                    # last valid language query
    else:
        last_lang = scores.shape[1] - 1
    return scores[:, last_lang, :].mean(dim=0)                  # [V] last-token received attention


def _topk_idx(keep, importance, N):
    k = max(1, int(round(keep * N)))
    if k >= N:
        return torch.arange(N, device=importance.device)
    idx = importance.reshape(-1).topk(k).indices
    return idx.sort().values


def _thresh_idx(importance, k_std, N):
    """T3R 'no keep threshold' (native, adaptive) mode: keep tokens whose saliency exceeds an
    ADAPTIVE threshold mean + k_std*std (no fixed keep budget -> variable token count per image).
    Mirrors T3R's SigLIP-SAM object mask (keep the salient object patches, however many)."""
    imp = importance.reshape(-1).float()
    thr = imp.mean() + k_std * imp.std()
    idx = torch.where(imp > thr)[0]
    if idx.numel() == 0:                       # safety: never drop everything
        idx = imp.topk(1).indices
    return idx.sort().values


def _team_merge(img_emb, lang_emb, keep_k):
    """Return (merged [1,Ns,D], anchor_idx [Ns]) — anchors keep their original indices."""
    B, V, D = img_emb.shape
    topk = min(keep_k, V)
    if topk >= V:
        return img_emb, torch.arange(V, device=img_emb.device)
    vf = img_emb.float()
    tf = lang_emb.float()
    v_n = F.normalize(vf, p=2, dim=-1)
    t_n = F.normalize(tf, p=2, dim=-1)
    sim = torch.bmm(v_n, t_n.transpose(1, 2)).max(dim=-1)[0]     # [B,V]
    _, top_idx = sim.topk(topk, dim=1)
    src_mask = torch.zeros(V, dtype=torch.bool, device=img_emb.device)
    src_mask[top_idx[0]] = True
    src_idx = torch.where(src_mask)[0]
    tgt_idx = torch.where(~src_mask)[0]
    S = vf[0, src_idx]
    T = vf[0, tgt_idx]
    if T.shape[0] == 0:
        merged = S
    else:
        def rms(x, eps=1e-6):
            return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)
        sim_st = torch.mm(rms(S), rms(T).transpose(0, 1)) / math.sqrt(D)
        W = F.softmax(sim_st.transpose(0, 1), dim=-1)
        A = torch.mm(W.transpose(0, 1), T)
        ones = torch.ones(T.shape[0], 1, device=T.device, dtype=T.dtype)
        s = torch.mm(W.transpose(0, 1), ones)
        merged = (S + A) / (1 + s)
    order = torch.argsort(src_idx)
    return merged[order].unsqueeze(0).to(img_emb.dtype), src_idx[order]


def _team_twostage(img_emb, salience, topk, K, u, tau):
    """FAITHFUL two-stage TEAM-VLA (arXiv 2512.09927), per-camera on the 16x16 patch grid.
    Stage 1 (PRUNE): a saliency seed map -> KxK ones-conv spatial EXPAND -> + a `u` fraction of
      random CONTEXT tokens; all others dropped (this stage discards info).
    Stage 2 (MERGE): the survivors are soft-bipartite-merged into the top-M anchors (by saliency).
    Returns (merged [1,M',D], anchor_idx [M'] original indices). `salience` = per-token score [V]
    (norm saliency in pi0, since the paper's language-image SigLIP cosine doesn't transfer to the
    fused Gemma space — Stage-2 feature-space merge is unchanged/faithful)."""
    B, V, D = img_emb.shape
    side = int(round(V ** 0.5))
    if side * side != V or V <= topk:
        return img_emb, torch.arange(V, device=img_emb.device)
    vf = img_emb[0].float()
    sim = salience.reshape(-1).float()
    # ---- Stage 1: seeds -> spatial expand -> + context, drop the rest ----
    thr = sim.mean() + sim.std()
    seed = (sim >= thr).view(side, side).float()
    if float(seed.sum()) == 0:
        seed.view(-1)[int(sim.argmax())] = 1.0
    K = max(1, int(K))
    kern = torch.ones(1, 1, K, K, device=img_emb.device, dtype=seed.dtype)
    dens = F.conv2d(seed[None, None], kern, padding=K // 2)[0, 0]
    keep = (dens >= float(tau)).view(-1).clone()
    rest = torch.where(~keep)[0]
    n_ctx = int(round(u * rest.numel()))
    if n_ctx > 0 and rest.numel() > 0:
        perm = rest[torch.randperm(rest.numel(), device=img_emb.device)[:n_ctx]]
        keep[perm] = True
    keep_idx = torch.where(keep)[0]
    if keep_idx.numel() == 0:
        keep_idx = torch.tensor([int(sim.argmax())], device=img_emb.device, dtype=torch.long)
    # ---- Stage 2: soft-bipartite merge survivors -> top-M anchors ----
    M = min(topk, int(keep_idx.numel()))
    sim_keep = sim[keep_idx]
    _, order = sim_keep.topk(M)
    anchor_idx = keep_idx[order]
    anchor_set = set(anchor_idx.tolist())
    tgt_list = [i for i in keep_idx.tolist() if i not in anchor_set]
    S = vf[anchor_idx]
    if len(tgt_list) == 0:
        merged = S
    else:
        tgt_idx = torch.tensor(tgt_list, device=img_emb.device, dtype=torch.long)
        T = vf[tgt_idx]

        def rms(x, eps=1e-6):
            return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)

        sim_st = torch.mm(rms(S), rms(T).transpose(0, 1)) / math.sqrt(D)
        W = F.softmax(sim_st.transpose(0, 1), dim=-1)
        A = torch.mm(W.transpose(0, 1), T)
        ones = torch.ones(T.shape[0], 1, device=T.device, dtype=T.dtype)
        s = torch.mm(W.transpose(0, 1), ones)
        merged = (S + A) / (1 + s)
    sort_a, sort_o = torch.sort(anchor_idx)
    return merged[sort_o].unsqueeze(0).to(img_emb.dtype), sort_a.long()


# ------- ADP dynamic motion gate (Adjacent Extrema Rule, arXiv 2509.22093 Algorithm 1) -------
def _adp_gate_prune(model, state):
    """Return True if ADP should PRUNE this step, False = full vision. Tracks end-effector-proxy
    displacement (L2 change of the robot state vector) across infer calls. Coarse motion (delta at
    the recent max) -> prune; fine manipulation (delta at recent min) -> full vision; else hold.
    Cold start: first 2 calls full vision. Reset to full after 3 consecutive prunes."""
    import collections
    if state is None:
        return False
    st = state.detach().float().reshape(-1).cpu()
    prev = getattr(model, "_adp_prev_state", None)
    model._adp_prev_state = st
    hist = getattr(model, "_adp_delta_hist", None)
    if hist is None:
        hist = collections.deque(maxlen=3); model._adp_delta_hist = hist
    ncall = getattr(model, "_adp_ncall", 0) + 1; model._adp_ncall = ncall
    consec = getattr(model, "_adp_consec", 0)
    if prev is None or prev.shape != st.shape:
        return False                                    # cold start
    delta = float((st - prev).norm())
    hist.append(delta)
    if ncall <= 2 or len(hist) < 3:                     # cold start: first 2 full vision
        model._adp_state = 0; return False
    U, Vmin = max(hist), min(hist)
    prune = getattr(model, "_adp_state", 0)
    if delta >= U:
        prune = 1
    elif delta <= Vmin:
        prune = 0
    if prune == 1 and consec >= 3:                      # consecutive-prune meltdown reset
        prune = 0; consec = 0
    model._adp_consec = consec + 1 if prune == 1 else 0
    model._adp_state = prune
    return bool(prune)


# ------- Faithful T3R IG saliency + attention bias (arXiv T3R: integrated-gradients bias) -------
def _ig_saliency(model, observation, n_steps=8):
    """FAITHFUL integrated-gradients saliency over the INSTRUCTION (language) tokens, computed once
    per episode. IG is defined as (x - x') * mean_alpha[ dF/dx |_{x'+alpha(x-x')} ], with x = the
    language-token embeddings, x' = zero baseline, and F = a scalar reduction of pi0's predicted
    action (||v_t||^2 from a 1-step denoise). This mirrors T3R's captum LayerIntegratedGradients over
    text tokens (target = predicted-action logit in CogACT; here the action itself, since pi0 is a
    flow-matching action model with no output tokens). Returns per-language-token saliency [L]."""
    device = observation.state.device
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    lang_emb0 = model.paligemma_with_expert.embed_language_tokens(lang_tokens).float()
    lang_emb0 = (lang_emb0 * math.sqrt(lang_emb0.shape[-1])).detach()          # [1,L,D] input point
    baseline = torch.zeros_like(lang_emb0)                                      # x' = 0
    total_grad = torch.zeros_like(lang_emb0)
    model._ig_running = True                                                    # disable bias + reuse during IG
    try:
        for k in range(1, n_steps + 1):
            alpha = float(k) / n_steps
            x = (baseline + alpha * (lang_emb0 - baseline)).detach().requires_grad_(True)
            model._ig_lang_override = x
            with torch.enable_grad():
                torch.manual_seed(0)
                act = model.sample_actions(device, observation, num_steps=1)     # 1-step denoise
                target = act.float().pow(2).sum()
            g = torch.autograd.grad(target, x, retain_graph=False, allow_unused=True)[0]
            if g is not None:
                total_grad = total_grad + g.detach()
    finally:
        model._ig_lang_override = None
        model._ig_running = False
    avg_grad = total_grad / n_steps
    ig = ((lang_emb0 - baseline) * avg_grad).sum(-1).abs()[0]                    # [L]
    ig = ig * lang_masks[0].float()                                             # zero the padding tokens
    return ig


# --------- FAITHFUL T3R SigLIP-SAM object-guided pruning (arXiv T3R prune component) ---------
# Real T3R prunes visual tokens via SigLIP text<->patch similarity -> EfficientTAM/SAM object mask.
# pi0/PaliGemma's SigLIP tower was co-trained with Gemma (no aligned SigLIP TEXT tower), so we use a
# fully STANDALONE SigLIP-So400m (both towers) for the text<->patch similarity (verbatim reference
# technique: per-token features + noun/adjective token weights). sel="siglip" = weighted-sim top-k;
# sel="siglipsam" = SigLIP seed -> EfficientTAM point-prompt mask -> object patches + SigLIP context fill.
_SIG_LOW = {"[cls]","[sep]","[pad]","<s>","</s>","<pad>","<unk>","the","a","an","to","and","or","in","on",
            "at","of","for","with","from","pick","up","put","place","move","grab","take","push","pull",
            "open","close","lift","drop","insert","remove","click","press","tap","touch","using"}
_SIG_MED = {"between","next","near","beside","front","behind","left","right","top","bottom","above","below",
            "under","into","onto","inside","outside","center","centre","side","button"}
_SIG_MODEL = "google/siglip-so400m-patch14-384"


def _sig_tokw(tokens):
    w = []
    for t in tokens:
        tc = t.lower().replace("\u2581", "").replace("##", "").strip()
        if not tc or t.startswith("[") or t.startswith("<"): w.append(0.05)
        elif tc in _SIG_LOW: w.append(0.1)
        elif tc in _SIG_MED: w.append(0.3)
        else: w.append(1.0)
    return torch.tensor(w)


def _load_siglip(model):
    if getattr(model, "_sig_vis", None) is not None:
        return
    from transformers import AutoTokenizer, AutoProcessor, AutoModel
    tok = AutoTokenizer.from_pretrained(_SIG_MODEL)
    proc = AutoProcessor.from_pretrained(_SIG_MODEL)
    sm = AutoModel.from_pretrained(_SIG_MODEL, torch_dtype=torch.float32).to("cuda").eval()
    model._sig_tok = tok
    model._sig_proc = proc
    model._sig_txt = sm.text_model
    model._sig_vis = sm.vision_model
    print("[t3r_pi0] loaded standalone SigLIP-So400m for SigLIP(-SAM) pruning", flush=True)


def _img_bchw_to_pil(img_bchw):
    """pi0 preprocessed image [B,3,H,W] in [-1,1] -> PIL RGB (de-normalized)."""
    from PIL import Image
    import numpy as np
    x = img_bchw[0].detach().float()
    x = ((x * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def _siglip_scores(model, img_bchw, text, side):
    """Standalone-SigLIP weighted text<->patch similarity for one camera image.
    Returns (scores[side*side] tensor on cuda, side) aligned to pi0's patch grid via interpolation."""
    _load_siglip(model)
    pil = _img_bchw_to_pil(img_bchw)
    with torch.no_grad():
        tin = model._sig_tok(text, return_tensors="pt", padding=True).to("cuda")
        toks = model._sig_tok.convert_ids_to_tokens(tin["input_ids"][0])
        tf = model._sig_txt(**tin).last_hidden_state[0]                     # [L,D]
        tw = _sig_tokw(toks).to("cuda")
        pim = model._sig_proc(images=pil, return_tensors="pt").to("cuda")
        pf = model._sig_vis(pixel_values=pim["pixel_values"].float()).last_hidden_state[0]  # [P,D]
    P = pf.shape[0]; s0 = int(round(P ** 0.5))
    tfn = F.normalize(tf, dim=-1); pfn = F.normalize(pf, dim=-1)
    sim = tfn @ pfn.T                                                        # [L,P]
    sc = (sim * tw[:, None]).sum(0) / (tw.sum() + 1e-8)                      # [P]
    sc = sc.reshape(1, 1, s0, s0).float()
    sc = F.interpolate(sc, size=(side, side), mode="bilinear", align_corners=False)
    return sc.reshape(-1), side                                             # [side*side]


def _load_etam(model):
    if getattr(model, "_etam", None) is not None:
        return model._etam
    import sys
    base = "/workspace/EfficientTAM"
    if base not in sys.path:
        sys.path.insert(0, base)
    try:
        from efficient_track_anything.build_efficienttam import build_efficienttam
        from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor
        m = build_efficienttam(
            config_file="configs/efficienttam/efficienttam_s.yaml",
            ckpt_path="/workspace/EfficientTAM/checkpoints/efficienttam_s.pt",
            device="cuda", mode="eval")
        model._etam = EfficientTAMImagePredictor(m)
        print("[t3r_pi0] loaded EfficientTAM image predictor", flush=True)
    except Exception as e:
        print(f"[t3r_pi0] EfficientTAM load FAILED ({e}); siglipsam falls back to siglip top-k", flush=True)
        model._etam = False
    return model._etam


def _siglipsam_keep(model, img_bchw, scores, keep, full_n, side):
    """SigLIP seed -> EfficientTAM point-prompt object mask -> object patches + SigLIP context fill.
    Faithful reference 'sam_fill' selection. Returns sorted kept_idx (torch, on device) or None on failure."""
    import numpy as np
    pred = _load_etam(model)
    k = max(1, int(round(keep * full_n)))
    if not pred:
        return None
    try:
        pil = _img_bchw_to_pil(img_bchw)
        W, H = pil.size
        sc = scores.detach().float().cpu().numpy().reshape(side, side)
        # positive points = top-8 SigLIP patch centers; negative = bottom-5 (reference defaults ~10/5)
        order = np.argsort(-sc.flatten())
        pos = order[:8]; neg = order[-5:]
        def ctr(idx):
            r, c = divmod(int(idx), side)
            return [(c + 0.5) / side * W, (r + 0.5) / side * H]
        pc = np.array([ctr(i) for i in pos] + [ctr(i) for i in neg], dtype=np.float32)
        pl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        pred.set_image(np.asarray(pil))
        masks, mscores, _ = pred.predict(point_coords=pc, point_labels=pl, multimask_output=True)
        mask = masks[int(np.argmax(mscores))].astype(np.float32)             # [H,W]
        # mask -> patch selection over the side x side grid
        mt = torch.from_numpy(mask)[None, None]
        mt = F.interpolate(mt, size=(side, side), mode="area")[0, 0]         # [side,side]
        obj = (mt.reshape(-1) > 0.5).cpu().numpy()                           # object patches
        scf = sc.flatten()
        keepv = np.zeros(full_n, bool)
        if obj.sum() > 0:
            if int(obj.sum()) > k:                                           # object bigger than budget
                oi = np.where(obj)[0]; keepv[oi[np.argsort(-scf[oi])[:k]]] = True
            else:
                keepv[obj] = True
                rem = k - int(keepv.sum())
                for i in np.argsort(-scf):                                   # SigLIP-ranked context fill
                    if rem <= 0: break
                    if not keepv[i]: keepv[i] = True; rem -= 1
        else:
            keepv[np.argsort(-scf)[:k]] = True                              # SAM empty -> SigLIP top-k
        idx = torch.from_numpy(np.where(keepv)[0]).to(img_bchw.device).long()
        return idx.sort().values
    except Exception as e:
        print(f"[t3r_pi0] siglipsam mask failed ({e}); fallback to siglip top-k", flush=True)
        return None


# ---------------------------------------------------------------- install (patch 3 methods)
def install(model):
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        cfg = _runtime_cfg()
        method = cfg.get("method", os.environ.get("T3R_METHOD", "none"))
        sel = cfg.get("sel", os.environ.get("T3R_SEL", "lang"))
        cam_keep_str = cfg.get("cam_keep", os.environ.get("T3R_CAM_KEEP", ""))
        keep_single = cfg.get("keep", os.environ.get("T3R_KEEP", "1.0"))
        thresh = cfg.get("thresh", os.environ.get("T3R_THRESH", None))   # T3R adaptive threshold (std-mult); None=off
        thresh = None if thresh in (None, "", "none") else float(thresh)
        # TeamVLA native two-stage knobs (arXiv 2512.09927): top-M anchors, expand kernel K, context u, density tau
        team_native = str(cfg.get("team_native", os.environ.get("TEAM_NATIVE", "0"))) not in ("0", "false", "False")
        team_topk = int(cfg.get("team_topk", os.environ.get("TEAM_TOPK", "80")))
        team_k = int(cfg.get("team_k", os.environ.get("TEAM_K", "3")))
        team_u = float(cfg.get("team_u", os.environ.get("TEAM_U", "0.25")))
        team_tau = float(cfg.get("team_tau", os.environ.get("TEAM_TAU", "1")))
        # ADP native dynamic motion gate (arXiv 2509.22093 Algorithm 1)
        adp_gate = str(cfg.get("adp_gate", os.environ.get("ADP_GATE", "0"))) not in ("0", "false", "False")
        posfix = str(cfg.get("posfix", os.environ.get("T3R_POSFIX", "1"))) not in ("0", "false", "False")

        # IG-bias language override (used during the IG attribution loop to inject interpolated embeddings)
        if getattr(self, "_ig_lang_override", None) is not None:
            lang_emb = self._ig_lang_override
        else:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        lm = lang_masks[..., None].to(lang_emb.dtype)
        lang_query = (lang_emb * lm).sum(1, keepdim=True) / lm.sum(1, keepdim=True).clamp(min=1)

        cam_keep = _parse_cam_keep(len(images), cam_keep_str)
        if not cam_keep_str:
            cam_keep = [float(keep_single)] * len(images)

        # ADP native dynamic gate: decide ONCE per step whether to prune (else full vision this step)
        adp_do_prune = True
        if method == "adp" and adp_gate:
            adp_do_prune = _adp_gate_prune(self, getattr(self, "_t3r_cur_state", None))

        embs, pad_masks, att_masks, counts = [], [], [], []
        pos_list = []
        base = 0
        for ci, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            img_emb = self.paligemma_with_expert.embed_image(img)
            full_n = img_emb.shape[1]
            kept_idx = None
            t3r_thresh_on = (method == "t3r" and thresh is not None)
            team_on = (method == "team" and (team_native or cam_keep[ci] < 1.0))
            adp_on = (method == "adp" and cam_keep[ci] < 1.0 and adp_do_prune)
            reduce_on = (t3r_thresh_on or team_on or adp_on
                         or (method in ("t3r", "fastv") and cam_keep[ci] < 1.0))
            if method != "none" and reduce_on:
                keep = cam_keep[ci]
                if method == "t3r":
                    if sel in ("siglip", "siglipsam"):
                        side0 = int(round(full_n ** 0.5))
                        text = getattr(self, "_t3r_prompt", None) or "the object on the table"
                        sc, side0 = _siglip_scores(self, img, text, side0)     # [full_n] SigLIP scores
                        if sel == "siglipsam":
                            kept_idx = _siglipsam_keep(self, img, sc, keep, full_n, side0)
                        if kept_idx is None:                                    # siglip top-k (or sam fallback)
                            kept_idx = _topk_idx(keep, sc, full_n)
                    else:
                        sal = _t3r_salience(img_emb, lang_emb, lang_masks, lang_query, sel)
                        if t3r_thresh_on:
                            kept_idx = _thresh_idx(sal, thresh, full_n)          # adaptive, variable count
                        else:
                            kept_idx = _topk_idx(keep, sal, full_n)
                    img_emb = img_emb[:, kept_idx, :]
                elif method == "adp":
                    kept_idx = _topk_idx(keep, _adp_importance(self, img_emb, lang_emb), full_n)
                    img_emb = img_emb[:, kept_idx, :]
                elif method == "fastv":
                    kept_idx = _topk_idx(keep, _fastv_importance(self, img_emb, lang_emb, lang_masks), full_n)
                    img_emb = img_emb[:, kept_idx, :]
                elif method == "team":
                    if team_native:
                        sal = _t3r_salience(img_emb, lang_emb, lang_masks, lang_query, "norm")
                        img_emb, kept_idx = _team_twostage(img_emb, sal, team_topk, team_k, team_u, team_tau)
                    else:
                        k = max(1, int(round(keep * full_n)))
                        img_emb, kept_idx = _team_merge(img_emb, lang_emb, k)
                else:
                    raise ValueError(f"unknown T3R_METHOD={method}")
            if kept_idx is None:
                kept_idx = torch.arange(full_n, device=img_emb.device)
            b, num = img_emb.shape[:2]
            counts.append(num)
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(b, num))
            att_masks += [0] * num
            pos_list.append(base + kept_idx.to(torch.long))
            base += full_n                                        # advance by FULL cam size (preserve layout)

        n_lang = lang_emb.shape[1]
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * n_lang
        pos_list.append(base + torch.arange(n_lang, device=lang_emb.device, dtype=torch.long))
        suffix_start = base + n_lang
        # token layout for IG attention bias: language keys occupy [img_total, img_total+n_lang) in the prefix
        self._t3r_img_total = int(sum(counts))
        self._t3r_lang_len = int(n_lang)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        # stash preserved RoPE positions for sample_actions / denoise_step
        if posfix:
            self._t3r_prefix_pos = torch.cat(pos_list).unsqueeze(0).to(pad_masks.device)  # [1, L]
            self._t3r_suffix_start = int(suffix_start)
        else:
            self._t3r_prefix_pos = None
            self._t3r_suffix_start = None

        cfgkey = f"{method}|{sel}|{cam_keep}|{thresh}|{counts}|pf{int(posfix)}"
        if _LOG["cfg"] != cfgkey or (method == "team" and team_native) or (method == "adp"):
            extra = ""
            if method == "team" and team_native:
                extra = f" team_native(M={team_topk},K={team_k},u={team_u},tau={team_tau})"
            if method == "adp":
                extra = f" adp_gate={int(adp_gate)} do_prune={int(adp_do_prune)}"
            if _LOG["cfg"] != cfgkey:
                print(f"[t3r_pi0] method={method} sel={sel} cam_keep={cam_keep} thresh={thresh} "
                      f"img_tokens={counts} prefix_len={embs.shape[1]} posfix={int(posfix)} "
                      f"suffix_start={suffix_start}{extra}", flush=True)
                _LOG["cfg"] = cfgkey
        return embs, pad_masks, att_masks

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        if noise is None:
            # Noise mode. fixnoise=1 (default): every step uses the SAME seed-0 flow-matching noise
            # -> fully PAIRED across methods (only the token set differs), lowest-variance ranking.
            # fixnoise=0: stochastic (representative average SR; use to estimate true base SR).
            # Runtime-configurable via /workspace/t3r_runtime.json {"fixnoise":0/1} or T3R_FIXNOISE.
            _fn = _runtime_cfg().get("fixnoise", os.environ.get("T3R_FIXNOISE", "1"))
            if str(_fn) not in ("0", "false", "False"):
                torch.manual_seed(0)
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        self._t3r_cur_state = observation.state          # stash raw robot state for the ADP motion gate

        # ---- Faithful T3R IG attention-bias setup (once per prompt; skipped during IG attribution) ----
        _cfg = _runtime_cfg()
        bias_on = str(_cfg.get("bias", os.environ.get("T3R_BIAS", "0"))) not in ("0", "false", "False")
        self._t3r_bias_cols = None
        if bias_on and not getattr(self, "_ig_running", False):
            n_ig = int(_cfg.get("bias_nsteps", os.environ.get("T3R_BIAS_NSTEPS", "8")))
            top_ratio = float(_cfg.get("bias_topratio", os.environ.get("T3R_BIAS_TOPRATIO", "0.25")))
            strength = float(_cfg.get("bias_strength", os.environ.get("T3R_BIAS_STRENGTH", "2.0")))
            key = int(lang_tokens.sum().item()) ^ (lang_tokens.shape[1] << 8)
            if getattr(self, "_ig_cache_key", None) != key:
                sal = _ig_saliency(self, observation, n_steps=n_ig)     # [L] per-lang-token IG saliency
                self._ig_cache_key = key
                self._ig_sal = sal
                if getattr(self, "_ig_logged", 0) < 3:
                    self._ig_logged = getattr(self, "_ig_logged", 0) + 1
                    L = sal.shape[0]; k = max(1, int(round(top_ratio * L)))
                    print(f"[t3r_pi0] IG bias: L={L} top{k} sal[min={float(sal.min()):.3e} "
                          f"max={float(sal.max()):.3e}] strength={strength}", flush=True)
            self._t3r_bias_pending = (top_ratio, strength)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # now that embed_prefix set the image-token total, map salient LANGUAGE tokens -> prefix key columns
        if bias_on and not getattr(self, "_ig_running", False) and getattr(self, "_ig_sal", None) is not None:
            top_ratio, strength = self._t3r_bias_pending
            sal = self._ig_sal
            L = sal.shape[0]; k = max(1, int(round(top_ratio * L)))
            top = torch.topk(sal, min(k, L)).indices
            w = sal[top]
            w = w / w.max().clamp(min=1e-8)                          # normalize weights to [0,1]
            self._t3r_bias_cols = (self._t3r_img_total + top).to(torch.long)
            self._t3r_bias_w = w.to(prefix_embs.dtype)
            self._t3r_bias_strength = strength

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        if getattr(self, "_t3r_prefix_pos", None) is not None:
            prefix_position_ids = self._t3r_prefix_pos
        else:
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        if getattr(self, "_t3r_suffix_start", None) is not None:
            prefix_offsets = torch.full((batch_size, 1), self._t3r_suffix_start,
                                        device=suffix_pad_masks.device, dtype=torch.long)
        else:
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # ---- Faithful T3R IG attention bias: add +strength*weight at salient LANGUAGE key columns,
        # for the ACTION query rows only (last action_horizon), at allowed positions. Steers the
        # action queries toward instruction-salient tokens (T3R additive SDPA bias analog). ----
        if getattr(self, "_t3r_bias_cols", None) is not None and not getattr(self, "_ig_running", False):
            S = full_att_2d_masks_4d.shape[-1]
            Q = full_att_2d_masks_4d.shape[-2]
            cols = self._t3r_bias_cols[(self._t3r_bias_cols >= 0) & (self._t3r_bias_cols < S)]
            if cols.numel() > 0:
                biasvec = torch.zeros(S, device=full_att_2d_masks_4d.device, dtype=full_att_2d_masks_4d.dtype)
                biasvec[cols] = self._t3r_bias_strength * self._t3r_bias_w[:cols.numel()].to(full_att_2d_masks_4d.dtype)
                aw = self.config.action_horizon
                rowmask = torch.zeros(Q, device=full_att_2d_masks_4d.device, dtype=full_att_2d_masks_4d.dtype)
                rowmask[max(0, Q - aw):] = 1.0                        # bias only the action-token query rows
                add = rowmask[None, None, :, None] * biasvec[None, None, None, :]
                add = add * (full_att_2d_masks_4d == 0).to(add.dtype)  # only bias currently-allowed keys
                full_att_2d_masks_4d = full_att_2d_masks_4d + add
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=[None, suffix_embs],
            use_cache=False, adarms_cond=[None, adarms_cond])
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    model.embed_prefix = types.MethodType(embed_prefix, model)
    model.sample_actions = types.MethodType(sample_actions, model)
    model.denoise_step = types.MethodType(denoise_step, model)
    print(f"[t3r_pi0] patched embed_prefix + sample_actions + denoise_step "
          f"(method={os.environ.get('T3R_METHOD','none')} posfix={os.environ.get('T3R_POSFIX','1')} "
          f"IG-bias={os.environ.get('T3R_BIAS','0')})", flush=True)
