"""
baselines_cogact.py — Faithful CogACT ports of two training-free VLA token-reduction
baselines, for a head-to-head comparison against T3R (SigLIP-SAM prune + DiT bias).

  ADP  — Action-aware Dynamic Pruning  (ICLR'26, arXiv:2509.22093, chen7086/VLA-ADP)
         * Selection signal: text -> vision QK attention importance at LLM layer 0
           (model-internal; contrast with T3R's external SigLIP similarity).
         * Action-aware dynamic gating: per motion-window the method prunes in COARSE
           phases (large end-effector motion) and keeps all tokens in FINE phases
           (small motion). Decided online from the commanded end-effector deltas with the
           adjacent/extrema rule from the paper's default config.
         * Native config: qk_keep_ratio = 0.75 when pruning is ON (a *dynamic* effective
           ratio, NOT a fixed keep ratio).

  TeamVLA — Token Expand-and-Merge   (arXiv:2512.09927, Jasper-aaa/TEAM-VLA)
         * Similarity-sample salient patches per language token, then soft bipartite
           MERGE the remaining tokens into the anchors (W' = (S + W^T T)/(1 + s)).
           Merging (not dropping) is its signature contribution.
         * Native config: its own adaptive token budget (merge_topk), NOT a fixed ratio.

Both run through the IDENTICAL CogACT pipeline + position-id (RoPE) fix as T3R, so the
ONLY thing that differs from our method is the token-reduction strategy itself. We
intercept at `projector.forward` (post-projection, LLM embedding space): the vision
attention mask in PrismaticVLM.forward is built from the post-projector token count, so
reducing there yields a fully consistent attention mask with no HF-version coupling.

Per the user's design decision, the keep ratio is NOT fixed for these baselines — each
runs in its native configuration and reports its own achieved prune ratio vs success.
"""
import os
import sys
import math
import types

sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")
import numpy as np
import torch
import torch.nn.functional as F


class _MotionGate:
    """Action-aware dynamic gate (ADP). state==1 -> prune (coarse phase); 0 -> keep all.

    FAITHFUL to VLA-ADP prune_v2 (verified against chen7086/VLA-ADP
    run_libero_eval_prune_v2.py + prune_v2_config.json), adapted to CogACT's closed-loop
    (1 inference/step) control by accumulating a fixed W-step window:
      - delta_method='net'  : window delta = ||sum of per-step position deltas|| (NET
                              endpoint displacement), NOT the arc-length sum of norms.
                              Uses the commanded world_vector (position) as the eef-delta
                              proxy; rotation excluded (ADP 'net' is position-only).
      - initial_state=0     : start in KEEP.
      - adjacent/extrema    : threshold current net vs the extrema (max/min) of the last
                              `lookback` window deltas; the hysteresis band -> PRUNE (1)
                              (adjacent_last_state=False in the native config).
      - limit_consecutive_pruned: force a KEEP window after `max_consec_prune` (=3)
                              consecutive prune windows.
    """

    def __init__(self, W=3, lookback=2, initial=0, max_consec_prune=3):
        self.W = int(W)
        self.lookback = int(lookback)
        self.state = int(initial)
        self.initial = int(initial)
        self.max_consec_prune = int(max_consec_prune)
        self.vbuf = []      # per-step 3D position-delta vectors in the current window
        self.hist = []      # net displacement per closed window
        self.prune_streak = 1 if int(initial) == 1 else 0

    def observe(self, dvec):
        v = np.asarray(dvec, dtype=np.float64).reshape(-1)
        v = v[:3] if v.size >= 3 else np.pad(v, (0, 3 - v.size))
        self.vbuf.append(v)
        if len(self.vbuf) >= self.W:
            net = float(np.linalg.norm(np.sum(self.vbuf, axis=0)))  # NET (vector sum) displacement
            self.vbuf = []
            self.hist.append(net)
            if len(self.hist) < self.lookback + 1:
                ns = self.initial
            else:
                prev = self.hist[-(self.lookback + 1):-1]           # last `lookback` deltas
                up_thr = max(prev)
                dn_thr = min(prev)
                if net >= up_thr:
                    ns = 1
                elif net < dn_thr:
                    ns = 0
                else:
                    ns = 1                                          # hysteresis band -> PRUNE
            # limit_consecutive_pruned: cap consecutive prune windows
            if ns == 1:
                if self.prune_streak >= self.max_consec_prune:
                    ns = 0
                    self.prune_streak = 0
                else:
                    self.prune_streak += 1
            else:
                self.prune_streak = 0
            self.state = int(ns)

    def prune_on(self):
        return self.state == 1


class BaselineController:
    def __init__(self, model, method="adp",
                 adp_keep_ratio=0.75, adp_dynamic=True,
                 team_topk=80):
        self.m = model
        self.method = method.lower()
        assert self.method in ("adp", "team"), f"unknown baseline method {method}"
        # ADP knobs (native defaults from VLA-ADP prune_v2_config.json)
        self.adp_keep_ratio = float(os.environ.get("ADP_KEEP", str(adp_keep_ratio)))
        self.adp_dynamic = os.environ.get("ADP_DYNAMIC", "1" if adp_dynamic else "0") == "1"
        self.adp_use_rot = os.environ.get("ADP_USE_ROT", "1") == "1"
        # TeamVLA knobs (native default merge_topk=80; paper "Final top-80").
        self.team_topk = int(os.environ.get("TEAM_TOPK", str(team_topk)))
        # Faithful two-stage TeamVLA (paper 2512.09927): stage-1 SigLIP+expand+context
        # PRUNE before backbone, then stage-2 soft-bipartite MERGE to top-M anchors.
        self.team_twostage = os.environ.get("TEAM_TWOSTAGE", "0") == "1"
        self.team_u = float(os.environ.get("TEAM_U", "0.35"))     # context proportion
        self.team_k = int(os.environ.get("TEAM_K", "3"))          # spatial-expand conv kernel
        self._siglip = None

        self.gate = _MotionGate(
            W=int(os.environ.get("ADP_WIN", "3")),
            lookback=int(os.environ.get("ADP_LOOKBACK", "2")),
            initial=int(os.environ.get("ADP_INIT", "0")),
            max_consec_prune=int(os.environ.get("ADP_MAXPRUNE", "3")),
        )
        self.cur = None
        self._text_emb = None        # [1, Tq, D] cached language embeddings (BOS+text)
        self._sel_idx = None         # original indices (0..255) of kept/anchor vis tokens
        self.calls = 0
        self.kept_hist = []
        self._last_motion = np.zeros(3)
        self.NPF = 256               # nominal vision patches before reduction

        self._wrap_projector()
        self._wrap_llm()

    # ------------------------------------------------------------------
    def reset_episode(self, instruction):
        self.cur = instruction
        self._text_emb = None
        self._sel_idx = None
        self.gate = _MotionGate(W=self.gate.W, lookback=self.gate.lookback,
                                initial=self.gate.initial,
                                max_consec_prune=self.gate.max_consec_prune)

    # ------------------------------------------------------------------
    def _build_text_emb(self, instruction):
        """Cache language embeddings (BOS + prompt text) exactly as predict_action builds
        the input_ids, so the QK queries / merge guides match what the LLM sees."""
        vlm = self.m.vlm
        tok = vlm.llm_backbone.tokenizer
        pb = vlm.get_prompt_builder()
        pb.add_turn(role="human",
                    message=f"What action should the robot take to {instruction.lower()}?")
        ids = tok(pb.get_prompt(), truncation=True, return_tensors="pt").input_ids.to(vlm.device)
        ids = torch.cat([ids, torch.tensor([[29871, 2]], device=vlm.device)], dim=1)
        with torch.no_grad():
            emb = vlm.llm_backbone.embed_input_ids(ids)   # [1, Tq, D]
        self._text_emb = emb

    # ------------------------------------------------------------------
    def _adp_importance(self, vis, txt):
        """Text->vision QK attention importance at LLM layer 0 (VLA-ADP, qk_layer=0).
        vis: [1, V, D] projected patches (keys); txt: [1, Tq, D] language (queries)."""
        llm = self.m.vlm.llm_backbone.llm
        layer0 = llm.model.layers[0]
        attn = layer0.self_attn
        ln = layer0.input_layernorm
        cfg = llm.config
        nh = cfg.num_attention_heads
        hd = cfg.hidden_size // nh
        vf = vis.float()
        tf = txt.float()
        ht = ln(tf)
        hv = ln(vf)
        q = F.linear(ht, attn.q_proj.weight.float(), getattr(attn.q_proj, "bias", None))
        k = F.linear(hv, attn.k_proj.weight.float(), getattr(attn.k_proj, "bias", None))
        B, Tq, _ = q.shape
        V = k.shape[1]
        q = q.view(B, Tq, nh, hd).permute(0, 2, 1, 3)   # [B, nh, Tq, hd]
        k = k.view(B, V, nh, hd).permute(0, 2, 1, 3)    # [B, nh, V, hd]
        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B, nh, Tq, V]
        importance = scores.mean(dim=(0, 1, 2))                 # [V]
        return importance

    def _adp_select(self, vis, txt):
        V = vis.shape[1]
        keep_total = max(1, int(round(V * self.adp_keep_ratio)))
        if keep_total >= V:
            return None
        importance = self._adp_importance(vis, txt)
        _, idx = torch.topk(importance, k=keep_total, largest=True, sorted=False)
        return torch.sort(idx)[0].long()

    # ------------------------------------------------------------------
    def _team_merge(self, vis, txt):
        """TeamVLA token expand-merge in LLM embedding space.
        Anchors = top-`team_topk` patches by max cosine similarity to language tokens
        (similarity sampling). Remaining patches are soft-bipartite-merged into the anchors:
            W = softmax(S_n @ T_n^T / sqrt(D)) ;  S' = (S + W^T T) / (1 + s)
        Returns merged tokens [1, M, D] and the sorted anchor indices for RoPE positions.
        """
        B, V, D = vis.shape
        topk = min(self.team_topk, V)
        if topk >= V:
            return None, None
        vf = vis.float()
        tf = txt.float()
        v_n = F.normalize(vf, p=2, dim=-1)
        t_n = F.normalize(tf, p=2, dim=-1)
        sim = torch.bmm(v_n, t_n.transpose(1, 2)).max(dim=-1)[0]   # [B, V] text relevance
        _, top_idx = sim.topk(topk, dim=1)
        src_mask = torch.zeros(B, V, dtype=torch.bool, device=vis.device)
        src_mask.scatter_(1, top_idx, True)

        b = 0
        src_idx = torch.where(src_mask[b])[0]
        tgt_idx = torch.where(~src_mask[b])[0]
        S = vf[b, src_idx]      # [Ns, D]
        T = vf[b, tgt_idx]      # [Nt, D]
        if T.shape[0] == 0:
            merged = S
        else:
            def rms(x, eps=1e-6):
                return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)
            S_n = rms(S)
            T_n = rms(T)
            sim_st = torch.mm(S_n, T_n.transpose(0, 1)) / math.sqrt(D)   # [Ns, Nt]
            W = F.softmax(sim_st.transpose(0, 1), dim=-1)               # [Nt, Ns]
            A = torch.mm(W.transpose(0, 1), T)                          # [Ns, D]
            ones = torch.ones(T.shape[0], 1, device=T.device, dtype=T.dtype)
            s = torch.mm(W.transpose(0, 1), ones)                      # [Ns, 1]
            merged = (S + A) / (1 + s)
        merged = merged.unsqueeze(0).to(vis.dtype)                     # [1, Ns, D]
        anchor_idx = src_idx.long()                                    # already sorted asc
        return merged, anchor_idx

    # ------------------------------------------------------------------
    def _team_twostage(self, vis, txt):
        """FAITHFUL two-stage TEAM-VLA (arXiv 2512.09927), in-model (no external SigLIP):
        Stage 1 (pre-backbone PRUNE): projected-token <-> language-embedding cosine gives a
          sparse foreground seed map; a K x K ones-kernel convolution spatially EXPANDS the
          seeds into coherent regions; a `u` fraction of random CONTEXT tokens is added; all
          other tokens are DROPPED (this stage discards information).
        Stage 2 (MERGE): the surviving tokens are soft-bipartite-merged into the top-M anchors
          (by the same similarity), preserving the survivors' information in compact form.
        Returns merged tokens [1, M', D] (M' = min(M, |survivors|)) and sorted anchor indices.
        Placement note: CogACT produces actions from the final EOS/cognition state (no in-LLM
        action tokens), so the paper's action-guided MIDDLE-layer merge is adapted to a
        projector-level merge using the text-similarity signal.
        """
        B, V, D = vis.shape
        side = int(round(V ** 0.5))
        # Re-entrancy / non-grid guard: only operate on the RAW square patch grid (256->16x16).
        # If we get a non-square token count or a count <= our target M, the input has already
        # been reduced (the projector is re-invoked on merged output within a predict_action) --
        # return unchanged so we never double-reduce or reshape a non-grid tensor.
        if side * side != V or V <= self.team_topk:
            if getattr(self, "_dbgn", 0) < 8:
                self._dbgn = getattr(self, "_dbgn", 0) + 1
                print(f"[team2stage] skip: V={V} (not raw grid)", flush=True)
            return None, None
        if getattr(self, "_dbgn", 0) < 8:
            self._dbgn = getattr(self, "_dbgn", 0) + 1
            print(f"[team2stage] stage1 on V={V} grid={side}x{side}", flush=True)
        vf = vis.float()
        tf = txt.float()
        v_n = F.normalize(vf, p=2, dim=-1)
        t_n = F.normalize(tf, p=2, dim=-1)
        sim = torch.bmm(v_n, t_n.transpose(1, 2)).max(dim=-1)[0][0]     # [V]

        # ---- Stage 1: seeds -> spatial expand -> + context, drop the rest ----
        thr = sim.mean() + sim.std()
        seed = (sim >= thr).view(side, side).float()
        if float(seed.sum()) == 0:
            seed.view(-1)[int(sim.argmax())] = 1.0
        K = max(1, self.team_k)
        kern = torch.ones(1, 1, K, K, device=vis.device, dtype=seed.dtype)
        dens = F.conv2d(seed[None, None], kern, padding=K // 2)[0, 0]
        keep = (dens > 0).view(-1).clone()                             # expanded foreground
        rest = torch.where(~keep)[0]
        n_ctx = int(round(self.team_u * rest.numel()))
        if n_ctx > 0 and rest.numel() > 0:
            perm = rest[torch.randperm(rest.numel(), device=vis.device)[:n_ctx]]
            keep[perm] = True
        keep_idx = torch.where(keep)[0]
        if keep_idx.numel() == 0:
            keep_idx = torch.tensor([int(sim.argmax())], device=vis.device, dtype=torch.long)

        # ---- Stage 2: soft-bipartite merge survivors -> top-M anchors ----
        M = min(self.team_topk, int(keep_idx.numel()))
        sim_keep = sim[keep_idx]
        _, order = sim_keep.topk(M)
        anchor_idx = keep_idx[order]                                    # [M] original indices
        anchor_set = set(anchor_idx.tolist())
        tgt_list = [i for i in keep_idx.tolist() if i not in anchor_set]
        S = vf[0, anchor_idx]                                           # [M, D]
        if len(tgt_list) == 0:
            merged = S
        else:
            tgt_idx = torch.tensor(tgt_list, device=vis.device, dtype=torch.long)
            T = vf[0, tgt_idx]                                          # [Nt, D]

            def rms(x, eps=1e-6):
                return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)

            sim_st = torch.mm(rms(S), rms(T).transpose(0, 1)) / math.sqrt(D)   # [M, Nt]
            W = F.softmax(sim_st.transpose(0, 1), dim=-1)                      # [Nt, M]
            A = torch.mm(W.transpose(0, 1), T)                                 # [M, D]
            ones = torch.ones(T.shape[0], 1, device=T.device, dtype=T.dtype)
            s = torch.mm(W.transpose(0, 1), ones)                             # [M, 1]
            merged = (S + A) / (1 + s)
        sort_a, sort_o = torch.sort(anchor_idx)                        # RoPE needs ascending idx
        merged = merged[sort_o].unsqueeze(0).to(vis.dtype)             # [1, M, D]
        return merged, sort_a.long()

    # ------------------------------------------------------------------
    def _wrap_projector(self):
        """Reduce/merge the projected vision tokens. Runs inside PrismaticVLM.forward right
        after projection, so the multimodal attention mask (built from the post-projector
        token count) stays consistent automatically."""
        vlm = self.m.vlm
        proj = vlm.projector
        orig = proj.forward
        ctrl = self

        def fwd(self2, patch_features):
            projected = orig(patch_features)            # [B, 256, D]
            ctrl._sel_idx = None
            try:
                if ctrl._text_emb is None and ctrl.cur is not None:
                    ctrl._build_text_emb(ctrl.cur)
                if ctrl._text_emb is None:
                    return projected
                txt = ctrl._text_emb.to(projected.dtype)
                if ctrl.method == "adp":
                    if ctrl.adp_dynamic and not ctrl.gate.prune_on():
                        ctrl.kept_hist.append(int(projected.shape[1]))  # KEEP phase: all 256
                        return projected                # fine phase -> keep all tokens
                    sel = ctrl._adp_select(projected, txt)
                    if sel is None:
                        return projected
                    ctrl._sel_idx = sel.detach().cpu().numpy()
                    out = projected.index_select(dim=1, index=sel)
                    ctrl.kept_hist.append(int(out.shape[1]))
                    return out
                else:  # team
                    merged, anchor_idx = (ctrl._team_twostage(projected, txt)
                                          if ctrl.team_twostage
                                          else ctrl._team_merge(projected, txt))
                    if merged is None:
                        return projected
                    ctrl._sel_idx = anchor_idx.detach().cpu().numpy()
                    ctrl.kept_hist.append(int(merged.shape[1]))
                    return merged
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[baseline:{ctrl.method}] projector reduce failed:", e)
                ctrl._sel_idx = None
                return projected

        proj.forward = types.MethodType(fwd, proj)

    # ------------------------------------------------------------------
    def _wrap_llm(self):
        """Preserve ORIGINAL token RoPE positions under reduction. Layout after reduction:
        [BOS(0)][reduced vision at original idx+1][text(257..)]. Without this the LLM
        assigns contiguous positions and shifts kept/merged + text tokens off trained RoPE
        geometry (the same critical fix used by the T3R controller)."""
        llm_model = self.m.vlm.llm_backbone.llm.model
        orig = llm_model.forward
        ctrl = self
        NPF = self.NPF

        def fwd(*args, **kwargs):
            ie = kwargs.get("inputs_embeds", None)
            pos = kwargs.get("position_ids", None)
            sel = ctrl._sel_idx
            if sel is not None and ie is not None and pos is None and ie.shape[1] > 1:
                M = int(len(sel))
                seq = ie.shape[1]
                T = seq - 1 - M
                if T >= 0 and seq == 1 + M + T and M < NPF:
                    dev = ie.device
                    keep = torch.from_numpy(np.asarray(sel)).to(dev).long()
                    p = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=dev),
                        keep + 1,
                        torch.arange(1 + NPF, 1 + NPF + T, device=dev, dtype=torch.long),
                    ]).unsqueeze(0)
                    kwargs["position_ids"] = p
            return orig(*args, **kwargs)

        llm_model.forward = fwd

    # ------------------------------------------------------------------
    def _motion_of(self, action):
        """Commanded end-effector POSITION delta vector (world_vector) from the predicted
        action, used as ADP's eef-delta proxy for the 'net' window displacement."""
        try:
            a = np.asarray(action)
            if a.ndim == 1:
                a = a[None]
            return np.asarray(a[0, :3], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    def attach(self):
        m = self.m
        orig = m.predict_action
        ctrl = self

        def wrapped(image=None, instruction=None, unnorm_key=None, **kw):
            if instruction != ctrl.cur:
                ctrl.reset_episode(instruction)
            ctrl.calls += 1
            out = orig(image=image, instruction=instruction, unnorm_key=unnorm_key, **kw)
            if ctrl.method == "adp" and ctrl.adp_dynamic:
                a = out[0] if isinstance(out, tuple) else out
                ctrl._last_motion = ctrl._motion_of(a)
                ctrl.gate.observe(ctrl._last_motion)
            if len(ctrl.kept_hist) and (len(ctrl.kept_hist) % 25 == 1):
                k = ctrl.kept_hist[-1]
                print(f"[baseline:{ctrl.method}] vis tokens {k}/256 "
                      f"({(1 - k / 256) * 100:.0f}% reduced)"
                      + (f"  gate={'PRUNE' if ctrl.gate.prune_on() else 'KEEP'}"
                         if ctrl.method == "adp" else ""))
            return out

        m.predict_action = wrapped
