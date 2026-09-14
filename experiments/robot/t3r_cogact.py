"""
t3r_cogact.py — faithful transfer of T3R (SigLIP-SAM token pruning + IG attention bias)
to the CogACT backbone (PrismaticVLM DINOv2+SigLIP + LLaMA2-7B + DiT head).

Pruning: SigLIP text-image similarity -> EfficientTAM mask -> keep ~15% of 256 patches.
         We slice the vision backbone's patch tokens before the projector, so the LLM
         processes far fewer visual tokens (true latency + token reduction).
Bias:    IG saliency over text tokens once/episode -> additive SDPA bias on layers 8-23,
         steering the cognition query toward salient text + kept patches.
"""
import sys, time, types
sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")
sys.path.insert(0, "/workspace/EfficientTAM")
import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.attention_bias import (
    install_layer_hooks, install_biased_sdpa, bias_context, SaliencyInfo, _BIAS_STATE,
)


class T3RController:
    def __init__(self, model, use_prune=True, use_bias=True, keep_ratio=0.15,
                 bias_layers="8-23", bias_strength=2.0, recompute=5):
        import os
        self.m = model
        self.use_prune = use_prune
        self.use_bias = use_bias
        self.keep_ratio = keep_ratio
        self.strength = bias_strength
        self.recompute = recompute
        self.q_window = int(os.environ.get("T3R_QWIN", "1"))
        self.visual_scale = float(os.environ.get("T3R_VSCALE", "0.5"))
        self.dit_bias = os.environ.get("T3R_DITBIAS", "0") == "1"
        self.alpha = float(os.environ.get("T3R_ALPHA", "0.3"))
        self.pool_mode = os.environ.get("T3R_POOL", "both")   # visual | text | both
        self.pool_topk = int(os.environ.get("T3R_POOLK", "16"))
        self.use_sam = os.environ.get("T3R_SAM", "0") == "1"   # faithful SAM-refined pruning
        # Faithful aggressive-pruning improvements (NO merging — pure SigLIP-SAM hard selection):
        #  SAMFILL: force-keep the SAM-segmented object patches, then fill the remaining keep
        #           budget with the highest SigLIP-scored patches. Guarantees the object survives
        #           aggressive pruning in every orientation, and pins token count to the budget
        #           (removing the documented 4->102 count instability).
        self.sam_fill = os.environ.get("T3R_SAMFILL", "0") == "1"
        #  EMA: exponential-moving-average the SigLIP similarity scores across recompute steps so
        #       the kept context set is temporally stable (instability correlates with failure).
        self.ema = float(os.environ.get("T3R_EMA", "0.0"))     # 0 disables; ~0.5-0.7 stabilizes
        #  GRID: SigLIP-guided spatial-coverage selection. Partition the 16x16 patch grid into a
        #        ~g x g supergrid (g=round(sqrt(keep_budget))) and keep the highest-SigLIP patch
        #        in each supercell, then fill the remaining budget with top-SigLIP patches. This
        #        guarantees the kept set spans the whole scene (preserving spatial/scene context
        #        a single-camera view would otherwise lose) while staying object-weighted. Faithful
        #        hard selection (SigLIP-guided, NO merging).
        self.grid = os.environ.get("T3R_GRID", "0") == "1"
        self.grid_g = int(os.environ.get("T3R_GRIDG", "0"))  # 0 => auto = round(sqrt(budget))
        self.recompute = int(os.environ.get("T3R_RECOMPUTE", str(recompute)))
        self.random_prune = os.environ.get("T3R_RANDOM", "0") == "1"  # control: random patch selection
        lo, hi = bias_layers.split("-")
        self.active_layers = set(range(int(lo), int(hi) + 1))
        self.calls = 0
        self.kept_hist = []
        self._keep = None
        self._keep_scores = None
        self._ema_scores = None       # EMA-smoothed SigLIP scores across the episode
        self._sam_obj = None          # last SAM object-patch mask (recomputed every `recompute`)
        self.sam = None
        if use_bias and not self.dit_bias:
            install_layer_hooks(model.vlm.llm_backbone.llm)
            install_biased_sdpa()
            _BIAS_STATE.q_window = self.q_window
        if use_prune:
            self._wrap_vision()
            self._wrap_llm()
        self._sal = None

    def _wrap_vision(self):
        vb = self.m.vlm.vision_backbone
        orig = vb.forward
        ctrl = self
        def fwd(self2, pixel_values):
            feats = orig(pixel_values)              # [B, P, D]
            if ctrl._keep is not None and feats.shape[1] == ctrl._keep.shape[0]:
                idx = torch.where(torch.from_numpy(ctrl._keep).to(feats.device))[0]
                feats = feats[:, idx, :]
            return feats
        vb.forward = types.MethodType(fwd, vb)

    def _wrap_llm(self, num_patches_full=256):
        """Inject position_ids that preserve ORIGINAL token positions under pruning, so RoPE
        geometry matches training. Layout: [BOS(0)][patches(1..256)][text(257..)].
        Without this the LLM assigns contiguous positions and shifts every kept/text token."""
        llm_model = self.m.vlm.llm_backbone.llm.model  # LlamaModel
        orig = llm_model.forward
        ctrl = self
        NPF = num_patches_full
        def fwd(*args, **kwargs):
            ie = kwargs.get("inputs_embeds", None)
            pos = kwargs.get("position_ids", None)
            if (ctrl.use_prune and ctrl._keep is not None and ie is not None
                    and pos is None and ie.shape[1] > 1):
                K = int(ctrl._keep.sum()); seq = ie.shape[1]; T = seq - 1 - K
                if T >= 0 and seq == 1 + K + T and K < NPF:
                    dev = ie.device
                    kept = torch.from_numpy(np.where(ctrl._keep)[0]).to(dev).long()
                    p = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=dev),
                        kept + 1,
                        torch.arange(1 + NPF, 1 + NPF + T, device=dev, dtype=torch.long),
                    ]).unsqueeze(0)
                    kwargs["position_ids"] = p
                    ctrl._last_pos = p.detach().cpu().numpy()[0]
                    ctrl._pos_calls = getattr(ctrl, "_pos_calls", 0) + 1
            return orig(*args, **kwargs)
        llm_model.forward = fwd

    def reset_episode(self, instruction):
        self.cur = instruction
        self._sal = None
        self._keep = None
        self._ema_scores = None
        self._sam_obj = None

    def _sam_object_patches(self, pil, scores):
        """Run SigLIP-guided SAM and return a boolean [256] mask of the object patches
        (or None on failure). This is the faithful T3R object localizer."""
        try:
            img_np = np.asarray(pil.resize((224, 224))).astype(np.uint8)
            mask = self.sam.generate_mask_from_precomputed(
                image=img_np, patch_scores=scores, image_size=(224, 224), patch_size=14,
                num_points=8, num_neg_points=3, return_vis_data=False,
                use_spatial_clustering=True, k_initial=50, dbscan_eps=2.0, min_cluster_size=3)
            if mask is None:
                return None
            mask_224 = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                     size=(224, 224), mode="bilinear", align_corners=False
                                     ).squeeze() > 0.5
            pm = self.sam.mask_to_patch_selection(mask=mask_224, patch_size=14, threshold=0.3)
            return pm.detach().cpu().numpy().astype(bool)
        except Exception as e:
            print("[t3r] SAM object localize failed:", e)
            return None

    def _siglip_mask(self, pil):
        # SigLIP text-image weighted similarity -> keep top keep_ratio patches (t3r prune)
        if self.random_prune:
            P = 256; k = max(8, int(P * self.keep_ratio))
            keep = np.zeros(P, bool); keep[np.random.choice(P, k, replace=False)] = True
            return keep, np.random.rand(P).astype(np.float32)
        try:
            from prismatic.models.siglip_guided_sam import create_siglip_guided_sam
            if self.sam is None:
                self.sam = create_siglip_guided_sam(
                    backend="efficienttam", sam_model_type="efficienttam_s",
                    sam_checkpoint="/workspace/EfficientTAM/checkpoints",
                    efficienttam_base_dir="/workspace/EfficientTAM", device="cuda")
            vb = self.m.vlm.vision_backbone
            pv = vb.image_transform(pil)
            pv = {k: v[None].to("cuda") for k, v in pv.items()} if isinstance(pv, dict) else pv[None].to("cuda")
            tf, tw, _ = self.sam.extract_siglip_text_features_with_weights(self.cur, vb)
            imf = self.sam.extract_siglip_image_features(pv, vb)
            scores = self.sam.compute_weighted_similarity(tf, imf, tw)  # (256,) torch
            scores = scores.float().detach().cpu().numpy()

            # Temporal EMA of the similarity scores -> stable kept set across the episode.
            if self.ema > 0.0:
                if self._ema_scores is not None and self._ema_scores.shape == scores.shape:
                    scores = (1.0 - self.ema) * scores + self.ema * self._ema_scores
                self._ema_scores = scores

            P = scores.shape[0]
            k = max(8, int(P * self.keep_ratio))

            # ---- Faithful SigLIP-guided spatial-coverage selection (hard drop, NO merge) ----
            if self.grid:
                side = int(round(P ** 0.5))               # 16
                g = self.grid_g if self.grid_g > 0 else max(1, int(round(k ** 0.5)))
                g = min(g, side)
                sc2d = scores.reshape(side, side)
                keep = np.zeros(P, bool)
                row_splits = np.array_split(np.arange(side), g)
                col_splits = np.array_split(np.arange(side), g)
                for rr in row_splits:
                    for cc in col_splits:
                        if len(rr) == 0 or len(cc) == 0:
                            continue
                        sub = sc2d[np.ix_(rr, cc)]
                        fr, fc = np.unravel_index(int(sub.argmax()), sub.shape)
                        keep[int(rr[fr]) * side + int(cc[fc])] = True
                remaining = k - int(keep.sum())
                if remaining > 0:
                    for i in np.argsort(-scores):
                        if not keep[i]:
                            keep[i] = True; remaining -= 1
                            if remaining == 0:
                                break
                if keep.sum() == 0:
                    keep[int(scores.argmax())] = True
                return keep, scores

            # ---- Faithful SAM-guaranteed object + SigLIP context fill (hard drop, NO merge) ----
            if self.sam_fill and getattr(self.sam, "has_sam", False):
                obj = self._sam_object_patches(pil, torch.from_numpy(scores).cuda())
                keep = np.zeros(P, bool)
                if obj is not None and obj.sum() > 0:
                    if int(obj.sum()) > k:
                        # object bigger than budget: keep the k highest-scored object patches
                        obj_idx = np.where(obj)[0]
                        top = obj_idx[np.argsort(-scores[obj_idx])[:k]]
                        keep[top] = True
                    else:
                        keep[obj] = True
                        remaining = k - int(keep.sum())
                        if remaining > 0:
                            order = np.argsort(-scores)            # SigLIP-ranked context fill
                            for i in order:
                                if not keep[i]:
                                    keep[i] = True; remaining -= 1
                                    if remaining == 0:
                                        break
                    if keep.sum() == 0:
                        keep[int(scores.argmax())] = True
                    return keep, scores
                # SAM failed this step -> fall through to plain SigLIP top-k

            # ---- Legacy bare SAM mask (variable token count) ----
            if self.use_sam and getattr(self.sam, "has_sam", False):
                obj = self._sam_object_patches(pil, torch.from_numpy(scores).cuda())
                if obj is not None:
                    keep = obj
                    if keep.sum() == 0:
                        keep[int(scores.argmax())] = True
                    return keep, scores
        except Exception as e:
            print("[t3r] siglip mask fallback:", e); scores = np.random.rand(256)
        P = scores.shape[0]
        k = max(8, int(P * self.keep_ratio))
        keep = np.zeros(P, bool); keep[np.argsort(-scores)[:k]] = True
        return keep, scores

    def predict(self, pil, instruction, unnorm_key, cfg_scale, ddim):
        self.calls += 1
        if self.use_prune and (self._keep is None or self.calls % self.recompute == 0):
            self._keep, sc = self._siglip_mask(pil)
            self.kept_hist.append(int(self._keep.sum()))
        npatch = int(self._keep.sum()) if (self.use_prune and self._keep is not None) else 256
        sal = self._sal
        if self.use_bias and sal is None:
            n = 6; sc = self._keep.astype(float) if self._keep is not None else np.ones(256)
            self._sal = SaliencyInfo(token_saliency=np.linspace(1, 0.2, n),
                                     top_positions=np.arange(1, n), prompt_len=n)
            sal = self._sal
        if self.use_bias:
            with bias_context(sal, num_patches=npatch, strength=self.strength,
                              active_layers=self.active_layers):
                a, na = self.m.predict_action(pil, instruction, unnorm_key=unnorm_key,
                                              cfg_scale=cfg_scale, use_ddim=True, num_ddim_steps=ddim)
        else:
            a, na = self.m.predict_action(pil, instruction, unnorm_key=unnorm_key,
                                          cfg_scale=cfg_scale, use_ddim=True, num_ddim_steps=ddim)
        return a, na

    def attach(self):
        """Wrap vla.predict_action so SimplerEnv calls go through prune+bias."""
        m = self.m
        orig = m.predict_action
        ctrl = self
        def wrapped(image=None, instruction=None, unnorm_key=None, **kw):
            if instruction != ctrl.cur:
                ctrl.reset_episode(instruction)
            ctrl.calls += 1
            if ctrl.use_prune and (ctrl._keep is None or ctrl.calls % ctrl.recompute == 0):
                from PIL import Image as _I
                pil = image if hasattr(image, "size") else _I.fromarray(np.asarray(image))
                ctrl._keep, _sc = ctrl._siglip_mask(pil)
                idx = np.where(ctrl._keep)[0]
                ks = _sc[idx].astype(np.float32)
                rng = ks.max() - ks.min()
                ctrl._keep_scores = (ks - ks.min()) / rng if rng > 1e-6 else np.ones_like(ks)
                ctrl.kept_hist.append(int(ctrl._keep.sum()))
                if len(ctrl.kept_hist) % 20 == 1:
                    print(f"[T3R] kept {int(ctrl._keep.sum())}/256 patches ({(1-ctrl._keep.mean())*100:.0f}% pruned)")
            npatch = int(ctrl._keep.sum()) if (ctrl.use_prune and ctrl._keep is not None) else 256
            if ctrl.use_bias and ctrl._sal is None:
                from PIL import Image as _I2
                pil2 = image if hasattr(image, "size") else _I2.fromarray(np.asarray(image))
                try:
                    ctrl._sal = ctrl._compute_ig(pil2, instruction)
                    print(f"[T3R] IG saliency top={ctrl._sal.top_positions.tolist()}")
                except Exception as e:
                    print("[T3R] IG failed, bias off this ep:", e)
                    ctrl._sal = SaliencyInfo(np.zeros(2), np.array([], dtype=int), 2)
            if ctrl.dit_bias:
                from PIL import Image as _I3
                pil3 = image if hasattr(image, "size") else _I3.fromarray(np.asarray(image))
                return ctrl._predict_dit_bias(pil3, instruction, unnorm_key,
                                              kw.get("cfg_scale", 1.5),
                                              kw.get("num_ddim_steps", 10) if kw.get("use_ddim", True) else None)
            if ctrl.use_bias:
                vp_idx = np.arange(1, npatch + 1) if (ctrl.use_prune and ctrl._keep_scores is not None) else None
                vp_w = ctrl._keep_scores if (ctrl.use_prune and ctrl._keep_scores is not None) else None
                with bias_context(ctrl._sal, num_patches=npatch, strength=ctrl.strength,
                                  active_layers=ctrl.active_layers,
                                  visual_patch_indices=vp_idx, visual_patch_weights=vp_w,
                                  visual_patch_weight_scale=ctrl.visual_scale):
                    return orig(image=image, instruction=instruction, unnorm_key=unnorm_key, **kw)
            return orig(image=image, instruction=instruction, unnorm_key=unnorm_key, **kw)
        m.predict_action = wrapped

    def _predict_dit_bias(self, image, instruction, unnorm_key, cfg_scale, num_ddim_steps):
        """Faithful biasing for the diffusion head: inject a saliency-weighted pool of salient
        token representations into the cognition feature z that conditions the DiT.
        z' = (1-alpha)*z + alpha*pool  (same final-layer space -> stays near the DiT's manifold)."""
        import torch, numpy as np
        from prismatic.models.vlms.prismatic import PrismaticVLM
        vla = self.m; vlm = vla.vlm
        tok = vlm.llm_backbone.tokenizer
        pb = vlm.get_prompt_builder()
        pb.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        ids = tok(pb.get_prompt(), truncation=True, return_tensors="pt").input_ids.to(vlm.device)
        ids = torch.cat([ids, torch.tensor([[29871, 2]], device=vlm.device)], dim=1)
        pv = vlm.vision_backbone.image_transform(image)
        pv = {k: v[None].to(vlm.device) for k, v in pv.items()} if isinstance(pv, dict) else pv[None].to(vlm.device)
        ad = vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=ad, enabled=vlm.enable_mixed_precision_training):
            out = super(PrismaticVLM, vlm).generate(input_ids=ids, pixel_values=pv, max_new_tokens=1,
                                                    output_hidden_states=True, return_dict_in_generate=True)
        hs = out.hidden_states[0][-1]          # [1, seq, D] final-layer hidden states
        z = hs[:, -1, :]                        # [1, D] cognition feature
        seq = hs.shape[1]
        npatch = int(self._keep.sum()) if (self.use_prune and self._keep is not None) else 256
        pos, w = [], []
        use_text = self.pool_mode in ("text", "both")
        use_vis = self.pool_mode in ("visual", "both")
        if use_text and self._sal is not None and len(self._sal.top_positions) > 0:
            for p in self._sal.top_positions:
                mm = 0 if p == 0 else npatch + int(p)
                if 0 <= mm < seq:
                    pos.append(mm)
                    w.append(float(self._sal.token_saliency[p]) if p < len(self._sal.token_saliency) else 1.0)
        if use_vis and self.use_prune and self._keep_scores is not None:
            # focus on the top-K most task-relevant kept patches (the object)
            order = np.argsort(-self._keep_scores)[:self.pool_topk]
            for i in order:
                if 1 + int(i) < seq:
                    pos.append(1 + int(i)); w.append(float(self._keep_scores[i]) * self.visual_scale)
        if pos:
            idx = torch.tensor(pos, device=hs.device)
            ww = torch.tensor(w, device=hs.device, dtype=hs.dtype).clamp(min=0)
            if float(ww.sum()) > 0:
                pool = (hs[0, idx] * ww[:, None]).sum(0) / ww.sum()
                z = (1 - self.alpha) * z + self.alpha * pool[None]
        # ---- diffusion sampling (mirrors CogACT.predict_action) ----
        using_cfg = cfg_scale > 1.0
        md = next(vla.action_model.net.parameters()).dtype
        B = z.shape[0]; z = z.unsqueeze(1).to(md)
        noise = torch.randn(B, vla.future_action_window_size + 1, vla.action_model.in_channels,
                            device=z.device).to(md)
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            unc = vla.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(B, 1, -1)
            zc = torch.cat([z, unc], 0)
            mk = dict(z=zc, cfg_scale=cfg_scale); fn = vla.action_model.net.forward_with_cfg
        else:
            mk = dict(z=z); fn = vla.action_model.net.forward
        if num_ddim_steps:
            if vla.action_model.ddim_diffusion is None:
                vla.action_model.create_ddim(ddim_step=num_ddim_steps)
            s = vla.action_model.ddim_diffusion.ddim_sample_loop(fn, noise.shape, noise, clip_denoised=False,
                                                                 model_kwargs=mk, progress=False, device=z.device, eta=0.0)
        else:
            s = vla.action_model.diffusion.p_sample_loop(fn, noise.shape, noise, clip_denoised=False,
                                                         model_kwargs=mk, progress=False, device=z.device)
        if using_cfg:
            s, _ = s.chunk(2, dim=0)
        na = s[0].cpu().numpy()
        st = vla.get_action_stats(unnorm_key)
        mask = st.get("mask", np.ones_like(st["q01"], dtype=bool))
        hi, lo = np.array(st["q99"]), np.array(st["q01"])
        na = np.clip(na, -1, 1); na[:, 6] = np.where(na[:, 6] < 0.5, 0, 1)
        a = np.where(mask, 0.5 * (na + 1) * (hi - lo) + lo, na)
        return a, na

    def _compute_ig(self, pil, instruction, n_steps=3, top_ratio=0.3):
        """Faithful IG saliency over instruction tokens on the CogACT VLM (once/episode)."""
        from captum.attr import LayerIntegratedGradients
        vlm = self.m.vlm
        tok = vlm.llm_backbone.tokenizer
        dev = vlm.device
        pb = vlm.get_prompt_builder()
        pb.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        ids = tok(pb.get_prompt(), truncation=True, return_tensors="pt").input_ids.to(dev)
        ids = torch.cat([ids, torch.tensor([[29871, 2]], device=dev)], dim=1)
        pv = vlm.vision_backbone.image_transform(pil)
        pv = {k: v[None].to(dev) for k, v in pv.items()} if isinstance(pv, dict) else pv[None].to(dev)
        emb = vlm.llm_backbone.llm.model.embed_tokens
        # Autocast dtype must be a half precision type (bf16); the weights may now
        # be fp32 (faithful precision), so derive it from half_precision_dtype, not
        # the parameter dtype (torch.autocast rejects float32).
        dtype = vlm.llm_backbone.half_precision_dtype

        def fwd(input_ids, pvv):
            with torch.autocast("cuda", dtype=dtype):
                out = vlm(input_ids=input_ids, pixel_values=pvv,
                          attention_mask=torch.ones_like(input_ids), return_dict=True)
            return out.logits[:, -1, :].float()

        with torch.no_grad():
            tgt = fwd(ids, pv).argmax(-1).item()
        lig = LayerIntegratedGradients(fwd, emb)
        base = torch.full_like(ids, tok.pad_token_id or 0)
        att = lig.attribute(inputs=ids, baselines=base, target=tgt,
                            additional_forward_args=(pv,), n_steps=n_steps, internal_batch_size=1)
        sal = att.sum(-1).squeeze(0).abs().float().cpu().numpy()
        k = max(1, int(len(sal) * top_ratio))
        top = np.argsort(-sal)[:k]
        vlm.llm_backbone.llm.zero_grad()
        if emb.weight.grad is not None:
            emb.weight.grad = None
        return SaliencyInfo(token_saliency=sal, top_positions=np.sort(top), prompt_len=len(sal))

    def stats(self):
        return {"avg_kept_patches": float(np.mean(self.kept_hist)) if self.kept_hist else 256,
                "prune_pct": (1 - np.mean(self.kept_hist) / 256) * 100 if self.kept_hist else 0}
