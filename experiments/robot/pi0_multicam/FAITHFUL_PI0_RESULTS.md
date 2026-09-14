# Faithful T3R on pi0.5 + RoboTwin (multi-camera): IG attention bias

This completes the T3R faithfulness audit on the pi0.5 backbone. The prior pi0 comparisons used a
**norm-saliency proxy** for token selection and had **no** integrated-gradients (IG) attention bias —
i.e., they exercised T3R's *pruning* but not its *biasing*. Here we add the real T3R **IG attention
bias** and measure whether it helps on a genuinely multi-camera benchmark.

## What "faithful" means for T3R (from the original OpenVLA-OFT/CogACT code)
Real T3R has two components:
1. **Pruning** — SigLIP-guided SAM object mask → keep ~top salient visual patches.
2. **IG attention bias** — `captum.LayerIntegratedGradients` over the instruction tokens (target =
   next-token / action logit), computed once per episode → additive SDPA bias on the action-query rows,
   steering them toward instruction-salient key positions (layers 8–23, strength 2.0).

### pi0.5 port (this work)
- **IG saliency** (`_ig_saliency` in `t3r_pi0.py`): manual integrated gradients (captum absent in the
  openpi venv) over the **language-token embeddings**, baseline = 0, `n_steps=8`. Since pi0 is a
  flow-matching action model with **no output logits**, the IG target is `||predicted action||²` from a
  1-step denoise (vs CogACT's argmax logit). Returns per-language-token saliency `[L]`. Cached per prompt.
- **Attention bias** (in `denoise_step`): the top-`bias_topratio` (0.25) salient language tokens map to
  their prefix key columns; an additive `+strength·weight` term is added to the 4-D attention mask on the
  **action-query rows only** (last `action_horizon`=32 suffix rows) at currently-allowed key positions.
  Mirrors T3R's `_biased_sdpa`. Skipped during IG attribution (`_ig_running` guard).
- **Selection stays norm-saliency**: pi0's Gemma-space instruction cosine and stock SigLIP *patch*↔text
  cosine were both verified NOT to localize on pi0 (SigLIP aligns the *pooled* embedding, not patches),
  so the pruning selection uses norm saliency (= random-competitive, established earlier). The IG bias is
  the faithful component that this port adds and tests.

### Validation (smoke test, dummy 3-cam obs)
- IG saliency finite & non-uniform: `L=200, nonzero=71, range 4.0e-3 … 29.6` (discriminates tokens).
- Bias active & on-manifold: prune-only action `|a|max=0.0155` → prune+IG-bias `0.0205`
  (base `0.0189`, trained ~0.02); `mean|Δ|=0.0029`. No blow-up, autograd works through the server.

## Result — does the faithful IG bias help? (keep 0.625 = 160 tok/cam, fixed noise, 12 ep/cell)

Fully-paired A/B on the identical build. Prune-only reproduced the prior sweep peak (66.7%) exactly.

| bias strength | click_bell | click_alarmclock | **avg** |
|---|---|---|---|
| **0.0 (prune-only)** | 75.0% | 58.3% | **66.7%** |
| 0.5 | 75.0% | 50.0% | 62.5% |
| 1.0 | 50.0% | 41.7% | 45.8% |
| 2.0 (T3R paper default) | 50.0% | 41.7% | 45.8% |

**The IG attention bias is monotonically harmful** — neutral at best (str0.5 on click_bell), never
helpful at any tested strength. At the paper-default strength 2.0 it costs **−20.8 pt** avg.

## Conclusion
On pi0.5 + RoboTwin (multi-camera), as on CogACT + SimplerEnv (single-camera):
- **T3R's pruning transfers.** Norm-saliency prune to keep 0.625 (160 tok/cam) scores 66.7% avg, far
  above the zero-shot base (~41.7%). Token *count* is what matters; the selection is
  random-competitive.
- **T3R's IG attention biasing does NOT transfer.** It is neutral-to-harmful at every strength on both
  backbones. Root cause is architectural: T3R's biasing was designed for OpenVLA-OFT's **parallel L1
  decoder**, where sharpening action→token attention helps. pi0's **flow-matching action expert** (like
  CogACT's DiT) is trained to denoise conditioned on the full prefix; an additive attention bias
  perturbs that learned conditioning off-manifold and degrades success.

This is the third, consistent evidence point: **T3R = a good training-free *pruner*, but its
IG-bias component is decoder-architecture-specific and does not port to diffusion/flow-matching VLAs.**

## Files
- `t3r_pi0.py` — faithful module (`_ig_saliency`, IG override in `embed_prefix`, bias setup in
  `sample_actions`, additive bias in `denoise_step`).
- `cfg_faithful.txt`, `cfg_biassweep.txt` — the A/B + strength-sweep configs.
- `faithful_results.csv`, `biassweep_results.csv` — raw success rates.
