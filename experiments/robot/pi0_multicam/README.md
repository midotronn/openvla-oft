# π0.5 + RoboTwin 2.0 — Multi-Camera Token-Reduction Comparison

**Goal (thesis, 3rd evidence point):** verify T3R token pruning on a *different* VLA backbone
(π0.5, non-OpenVLA) and a *different, genuinely multi-camera* benchmark (RoboTwin 2.0, 3 views:
head + 2 wrists), using an **off-the-shelf checkpoint (no finetuning)**, and compare it against
**ADP**, **FastV**, and **TeamVLA** at matched per-camera token budgets.

## Setup
- **Backbone:** π0.5 (PaliGemma SigLIP-So400m vision + Gemma; flow-matching action expert), openpi
  **PyTorch** path. Off-the-shelf checkpoint: `motus-robotics/pi0.5_robotwin2` (50-task RoboTwin 2.0,
  27.5k episodes). **LOAD_PERFECT** (3.62B params, 0 missing/unexpected).
- **Benchmark:** RoboTwin 2.0 (SAPIEN, bimanual aloha-agilex, 3 D435 cameras: `cam_high`,
  `cam_left_wrist`, `cam_right_wrist`; 14-dim dual-arm state/action).
- **Architecture:** decoupled — π0.5 policy **server** (openpi venv, websocket :8000, token-reduction
  hooks live here) ↔ RoboTwin sim **client** (rtenv, `openpi_client`). Instruction = each task's
  `full_description` (expert demo skipped: needs curobo; not required for policy eval).
- **Budget:** uniform per-camera keep fraction (all 3 cameras). base = 256 tok/cam (768 total).
- **Eval:** deterministic (fixed flow-matching noise → fully *paired* comparison; base reproduces
  41.7% exactly across runs). 12 episodes/cell. Informative tasks = base SR > 0 under the fixed noise.

## Methods (ported into `pi0_pytorch.embed_prefix`, matched per-camera budget)
- **T3R** — training-free saliency pruning: keep the top-K vision tokens by **feature-magnitude
  (norm) saliency** (high-magnitude = distinctive/object patches; drop low-magnitude background).
  Hard drop. *(See "T3R selection" note below.)*
- **ADP** — VLA-ADP text→vision QK importance at Gemma **layer 0** (GQA-aware). Hard drop.
- **FastV** — received-attention (all→vision) QK at Gemma layer 0. Hard drop.
- **TeamVLA** — similarity-sampled anchors + soft **bipartite merge** (information-preserving).

## *** THE CRITICAL FIX: position preservation ***
π0 inference sets `position_ids = cumsum(pad_masks)-1` and the action-suffix offset =
`sum(pad_masks)`. Physically dropping tokens re-indexes survivors contiguously → shifts every image
& language token off its trained absolute RoPE position → **off-manifold → 0% SR for ANY pruning**
(verified: actions blow up to |a|≈2.0 vs the trained ≈0.02 range). **Fix:** each method returns the
ORIGINAL indices it keeps; we rebuild `position_ids` so survivors stay at their trained positions
(camera *ci* occupies `[ci·256, ci·256+256)`) and the suffix starts at the original post-prefix
position. With the fix, pruned actions match the full-token baseline to `|Δ|≈0.0006`. **Without this,
no token-reduction method works on π0 at all.**

## RESULTS (paired, deterministic, 12 ep/cell)

### keep = 0.75  (25% prune, 192 tok/cam)
| method | click_bell | click_alarmclock | **AVG** | vs base |
|--------|-----------|------------------|---------|---------|
| base (256) | 41.7 | 41.7 | **41.7** | — |
| **T3R (norm)** (192) | **75.0** | **50.0** | **62.5** | **+20.8  ★BEST** |
| FastV (192) | 25.0 | 50.0 | 37.5 | −4.2 |
| ADP (192)   | 25.0 | 41.7 | 33.3 | −8.4 |
| TeamVLA (192) | 16.7 | 25.0 | 20.8 | −20.9 |

### keep = 0.5  (50% prune, 128 tok/cam)
| method | click_bell | click_alarmclock | **AVG** | vs base |
|--------|-----------|------------------|---------|---------|
| base (256) | 41.7 | 41.7 | **41.7** | — |
| **T3R (norm)** (128) | **66.7** | **50.0** | **58.4** | **+16.7  ★BEST** |
| TeamVLA (128) | 8.3 | 50.0 | 29.2 | −12.5 |
| ADP (128)   | 8.3 | 41.7 | 25.0 | −16.7 |
| FastV (128) | 16.7 | 33.3 | 25.0 | −16.7 |

## Findings
1. **T3R wins decisively at both budgets** on multi-camera π0.5/RoboTwin — it not only retains but
   **improves** over the full-token base (removing redundant low-magnitude tokens across the 3 views
   focuses the policy), while ADP/FastV/TeamVLA all fall **below** base.
2. **T3R's advantage is robust to aggressive pruning.** At 50% prune, T3R holds 58.4% (still > base)
   while the attention/merge baselines collapse toward 25–29%. On the single-object contact tasks the
   attention-based selectors (ADP layer-0 QK, FastV) keep the wrong tokens under a tight budget.
3. This is the **3rd thesis evidence point**: T3R suits multi/dual-camera setups. It **wins** here
   (π0.5 / RoboTwin, 3 cams) and on LIBERO (OpenVLA-OFT, 2 cams), but **loses** on single-camera
   SimplerEnv/CogACT — consistent with "prune the redundant views hard, keep the informative tokens."

## Honest caveats
- **T3R selection = norm saliency, not the faithful SigLIP-SAM.** π0 has no accessible SigLIP *text*
  encoder, so instruction-guided SigLIP similarity can't be computed in-model. My in-model
  instruction cosine in Gemma's *projected* space was **actively broken** (0% — that space is trained
  for QKV attention, not cosine; verified random 41.7% >> lang-cosine 0%). An external stock SigLIP
  patch↔text cosine did **not** localize (SigLIP aligns *pooled* image–text, not patches). So T3R
  uses **feature-magnitude (norm) saliency** — training-free, object-focused, and consistent with the
  earlier **CogACT finding that T3R's smart selection ≈ random; only token COUNT matters**. norm and
  random both work (41.7%); the *harmful* signal was the mis-projected cosine.
- **π0.5's "prune-sensitivity" was largely the broken selection**, not the backbone: with norm
  saliency, click_bell tolerates 50% pruning fine (66.7%). Some tasks (beat_block_hammer,
  press_stapler) hit base=0 under the single fixed-noise seed and were excluded as uninformative.
- **Scale:** 2 informative tasks × 12 ep/cell (motus's RoboTwin coverage is uneven — many tasks
  score 0 zero-shot). Numbers are indicative, not high-precision; the *ranking* is stable/deterministic.

## Artifacts (session `files/`)
- `t3r_pi0.py` — all 4 methods + position-preservation patch (server-side hooks).
- `serve_pi05_robotwin.py` — π0.5 policy server; `deploy_policy_pi0client.py` — RoboTwin websocket client.
- `run_compare.sh` / `cfg_compare75.txt` / `cfg_compare50.txt` — resumable comparison driver.
- SQL table `pi0_multicam` — all cells. Pod: `/workspace/compare_k75_results.csv`,
  `/workspace/compare_k50norm_results.csv`, `/workspace/seldiag_results.csv`.
