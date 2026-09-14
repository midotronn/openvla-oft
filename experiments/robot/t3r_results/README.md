# T3R → CogACT transfer: results & artifacts

This folder is a **safety snapshot** of the training-free token-reduction (T3R) transfer study
from OpenVLA-OFT to the **CogACT** VLA (DiT diffusion action head), evaluated **zero-shot on
SimplerEnv** (single third-person camera) — Google Robot + Bridge/WidowX — on a RunPod L40S (fp32).

## Code (in `experiments/robot/`, the parent dir)
- `cogact_loader.py` — builds CogACT-Base (Prismatic VLM + DiT). **fp32 by default** (`COGACT_BF16=1`
  opt-in); the bf16→fp32 fix was critical (bf16 corrupted the DiT's tiny action deltas).
- `cogact_simpler_policy.py` — SimplerEnv inference wrapper (Google Robot + WidowX Bridge).
- `t3r_cogact.py` — the T3R port: SigLIP-SAM vision-token **prune** + integrated-gradients saliency
  **bias**, with the RoPE-position-preserving LLM hook.
- `baselines_cogact.py` — faithful baselines: **ADP** (layer-0 QK-importance + action-aware dynamic
  gate) and **TeamVLA** (faithful two-stage: SigLIP-seed → spatial-expand → context prune, then
  soft-bipartite merge). Env-knob configurable.
- `fastv_cogact.py` — faithful **FastV** (official pkunlp-icler algorithm, arXiv 2403.06764),
  token-masking variant, adapted to transformers 4.40.1 LlamaModel.

## Data
- `*_results.json` — raw per-unit success-rate results (the source of truth):
  - `bench_results.json` (base, Google), `bridge_results.json` (base, Bridge)
  - `k60_results.json` / `bridge_k60_results.json` — t3r_orig @40% (154 tok)
  - `native_results.json` — native ADP/TeamVLA + weakened topK sweep
  - `faithful_results.json` / `bridge_faithful_results.json` — faithful ADP/TeamVLA + adp40 (both suites)
  - `fastv_google_results.json` / `fastv_bridge_results.json` — FastV @40% (both suites)
  - `sweep_af192_google_results.json` — lighter-prune (25%/192-tok) sweep, adp192+fastv192 Google (partial)
  - `multitask_results.json` — multitask ablation
- `sql_csv/*.csv` — curated/aggregated analysis tables (bench, native_faithful, fastv, pareto, etc.).
- `docs/*.md` — full write-ups. **Start with `BOTH_SUITE_FAITHFUL.md`** (the 5-method matched-budget
  comparison) and `BASELINE_FAITHFULNESS.md` (the faithfulness audit vs primary sources).
- `drivers/*.py` + `drivers/*.sh` — the evaluation drivers / orchestration used to produce the data.

## Headline result (matched 154-token / 40%-prune budget, both suites)
| method | tok | Google overall | Bridge info-3 |
|---|---|---|---|
| adp40 (ADP QK) | 154 | 81.6 | 19.4 |
| fastv (FastV) | 154 | 80.6 | 20.8 |
| **t3r_orig @40%** | 154 | **73.4** | **12.5** |

On **single-camera** SimplerEnv, T3R's SigLIP-SAM pruning is the lowest of the pruning baselines at a
matched budget. This is consistent with the hypothesis that T3R's design targets **multi-camera**
inter-view redundancy (its native OpenVLA-OFT/LIBERO setup keeps a full wrist camera). The dual-camera
verification lives in the OpenVLA-OFT/LIBERO work; the next step (branch `t3r-pi0`) tests a different
backbone (π0) on a multi-camera benchmark (ALOHA-sim) to probe generalization.

_Snapshot committed for safety before the π0 pivot._
