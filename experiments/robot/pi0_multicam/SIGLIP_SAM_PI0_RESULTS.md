# Faithful T3R SigLIP-SAM pruning on pi0.5 + RoboTwin (multi-camera)

This completes the T3R *pruning* faithfulness study on pi0.5. Earlier work added the faithful **IG
attention bias**; the pruning *selection*, however, used a **norm-saliency proxy**, not T3R's real
**SigLIP-guided SAM** object localizer. Here we port the real SigLIP-SAM pruning, validate it, and
measure it head-to-head against norm / random / the paper baselines across a keep-ratio sweep.

## What real T3R pruning does (from OpenVLA-OFT `siglip_guided_sam.py`)
1. **SigLIP text↔patch weighted similarity** — a SigLIP text tower encodes the instruction (per-token
   features + noun/adjective token weights); a SigLIP vision tower encodes the image patches; weighted
   cosine → per-patch saliency.
2. **EfficientTAM/SAM object mask** — the saliency seeds positive/negative point prompts → EfficientTAM
   segments the object → patch-level mask.
3. **Keep** the object patches (+ SigLIP-ranked context fill to budget).

## The pi0 port (this work)
- **Blocker:** pi0/PaliGemma's SigLIP vision tower was co-trained with the Gemma LLM and exposes **no
  aligned SigLIP text tower**, so the faithful "VLA-vision + SigLIP-text" similarity is impossible.
  We therefore use a **fully standalone `google/siglip-so400m-patch14-384`** (both towers — the exact
  model the reference names) for the text↔patch similarity, verbatim weighted-similarity technique.
- **`sel="siglip"`** — SigLIP weighted-sim → top-k patches (interp 27×27 → pi0's 16×16 grid).
- **`sel="siglipsam"`** — SigLIP seed → **EfficientTAM** (efficienttam_s) point-prompt object mask →
  object patches + SigLIP context fill (faithful reference "sam_fill"). EfficientTAM loaded in-server
  via `sys.path` + hydra-core/iopath (installed `--no-deps`; torch untouched).
- Server passes the raw instruction to the model each infer (prompt hook in `serve_pi05_robotwin.py`).
- Selection returns ORIGINAL indices → the existing RoPE-position-preservation (`posfix`) keeps
  survivors on trained positions (on-manifold verified).

### Does SigLIP localize the object here? (standalone test, 3 real RoboTwin frames)
**No.** Weighted text↔patch similarity does not localize the target: top-15%-patch spread 11.6–12.1
(random-uniform ≈ 11.0; localized < 6.6), peaks off-object, heatmaps = texture noise. SAM seeded by a
noisy peak segments a background-biased region. Cause: SigLIP aligns the *pooled* embedding, not
per-patch; sim renders are OOD. (Verified with the faithful weighted technique + So400m, not a naive
cosine.) The full EfficientTAM pipeline nonetheless runs end-to-end and produces a valid 160/256 mask.

## Selection comparison at keep 0.625 (160 tok/cam, fixnoise, 12 ep/task) — base = 41.7
| selection | click_bell | click_alarmclock | **avg** |
|---|---|---|---|
| norm-saliency | 75.0 | 58.3 | **66.7** |
| SigLIP top-k | 58.3 | 41.7 | **50.0** |
| SigLIP-SAM (EfficientTAM) | 50.0 | 33.3 | **41.6** |
| random | 16.7 | 41.7 | **29.2** |

**Selection is high-leverage here** (37.5-pt spread) — this *corrects* an earlier claim that "norm ≈
random." SigLIP-SAM is *not* random (41.6 > 29.2 → its signal carries weak info) but only matches base
and trails norm by 25 pt. **SAM refinement hurts vs pure SigLIP top-k** (41.6 < 50.0): it commits
confidently to the wrong (background-biased) region.

## Keep-ratio sweep — avg SR (base = 41.7)
| keep | tok/cam | norm | SigLIP | SigLIP-SAM |
|---|---|---|---|---|
| 0.25 | 64 | 25.0 | — | — |
| 0.375 | 96 | 50.0 | 33.3 | 29.1 |
| 0.5 | 128 | 58.4 | 41.6 | 45.8 |
| 0.625 | 160 | 66.7 | 50.0 | 41.6 |
| 0.75 | 192 | 62.5 | 50.0 | 37.5 |

### Minimum tokens to match/beat base (41.7)
| method | min tokens | scaling above crossover |
|---|---|---|
| **norm-saliency** | **~96 (keep 0.375)** | keeps climbing → peak 66.7 @160 |
| SigLIP top-k | 128 (keep 0.5) | rises to 50, then plateaus |
| SigLIP-SAM | 128 (keep 0.5 = 45.8, its peak) | **non-monotonic — DEGRADES with more tokens** (→37.5 @192) |
| random | never (≤192) | — |

**norm is the most token-efficient and the only method that scales.** SigLIP needs 33% more tokens to
reach base and plateaus below norm. **SigLIP-SAM only works in a narrow ~128-token window and gets
*worse* with more budget** — SAM's mask + context-fill expands the wrong region, inverting the usual
"more tokens = better" trend.

## Faithful IG-bias keep-sweep (bias strength 0.5, for reference)
| keep | 0.375 | 0.5 | 0.625 | 0.75 |
|---|---|---|---|---|
| avg SR | 50.0 | 45.9 | 62.5 | 66.7 |

## Full leaderboard (avg both tasks)
norm-T3R 62–67 > **SigLIP top-k = TeamVLA-native 50.0** > base = **SigLIP-SAM 41.6** > ADP/FastV 25–37.5
> random 29.2. Even the "broken" SigLIP-SAM outranks the attention-pruning baselines (ADP/FastV) because
at 160 tokens it retains base-level info, whereas ADP/FastV's learned importance picks actively-worse
tokens.

## Conclusion
The faithful **SigLIP-SAM pruning does not transfer to pi0**: the SigLIP guidance signal doesn't
localize objects on this backbone/domain, so it underperforms the norm-saliency proxy by 25 pt at
matched budget, needs more tokens to reach base, and — uniquely — *degrades* as budget grows because
of SAM overcommitment. This is consistent with the IG-bias result: **T3R's training-free machinery
(SigLIP-SAM prune + IG bias) is tuned to OpenVLA-OFT's parallel-decoder + SigLIP-vision design and does
not port to pi0's flow-matching action expert + Gemma-fused SigLIP.** On pi0, plain norm-saliency
top-k is the better, more token-efficient, and better-scaling pruner.

## Files
- `t3r_pi0.py` — adds `_siglip_scores`, `_siglipsam_keep` (+ `_load_siglip`/`_load_etam`), `sel=siglip`/
  `siglipsam` branch. Retains the faithful IG bias.
- `serve_pi05_robotwin.py` — raw-instruction prompt hook.
- `cfg_siglipsam.txt`, `cfg_sweep.txt`, `cfg_fb05.txt` — configs.
- `siglipsam_results.csv`, `sweep_results.csv`, `fb05_results.csv` — raw success rates.
- Infra: `podput_big.py` / `podget_big.py` (chunked, md5-verified transfer — the old helpers truncated
  files > one channel window), `podhold.py` (long PTY keepalive that disables echo so the completion
  sentinel isn't mis-detected).
