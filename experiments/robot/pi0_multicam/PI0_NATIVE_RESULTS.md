# π0.5 + RoboTwin 2.0 — PAPER-NATIVE Token-Reduction Comparison

Companion to `PI0_MULTICAM_RESULTS.md` (which used **matched** per-camera budgets). Here every method
runs at its **own paper-native default configuration** — different token budgets per method, each
"as designed" in its original paper. Same setup otherwise (π0.5 motus checkpoint, RoboTwin 3-cam,
deterministic fixed-noise paired eval, 12 ep/cell, informative tasks click_bell + click_alarmclock).

## Paper-native configurations used
| method | paper | native config | tok/cam (observed) |
|--------|-------|---------------|--------------------|
| **T3R** | SigLIP-SAM (own) | **NO keep ratio** — adaptive object-mask threshold. pi0 analog: keep norm-saliency > mean+1·std (variable count) | **~10–37** (~5–15%) |
| **FastV** | arXiv 2403.06764 (ECCV'24) | **K=2, R=50%** (keep 50%), rank by last prompt-token received attention | 128 (50%) |
| **VLA-ADP** | arXiv 2509.22093 (ICLR'26) | **keep 0.75, layer-0** text→vision QK, **+ dynamic motion gate** (Adjacent-Extrema Rule: prune coarse-motion steps, full-vision fine-manipulation; cold-start 2, reset after 3 consec) | 192 when gated-on / 256 when gated-off |
| **TEAM-VLA** | arXiv 2512.09927 | **two-stage**: similarity seed → K=3 conv spatial expand → context u=0.25 → soft bipartite **merge to top-M=80** (τ=1) | 80 |

Native selection notes: ADP/FastV use their real learned attention (transfers to pi0). T3R and
Team-VLA natively use SigLIP language–image cosine, which does **not** transfer to π0's fused Gemma
space (verified broken earlier) → both use **norm saliency** for token selection/seeding; TeamVLA's
Stage-2 feature-space merge is unchanged/faithful. TeamVLA's native merge layer is 16 (mid-LLM); we
apply the merge at embed (pre-LLM), the same simplification documented for the CogACT port.

## RESULTS (paired, deterministic, 12 ep/cell)
| method | tok/cam | click_bell | click_alarmclock | **AVG** | vs base |
|--------|---------|-----------|------------------|---------|---------|
| **TEAM-VLA native** (top-80) | 80 | **58.3** | 41.7 | **50.0** | **+8.3  ★BEST** |
| base (full) | 256 | 41.7 | 41.7 | 41.7 | — |
| VLA-ADP native (keep0.75+gate) | 192–256 | 33.3 | 41.7 | 37.5 | −4.2 |
| FastV native (R=50%) | 128 | 16.7 | 50.0 | 33.4 | −8.3 |
| T3R native (no-ratio) | ~10–37 | 16.7 | 16.7 | 16.7 | −25.0 |

## Findings
1. **At paper-native configs, TEAM-VLA wins** (50.0 avg, beats base). Its information-preserving
   two-stage merge at the native top-80 anchors is the sweet spot on this backbone — merging (vs hard
   dropping) retains signal while cutting tokens ~69% (256→80).
2. **T3R's native "no keep ratio" is too aggressive on π0/RoboTwin.** The adaptive norm-saliency
   threshold (mean+1·std) keeps only ~10–37 tokens/camera (~5–15%) because the per-token norm
   distribution on real RoboTwin frames is sharply peaked. That extreme reduction (the *fewest* tokens
   of any method — most efficient) costs the most SR (16.7). Note this ~15% coincides with T3R's own
   paper default `keep_ratio=0.15` — so it is genuinely T3R-native, just aggressive here.
3. **VLA-ADP's dynamic motion gate is conservative** — it holds full vision on most steps and prunes
   (to keep-0.75) only during high-displacement segments, so it tracks base closely (37.5 vs 41.7).
4. **FastV (R=50%) is task-dependent**: helps click_alarmclock (50.0 > base) but hurts click_bell.

## The two comparisons together (the real story)
| regime | winner | why |
|--------|--------|-----|
| **Matched budget** (all keep 0.75 / 0.5) | **T3R (norm)** — 62.5 / 58.4 avg, beats all | at *equal* token count, norm-saliency selection keeps the best tokens |
| **Paper-native** (each own config) | **TEAM-VLA** (top-80 merge) — 50.0 | at its native budget, info-preserving *merge* > hard-drop; T3R's native threshold over-prunes π0 |

Take-away: **T3R is the strongest token *selector* at a matched budget**, but its native
no-fixed-ratio threshold is mis-calibrated for π0's peaked norm distribution (over-prunes). TEAM-VLA's
merge is the most robust *native* operating point on this backbone. Both are honest, complementary
results; neither is a single-number "T3R wins/loses."

## Caveats
- 2 informative tasks × 12 ep (motus's zero-shot RoboTwin coverage is uneven). Ranking is
  deterministic/stable (base reproduces 41.7 exactly); absolute SRs are indicative.
- Native token budgets differ per method by design — this compares each method's *own* efficiency–SR
  operating point, not selection quality at fixed budget (that's the matched-budget table).
- T3R/TeamVLA selection uses norm saliency (π0 lacks a usable SigLIP text encoder); ADP/FastV use
  their real attention. The T3R adaptive threshold multiplier (mean+1·std) is a design choice; a
  smaller multiplier would keep more tokens and raise SR toward the matched-budget result.

## Artifacts
- `t3r_pi0.py` — adds `_thresh_idx` (T3R no-ratio), last-token `_fastv_importance`, `_team_twostage`
  (native 2-stage), `_adp_gate_prune` (motion gate). Runtime-json knobs: `thresh`, `team_native`,
  `team_topk/k/u/tau`, `adp_gate`.
- `cfg_native.txt` — the 5 native configs. Pod: `/workspace/native_results.csv`.
- SQL `pi0_multicam` (native rows) + `paper_native_cfg` (the researched defaults + citations).
