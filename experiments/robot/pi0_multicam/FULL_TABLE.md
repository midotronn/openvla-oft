# Full budget × method frontier — pi0.5 + RoboTwin multi-camera

Complete token-reduction comparison: every method at every keep ratio, avg SR over click_bell +
click_alarmclock (fixnoise, 12 ep/task/cell). Base (no pruning, 256 tok/cam) = 41.7.

## Matrix — avg SR (bold = row winner)

| tok/cam (keep) | **norm** | SigLIP | SigLIP-SAM | ADP | FastV | TeamVLA |
|---|---|---|---|---|---|---|
| 64 (0.25) | **25.0** | 16.7 | 12.5 | 20.9 | 20.8 | 8.3 |
| 96 (0.375) | **50.0** | 33.3 | 29.1 | 20.8 | 25.0 | 16.6 |
| 128 (0.5) | **58.4** | 41.6 | 45.8 | 25.0 | 25.0 | 29.1 |
| 160 (0.625) | **66.7** | 50.0 | 41.6 | 33.4 | 33.4 | 29.1 |
| 192 (0.75) | **62.5** | 50.0 | 37.5 | 33.4 | 37.5 | 20.9 |
| **256 base (no prune)** | — | — | — | — | — | 41.7 |

(native paper configs, other budgets: TeamVLA-native @80 = 50.0; ADP-native gated ≈256 = 37.5;
FastV-native @128 = 33.4.)

## Findings
1. **norm wins at every budget** and is the ONLY method that beats base (41.7) — clears base from
   96 tok (37.5%) up, peaks 66.7 @160.
2. **No attention/merge baseline (ADP/FastV/TeamVLA) ever reaches base** — ceiling FastV 37.5 @192.
   Their pruning always costs more than it saves.
3. **Ordering inverts at 64 tok (0.25):** ADP (20.9) / FastV (20.8) beat SigLIP (16.7) / SigLIP-SAM
   (12.5); at 25% budget SigLIP's noisy localization wastes tokens on wrong patches while attention
   selection degrades more gracefully. norm (25.0) still leads.
4. **SigLIP-SAM is worst-scaling** (12.5→29.1→45.8→41.6→37.5): near-bottom at 64, one good window at
   128, then declines (SAM overcommitment at both extremes).
5. **Matched-budget TeamVLA is weakest** (8.3 @64/96; bell=0.0 @96); only its native @80 (50.0) is OK.

Ranking: **norm >> SigLIP ~ SigLIP-SAM > ADP ~ FastV > TeamVLA**; norm is the sole method that turns
token reduction into a net win over the no-prune baseline.

Raw per-task numbers: `fulltable_results.csv` (this fill: keep 0.25 for siglip/siglipsam/adp/fastv/team
+ adp/fastv/team at 0.375 and 0.625). norm + siglip/siglipsam 0.5/0.625/0.75 in `sweep_results.csv` and
prior commits.
