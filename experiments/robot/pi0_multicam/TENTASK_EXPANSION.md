# pi0.5 + RoboTwin task-expansion: does the pruning win generalize? (CORRECTION)

The earlier pi0 result ("T3R-norm beats base by +25 at keep 0.625") was measured on **only 2 near-identical
click tasks** (click_bell, click_alarmclock). This expansion tests whether it holds across a broader,
more diverse task set — and it **substantially corrects the claim**.

## Hard constraint: motus pi0.5 is ~0% zero-shot on most RoboTwin tasks
Probed base SR (12 ep, fixed noise) on 12 new tasks. Most are **0% zero-shot**:

| task | base SR | usable? |
|---|---|---|
| place_empty_cup | 41.7 | ✓ |
| press_stapler | 33.3 (old per-obs-noise) / **0.0 (fixnoise)** | ✗ (0 under fixnoise) |
| grab_roller | 25.0 | ✓ |
| place_object_stand | 16.7 | ✓ (marginal) |
| handover_mic / turn_switch | 8.3 | ✗ |
| beat_block_hammer / move_can_pot / stack_blocks_two / stamp_seal | 0.0 | ✗ |

Only **3 new tasks** have usable base → **5-task set** with the 2 existing clicks. (This is *why* only 2
tasks were used originally — few RoboTwin tasks have non-trivial zero-shot base.)

## 5-task comparison at keep 0.625 (160 tok/cam, 12 ep/task, fixed noise)

| task | base | **norm** | ADP | FastV | Team | norm vs base |
|---|---|---|---|---|---|---|
| click_bell | 41.7 | **75.0** | 25.0 | 25.0 | 25.0 | +33 ✓ |
| click_alarmclock | 41.7 | **58.3** | 41.7 | 41.7 | 33.3 | +17 ✓ |
| place_empty_cup | 41.7 | **50.0** | 33.3 | 50.0 | 0.0 | +8 ✓ |
| grab_roller | 25.0 | 8.3 | 0.0 | 8.3 | **16.7** | **−17 ✗** |
| place_object_stand | 16.7 | 0.0 | **25.0** | 16.7 | 0.0 | **−17 ✗** |
| **AVG (5 tasks)** | **33.4** | **38.3** | 25.0 | 28.3 | 15.0 | **+4.9** |

## Findings
1. **Ranking holds:** norm 38.3 > base 33.4 > fastv 28.3 > adp 25.0 > team 15.0. The core story (norm-prune
   wins; attention/merge baselines fall below base) survives on the 5-task average.
2. **Margin collapses +25 → +4.9.** The original +25 was inflated by 2 favorable click tasks.
3. **norm is strongly task-dependent:** helps click/reach-press/place-cup (+8 to +33), **hurts grasp
   (grab_roller −17) and place-on-stand (place_object_stand −17)** — tasks needing fine visual detail that
   pruning destroys.
4. **Per-task winners swap:** norm wins clicks/cup; ADP wins place_object_stand (25 vs base 16.7); Team wins
   grab_roller. No method dominates every task.

## Honest revised conclusion
On pi0/RoboTwin, norm-saliency pruning gives a **real but modest (~+5) average gain that is highly
task-dependent** — not the universal +25 the two click tasks implied. The CogACT finding (T3R loses;
attention-methods win) is unaffected — it was well-sampled (96–189 ep across 14–15 tasks). The broader
lesson stands: token-reduction benefit on VLAs is backbone- AND task-dependent; there is no universal win.

Raw per-task numbers: `tentask_results.csv`. SQL `pi0_multicam` (place_empty_cup / grab_roller /
place_object_stand rows). Probe budget scope: keep 0.625 (sweet spot) — full 0.375/0.5 sweep on the new
tasks was not run (each cell ~8 min + pod must hold an SSH channel to progress, making the full 152-cell
matrix impractical).
