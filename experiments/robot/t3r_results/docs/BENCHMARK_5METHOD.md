# 5-method benchmark on SimplerEnv Google Robot — matched 64-token budget (75% prune)

CogACT-Base zero-shot. All PRUNING methods forced to the SAME budget (25% keep = 64 tokens) for
a fair comparison. urdf=None appearance. Methods:
- **base**     : no reduction (256 tokens)
- **t3r_dit**  : T3R full = SigLIP-SAM prune(64) + DiT-conditioning bias (alpha 0.15)
- **t3r_orig** : T3R full = SigLIP-SAM prune(64) + original IG attention bias (str 1.0)
- **adp64**    : ADP forced to 64 (QK text->vision importance @ layer0, keep0.25, action-gate OFF)
- **team64**   : TeamVLA forced to 64 (expand + bipartite merge, topk=64)

## Per-unit SR (%)

| unit             | base  | t3r_dit | t3r_orig | adp64 | team64 | n  |
|------------------|-------|---------|----------|-------|--------|----|
| coke_upright     | 80.0  | 92.0    | 84.0     | 80.0  | 84.0   | 25 |
| coke_lr_switch   | 88.0  | 56.0    | 84.0     | 80.0  | 76.0   | 25 |
| coke_laid_vert   | 60.0  | 52.0    | 52.0     | 68.0  | 60.0   | 25 |
| move_near        | 66.7  | 26.7    | 28.3     | 55.0  | 55.0   | 60 |
| drawer_open_top  | 55.6  | 22.2    | 11.1     | 55.6  | 44.4   | 9  |
| drawer_open_mid  | 88.9  | 33.3    | 33.3     | 77.8  | 77.8   | 9  |
| drawer_open_bot  | 44.4  | 0.0     | 0.0      | 66.7  | 55.6   | 9  |
| drawer_close_top | 100.0 | 33.3    | 55.6     | 100.0 | 100.0  | 9  |
| drawer_close_mid | 77.8  | 88.9    | 77.8     | 100.0 | 100.0  | 9  |
| drawer_close_bot | 88.9  | 66.7    | 77.8     | 100.0 | 88.9   | 9  |
| put_in_drawer    | 0.0   | 0.0     | 0.0      | 0.0   | 0.0    | 9  |

## Aggregates (SR %, excluding put_in_drawer = uninformative, all 0)

| Method   | Tokens | Coke (3-orient) | move_near | Drawer (6) | **Overall** |
|----------|--------|-----------------|-----------|------------|-------------|
| adp64    | 64     | 76.0            | 55.0      | 83.4       | **78.3**    |
| base     | 256    | 76.0            | 66.7      | 75.9       | **75.0**    |
| team64   | 64     | 73.3            | 55.0      | 77.8       | **74.2**    |
| t3r_orig | 64     | 73.3            | 28.3      | 42.6       | **50.4**    |
| t3r_dit  | 64     | 66.7            | 26.7      | 40.7       | **47.1**    |

## Read-out (honest)

1. **At 75% prune, ADP is the best pruning method — it even EXCEEDS base overall (78.3 vs 75.0).**
   Its QK text->vision attention importance keeps the task-relevant tokens; on drawer it beats
   base (83.4 vs 75.9) by dropping distractors. TeamVLA (merging) ≈ base (74.2).

2. **Both T3R variants collapse to ~half the baselines (47-50 vs 74-78) at the same 64-token
   budget.** T3R is competitive only on coke (single object). On the spatial/multi-object tasks
   (move_near, drawer open) it fails badly (0-33%). SigLIP-SAM's object-focused selection throws
   away the scene context these tasks need; a single "cognition token" then can't recover it.

3. **t3r_dit vs t3r_orig:** the original IG attention bias (50.4) slightly beats the DiT-injection
   variant (47.1). t3r_dit is more volatile (coke_upright 92 but coke_lr 56).

4. **Coke is misleading in isolation:** on coke all methods look close (66-76). Only the full task
   suite reveals T3R's weakness — coke is a single centered object where object-only pruning is fine.

5. **put_in_drawer:** CogACT-Base scores 0 zero-shot (open drawer + place apple, 200 steps) for all
   methods — uninformative here.

## Bottom line
At an aggressive, matched 64-token budget across the full Google Robot suite, **the attention-based
baselines (ADP, TeamVLA) clearly beat T3R**, and ADP even beats the unpruned base. T3R's SigLIP-SAM
object pruning does not transfer to the spatial/multi-object tasks on CogACT's single-cognition-token
architecture. This is the decisive multi-task picture the coke-only experiments hid.

Caveats: drawer/put_in_drawer n=9 (coarse, +/-0.11/episode); urdf=None only (4-URDF aggregate is a
4x follow-up); Bridge/WidowX suite not run (separate embodiment/setup).
