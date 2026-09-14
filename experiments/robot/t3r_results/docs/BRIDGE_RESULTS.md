# WidowX/Bridge benchmark — matched 64-token budget (75% prune)

CogACT-Base zero-shot, policy_setup=widowx_bridge. Methods: base (256 tok), t3r_orig (SigLIP-SAM
prune64 + IG attention bias), adp64 (QK-importance @64), team64 (expand-merge @64). t3r_dit
excluded per request. 24 episodes/task.

## Per-task SR (%)

| task               | base | t3r_orig | adp64 | team64 | n  |
|--------------------|------|----------|-------|--------|----|
| stack_cube         | 0.0  | 0.0      | 4.2   | 0.0    | 24 |
| carrot_on_plate    | 25.0 | 0.0      | 25.0  | 8.3    | 24 |
| spoon_on_towel     | 25.0 | 12.5     | 12.5  | 25.0   | 24 |
| eggplant_in_basket | 25.0 | 4.2      | 0.0   | 8.3    | 24 |

## Aggregates (SR %)

| Method   | Tokens | All 4 | Excl. stack_cube (informative 3) |
|----------|--------|-------|----------------------------------|
| base     | 256    | 18.8  | 25.0                             |
| team64   | 64     | 10.4  | 13.9                             |
| adp64    | 64     | 10.4  | 12.5                             |
| t3r_orig | 64     | 4.2   | 5.6                              |

## Read-out

1. **CogACT-Base is weak on Bridge zero-shot (~25% on the informative tasks)** — Bridge is harder
   for it than the Google Robot suite. stack_cube ≈ 0 for all methods (uninformative).
2. **Bridge is very pruning-sensitive:** at 64 tokens every method drops well below base.
3. **Ranking matches the Google Robot benchmark:** team64 (13.9) ≈ adp64 (12.5) > t3r_orig (5.6).
   **T3R is again the weakest — roughly half the attention-based baselines.**
4. Per-task the best pruner varies (carrot: adp64 ties base; spoon: team64 ties base; eggplant:
   all collapse), but **T3R never leads on any Bridge task.**

## Combined verdict (Google Robot + Bridge)
Across BOTH embodiments, at a matched 64-token / 75%-prune budget, the attention-based baselines
(ADP, TeamVLA) consistently beat T3R's SigLIP-SAM pruning, often by ~2x. T3R is competitive only
on single-object grasping (coke); it does not transfer to spatial/multi-object manipulation on
CogACT's single-cognition-token architecture.

Caveats: 24 ep/task (+/-~0.09 CI); low absolute base SR on Bridge makes ratios noisy; single robot
position per task (not the full multi-position protocol).
