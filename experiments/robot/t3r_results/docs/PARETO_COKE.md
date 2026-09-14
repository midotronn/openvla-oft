# CogACT pruning Pareto: aggressiveness vs SR (coke_can, 75-trial, single session)

Task: pick_coke_can, 3 orientations (upright/lr_switch/laid_vertically) × 25 = **75 trials**
per point. CogACT-Base zero-shot. ALL points measured in ONE process/session for direct
comparability. "ours" = faithful T3R raw SigLIP-SAM hard pruning (no SAM-fill, no bias).

## The curve (avg SR over 75 trials)

| Method        | Tokens | Prune % | up   | lr   | lv   | **AVG SR** |
|---------------|--------|---------|------|------|------|------------|
| base          | 256    | 0%      | 0.80 | 0.84 | 0.64 | **76.0**   |
| ADP (native)  | 192*   | 25%*    | 0.96 | 0.92 | 0.64 | **84.0**   |
| ours keep0.75 | 192    | 25%     | 0.76 | 0.88 | 0.60 | **74.7**   |
| ours keep0.50 | 128    | 50%     | 0.92 | 0.76 | 0.60 | **76.0**   |
| TeamVLA(nat.) | 80     | 69%     | 0.80 | 0.80 | 0.60 | **73.3**   |
| ours keep0.31 | 80     | 69%     | 0.80 | 0.72 | 0.52 | **68.0**   |
| ours keep0.25 | 64     | 75%     | 0.92 | 0.64 | 0.44 | **66.7**   |
| ours keep0.15 | 38     | 85%     | 0.76 | 0.56 | 0.52 | **61.3**   |

*ADP's 192 is a per-step AVERAGE from its action-aware gate: it keeps all 256 tokens in
fine-manipulation (grasp) frames and prunes only in coarse-motion frames. So it is not a
uniform-192 method — it pays full cost exactly when it matters (grasp) and saves elsewhere.

## SR vs prune %  (ours = O, baselines = *)

```
SR%
86 |                    * ADP(25%)
84 |                    *
82 |
80 |
78 |
76 | O base(0%)   O(50%)
74 |          O(25%)
72 |
70 |
68 |                              O(69%)
66 |                                 O(75%)
64 |                              * TeamVLA(69%)
62 |                                      O(85%)
60 +----+----+----+----+----+----+----+----+---> prune%
   0    10   20   30   40   50   60   70   80
```

## Read-out (honest)

1. **Free efficiency to 50% prune.** ours holds base SR (76.0) all the way to **128 tokens
   (50% prune)** — keep0.75=74.7, keep0.50=76.0 both ≈ base 76.0. Half the visual tokens for
   no measurable SR loss.

2. **Both baselines beat pure pruning at their budgets (this session):**
   - At **192 tokens**: ADP **84.0** ≫ ours 74.7 (+9.3). ADP's dynamic gate preserves the
     grasp-critical frames at full resolution — it even beats base. This is a real advantage
     of *phase-adaptive* pruning over *uniform* pruning.
   - At **80 tokens**: TeamVLA **73.3** > ours 68.0 (+5.3). Merging retains dropped-token info
     that hard pruning discards.

3. **Aggressive tail:** ours degrades gracefully but sub-linearly — 69%→68.0, 75%→66.7,
   85%→61.3. Even at **85% prune (38 tokens)** upright stays 0.76 (≈ base); the loss is
   concentrated in the hard rotated/laid poses.

4. **Variance caveat:** 25 trials/orientation → ~±0.10 CI per cell; laid_vertically is the
   noisiest axis. ADP's 84.0 this session vs 78.7 a prior session shows run-to-run spread.
   Trends are reliable; single-point gaps <5 pts are within noise.

## Bottom line
Faithful T3R hard-pruning is **Pareto-competitive at mild/moderate pruning** (matches base to
50% prune) but is **dominated at high pruning** by ADP (phase-adaptive gating) and TeamVLA
(merging). To win at high pruning, the mechanisms that help are exactly the ones outside pure
SigLIP-SAM dropping: *adaptivity* (keep more during grasp) and *merging* (retain dropped info).
Within faithful pruning, the honest efficiency claim is "50% token reduction at no SR cost."
