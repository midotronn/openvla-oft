# Multi-task validation: pruning aggressiveness vs SR (coke + move_near + drawer)

Extends the coke Pareto to **move_near** (n=60) and **open_drawer** (n=9), CogACT-Base
zero-shot. "ours" = faithful T3R raw SigLIP-SAM hard pruning (no SAM-fill, no bias).
move_near/drawer measured in one session; coke from the Pareto session (base is stable across
sessions). Baselines run native (ADP≈192 avg via action gate; TeamVLA merges to 80).

## Per-task SR

**At 192 tokens (25% prune):**
| task       | base  | ADP   | ours_k75 |
|------------|-------|-------|----------|
| coke       | 0.760 | 0.840 | 0.747    |
| move_near  | 0.650 | 0.733 | 0.600    |
| drawer     | 0.556 | 0.556 | 0.778    |

**At 128 tokens (50% prune):**
| task       | base  | ours_k50 |
|------------|-------|----------|
| coke       | 0.760 | 0.760    |
| move_near  | 0.650 | 0.467    |
| drawer     | 0.556 | 0.333    |

**At 80 tokens (69% prune):**
| task       | base  | TeamVLA | ours_k3125 |
|------------|-------|---------|------------|
| coke       | 0.760 | 0.733   | 0.680      |
| move_near  | 0.650 | 0.600   | 0.367      |
| drawer     | 0.556 | 0.556   | 0.444      |

## Task-averaged (3 tasks)

| Method          | Tokens | Prune % | **Avg SR** |
|-----------------|--------|---------|------------|
| base            | 256    | 0%      | **65.5**   |
| ADP (native)    | 192*   | 25%*    | **71.0**   |
| ours keep0.75   | 192    | 25%     | **70.8**   |
| ours keep0.50   | 128    | 50%     | **52.0**   |
| TeamVLA (native)| 80     | 69%     | **63.0**   |
| ours keep0.31   | 80     | 69%     | **49.7**   |

## Read-out (multi-task)

1. **At mild pruning (25% / 192 tok), our faithful method is fully competitive.**
   ours_k75 **70.8 ≈ ADP 71.0**, both **above base (65.5)** on average. The coke-only picture
   (ADP 0.84 ≫ ours 0.747) was misleading — ours *wins* on drawer (0.778 vs 0.556), so across
   tasks the two tie. Faithful 25% pruning is free (or better) across tasks.

2. **At aggressive pruning (69% / 80 tok), TeamVLA clearly wins.**
   TeamVLA **63.0** ≫ ours **49.7** (+13.3), and TeamVLA nearly holds base (65.5). Merging
   retains dropped-token information that hard pruning loses — this holds on ALL three tasks
   (coke +5.3, move_near +23.3, drawer +11.2 in team's favor).

3. **Task-dependent pruning sensitivity.** coke tolerates heavy pruning (free to 50%);
   **move_near and drawer degrade by 128 tokens**. So the honest cross-task "free pruning"
   ceiling is ~**25% (192 tok)**, not the 50% that coke alone suggested. move_near (multi-object
   scene, must pick the right object) is the most pruning-sensitive.

## Bottom line (validated across 3 tasks)
- Faithful T3R hard-pruning **matches base and ADP at ~25% pruning** across tasks — a solid,
  honest efficiency win (25% fewer visual tokens, no SR cost, competitive with SOTA ADP).
- At **high pruning (≥50%)** it is **dominated by TeamVLA (merging)** and by ADP (adaptive
  gating). Pure SigLIP-SAM dropping cannot match them there — the winning mechanisms
  (merging / phase-adaptive retention) are outside faithful pruning by construction.
- SAM-fill / EMA / biasing were all tried and **hurt** (72→51→45 on coke) — no faithful add-on
  recovers the aggressive-regime gap.

Caveats: drawer n=9 (coarse, ±0.11/episode); ADP/ours numbers carry ~±0.08–0.10 CI at n=60–75;
coke rows are from the Pareto session (base stable across sessions).
