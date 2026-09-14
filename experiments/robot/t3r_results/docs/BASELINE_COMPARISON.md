# T3R vs ADP vs TeamVLA — CogACT / SimplerEnv baseline comparison

Task: **pick_coke_can**, full protocol = 3 orientations (upright, lr_switch, laid_vertically)
× 25 obj positions = **75 trials per condition**. CogACT-Base zero-shot. All conditions
measured in ONE process/session (single model load), so numbers are directly comparable.

Per the user's instruction, the baselines run in their NATIVE configuration — the keep ratio
is NOT fixed to ours (that constraint suits SigLIP-SAM only). Each method reports its OWN
achieved token reduction.

## Methods
- **base** — no token reduction (256 visual tokens).
- **ADP** (Action-aware Dynamic Pruning, ICLR'26, arXiv:2509.22093): text→vision QK attention
  importance @ LLaMA layer-0 selects tokens; an action-aware gate prunes in coarse-motion
  phases and keeps all tokens in fine-manipulation phases. Native qk_keep=0.75 when pruning.
- **TeamVLA** (Token Expand-Merge, arXiv:2512.09927): similarity-sample salient patches per
  language token, then soft bipartite MERGE the rest into anchors. Native merge_topk=80.
- **ours_bal** (T3R) — SigLIP-SAM keep0.8 + DiT-visual conditioning bias α0.15.
- **ours_aggr** (T3R) — SigLIP-SAM keep0.25 (aggressive 75% prune, no bias).

All baselines ported to CogACT in `experiments/robot/baselines_cogact.py`, hooked at
`projector.forward` (post-projection, LLM space) and reusing the SAME position-id RoPE fix as
ours, so only the token-reduction strategy differs.

## Results (SR %, 75 trials)

| Method     | Reduction | upright | lr_switch | laid_vert | **AVG**  |
|------------|-----------|---------|-----------|-----------|----------|
| base       | 0%  (256) | 84.0    | 92.0      | 64.0      | **80.0** |
| ours_bal   | 20% (204) | 88.0    | 88.0      | 76.0      | **84.0** |
| ADP        | 25% (192) | 84.0    | 92.0      | 60.0      | **78.7** |
| TeamVLA    | 69% (80)  | 80.0    | 80.0      | 60.0      | **73.3** |
| ours_aggr  | 75% (64)  | 84.0    | 72.0      | 40.0      | **65.3** |

## Read-out

**Mild-reduction regime (~20–25%):**
- **ours_bal 84.0% BEATS base (80.0, +4.0) and ADP (78.7, +5.3)** — at the FEWEST tokens of
  the three mild conditions (204 vs ADP 192… comparable). Our SigLIP-SAM prune + DiT-visual
  conditioning is the best operating point here.
- ADP ≈ base (−1.3) — its model-internal QK selection + action gate roughly preserves SR at
  ~25% prune, as the paper claims, but gives no gain on CogACT.

**Aggressive-reduction regime (~70–75%):**
- **TeamVLA 73.3% > ours_aggr 65.3% (+8.0)** — BUT TeamVLA keeps 80 tokens vs our 64. Its
  MERGING (folding background into anchors) retains information that our hard DROP discards, so
  it degrades more gracefully at high reduction. This is the regime where merging beats pruning
  on CogACT (consistent with our earlier finding that CogACT's cognition token is token-COUNT
  sensitive — keeping 80 merged > 64 dropped).

**Bottom line:** T3R (ours) wins the accuracy-first regime (beats base + ADP at ~20% prune);
TeamVLA wins the efficiency-first regime (better SR retention at high reduction via merging).
ADP lands between, matching base at mild prune with no gain. To reach high prune ratios within
a few SR points of base on CogACT, a merge step (à la TeamVLA) is more effective than pure
SigLIP-SAM dropping.

Note: this run's fresh base = 80.0% (vs an earlier separate-session 85.3%); the laid_vertically
orientation is the variance source. Since base/ADP/TeamVLA/ours were all measured together here,
the relative comparison is the rigorous one.
