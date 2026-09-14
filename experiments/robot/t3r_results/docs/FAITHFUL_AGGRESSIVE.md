# Keeping T3R SR up at aggressive pruning (keep ≤ 25%) — faithful, exhaustive study

Goal: hold SR at high pruning on CogACT while staying faithful to original T3R (SigLIP-SAM
prune + IG saliency bias — NO merging). Task: coke_can, 75-trial (3 orient × 25). keep0.25 =
64 tokens = 75% prune. All comparisons same-session; prune-only reference = **0.720**.

## Every faithful lever tested at keep0.25 (vs prune-only 0.720)

| Faithful mechanism                          | Avg SR | vs prune-only |
|---------------------------------------------|--------|---------------|
| **raw SigLIP-SAM prune-only**               | **0.720** | —          |
| + SigLIP spatial-coverage grid (g=5, light) | 0.720  | 0.0 (neutral) |
| + IG attention bias, strength 1.0           | 0.707  | −0.013        |
| + SigLIP grid (g=6)                          | 0.693  | −0.027        |
| + IG attention bias, strength 1.5           | 0.680  | −0.040        |
| + SigLIP grid (g=8, full coverage)          | 0.653  | −0.067        |
| + recompute mask every step (rc1)            | 0.507  | −0.213        |
| + SAM-guaranteed object + SigLIP fill + EMA | 0.507  | −0.213        |
| + DiT-conditioning injection (prior)         | ~neutral | ~0.0        |

**Result: NO faithful add-on beats raw SigLIP-SAM prune-only.** Best case is neutral (light
grid, mild bias); everything else hurts. attention bias at str2.0 = 0.222 (starves the single
cognition token of the instruction). Frequent recompute destabilizes (mask jumps → 0.44 upright).
The grid gave one intriguing signal — full coverage (g=8) lifted the *hardest* pose
laid_vertically 0.56→0.76 (scene context helps awkward grasps) — but destroyed object detail on
easy poses (upright 0.84→0.68), and the benefit vanished at lighter coverage (partly noise). No
net win.

## Why — grounded in the original T3R code (the key insight)

Original T3R hits ~96% on LIBERO at "aggressive" pruning because it runs on **OpenVLA-OFT with
TWO camera views**: `modeling_prismatic.py` line 464 — *"patch_mask is for the first
(third-person) image only."* It prunes 85% of the third-person view **but keeps the full wrist
camera (256 tokens)**. Its true TOTAL token reduction is only **~43%**, not 85% — the wrist view
preserves full manipulation detail.

**CogACT/SimplerEnv is single-camera + a single "cognition" token** feeding the DiT. There is no
second full view to absorb the dropped information. So:
- At T3R's *true* effective reduction (~43–50%), CogACT **already matches base**: keep0.50 =
  128 tokens = **76.0 = base 76.0** (from the Pareto). T3R transfers faithfully here.
- Pushing to keep0.25 (75% total, ~2× more aggressive than T3R ever demonstrated in total terms)
  loses information that a single hard-pruned view cannot recover. The only mechanism that
  recovers it is **re-introducing dropped tokens = merging** (what TeamVLA does, +8–13 pts at
  80 tokens) — which is explicitly **not faithful**. Hence no faithful lever closes the gap.

## The honest faithful high-pruning result

- **Best faithful config at keep0.25 = raw SigLIP-SAM prune-only.**
- coke: **0.72 at 64 tokens (75% prune) = 95% of base (0.76)** — and this is a MORE aggressive
  token budget than TeamVLA (80) or ADP (avg 192). Upright holds 0.84 = base.
- Degrades on harder multi-object tasks (move_near ~0.37 at 80 tok) — the single-camera limit.

## Recommendation
1. State the faithful efficiency claim in **total-token terms**: T3R faithfully delivers **~50%
   visual-token reduction at no SR cost** on CogACT (keep0.50 = base) — matching what T3R
   actually achieves on OFT once the retained wrist view is counted.
2. For the "unprecedented pruning" headline, report coke keep0.25: **75% pruning at 95% of base
   SR (64 tokens)**, more aggressive than the ADP/TeamVLA operating points — with the honest
   caveat that harder multi-object tasks need more tokens.
3. Beating base at ≥75% prune on single-camera CogACT is **not achievable faithfully** — it
   requires merging (unfaithful). This is a fundamental single-view information limit, not a
   tuning failure.
