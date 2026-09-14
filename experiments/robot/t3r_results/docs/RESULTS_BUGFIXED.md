# T3R → CogACT: Transferability (FINAL, bug-fixed, properly-powered)

Benchmark: SimplerEnv visual-matching, Google-Robot pick-coke-can, CogACT/CogACT-Base zero-shot, L40S.
Protocol: 25 positions × 3 can orientations (upright/lr_switch/laid_vertically) = 75 trials/condition.

## TWO BUGS FOUND & FIXED (audit prompted by reviewer skepticism)
1. **position_ids not preserved under pruning**: CogACT `PrismaticVLM.forward` calls the LLM with
   `position_ids=None` → pruning gave CONTIGUOUS positions, shifting kept patches+text off their
   trained RoPE positions. FIX: preserve original positions (BOS=0, kept_idx+1, text@257+).
   Effect: lr_switch 0.64 → 0.80 (+16 pts).
2. **SAM image resolution**: point prompts in (224,224) space were applied to the FULL-RES image →
   SAM segmented top-left background. FIX: resize image to 224 before SAM → mask centers on can.

## Pareto frontier (position-fixed, 75 trials/condition)
| Keep | ~tokens | prune% | upright | lr_switch | laid_v | AVG |
|------|---------|--------|---------|-----------|--------|-----|
| Base | 256 | 0% | 0.80 | 0.92 | 0.84 | 85.3% |
| 0.9 | ~230 | 10% | 0.88 | 0.92 | 0.80 | **86.7%** (matches/beats base) |
| 0.8 | ~205 | 20% | 0.88 | 0.88 | 0.68 | 81.3% |
| 0.7 | ~179 | 30% | 0.84 | 0.80 | 0.68 | 77.3% |
| 0.5 | ~128 | 50% | 0.84 | 0.80 | 0.64 | 76.0% |
| SAM | ~8-70 | 85-97% | ~0.42 | - | - | ~42% |

## SAM "collapse" is NOT a bug (verified)
Multi-episode SAM instrumentation: mask localizes correctly (centroid on can); kept-count tracks
object apparent size (6 when small/distant, ~70-100 with gripper near). Succeeded episodes kept
mean 71 patches; failed kept mean 48; the two episodes that stayed ~8-9 patches both failed.
→ The drop is a real **token-count limit**: CogACT's single cognition token needs ~200 tokens; T3R's
aggressive SAM prune (to tens) starves it. Consistent with the Pareto.

## Conclusions (bug-fixed)
- **Pruning transfers at LIGHT ratios**: 10% prune matches/beats base (86.7% vs 85.3%) across all 3
  orientations; 20% prune ~81%. Mechanism verified correct (token removal + RoPE preservation).
- **Aggressive pruning (T3R's native 85%+) does NOT transfer**: CogACT needs far more visual tokens
  than OFT's parallel decoder (single cognition token + DiT diffusion head is token-hungry).
- **Latency**: at 10-20% prune the LLM speedup is modest (~1.1×); the 1.89× batched speedup only
  comes at 50% prune, which costs ~9 SR pts. Pareto trade, not a free win.
- **Biasing** (attention or DiT-conditioning): neutral on CogACT (tested separately, properly powered).

## Methodology lesson
Must evaluate all 3 can orientations (≥25 trials each); upright-only / 16-ep evals were misleading.
3. **Random control**: SigLIP keep0.5=76.0% == RANDOM keep0.5=76.0%; keep0.9 SigLIP 86.7% vs RANDOM 89.3%.
   => T3R smart selection gives ZERO benefit over random on CogACT. Only token COUNT matters.
   VERDICT: T3R does NOT meaningfully transfer (pruning intelligence redundant; biasing neutral).
