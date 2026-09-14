# T3R → CogACT: Transferability (FINAL, properly-powered, CORRECTED)

Benchmark: SimplerEnv visual-matching, Google-Robot pick-coke-can, `CogACT/CogACT-Base` zero-shot, L40S.
Protocol: 25 object positions × 3 can orientations (upright / lr_switch / laid_vertically) = 75 trials/condition.

## Result (75 trials/condition)
| Config | upright | lr_switch | laid_vert | AVG/75 |
|---|---|---|---|---|
| Base (256 tok) | 0.80 | 0.92 | 0.84 | **85.3%** |
| keep0.7 (30% prune) | 0.80 | 0.76 | 0.60 | 72.0% |
| keep0.6 (40% prune) | 0.92 | 0.72 | 0.60 | 74.7% |
| keep0.5 (50% prune) | 0.92 | 0.64 | 0.64 | 73.3% |
| keep0.5 + bias | 0.88 | 0.60 | 0.60 | 69.3% |

## CORRECTION of earlier claim
Earlier "T3R-CogACT matches base at 50% pruning" was measured on **upright only** (the easiest pose,
where pruning even helps 0.80→0.92) and with noisy 16-ep samples. On the **full 3-orientation
protocol**, pruning costs ~11–16 SR points; **no** keep ratio in [30%,50%] matches base. The cost
concentrates on harder object poses (lr_switch, laid_vertically). Biasing does not recover.

## Honest conclusions
- **Mechanism transfers** (verified): SigLIP pruning physically removes vision tokens (LLM 278→150),
  actions change, per-episode outcomes differ. Pipeline runs end-to-end on CogACT.
- **Latency gain is real**: 50% fewer LLM vision tokens → up to 1.89× faster LLM stage (batched).
- **SR does NOT transfer at parity** on the full protocol: pruning trades ~12 SR points for that speed.
  T3R-CogACT matches/beats base **only** on the easy upright orientation.
- **Biasing does not help** on any orientation (attention bias or DiT-conditioning injection).

## Why (architectural)
CogACT condenses everything into ONE cognition token feeding a DiT diffusion head. Removing vision
tokens degrades that single summary more on hard poses than OFT's parallel decoder (which reads many
action-token queries over full context). T3R's aggressive pruning is a poorer fit for a
single-cognition-token diffusion VLA than for OFT.

## Methodology lesson
SimplerEnv SR must be measured over all 3 can orientations (ideally 4 URDF variants = 300 trials);
single-orientation or <25-trial evals are misleading. 64 upright-only episodes gave a false positive.
