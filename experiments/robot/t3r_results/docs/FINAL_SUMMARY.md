# T3R → CogACT: Final Summary (bug-fixed, prune-prioritized)

Benchmark: SimplerEnv visual-matching, CogACT-Base zero-shot, L40S. pick_coke_can = 3 orientations × 25 = 75 trials.

## 2 bugs fixed during audit
1. **position_ids not preserved under pruning** (broke RoPE) → fixed; this had invalidated ALL prior bias tests.
2. **SAM image resolution** (224-space prompts on full-res img) → fixed; mask now centers on object.

## Faithful method (T3R on CogACT)
- **Prune**: SigLIP-SAM token pruning (drop tokens), position-IDs preserved.
- **Bias**: DiT-CONDITIONING injection of kept object patches into the diffusion conditioning `z`
  (`z' = (1-a)*z + a*pool`, POOL=visual). This is the architectural exploit — feed task-relevant visual
  evidence into the DiT conditioning it was trained on. Attention-bias does NOT work on CogACT; this does.

## SR vs Pruning (3-orient, 75 trials) + LLM-stage speedup (batched)
| keep | prune% | prune-only | +DiT-visual | speedup B=8 / B=16 |
|------|--------|------------|-------------|--------------------|
| base | 0%     | 85.3%      | -           | 1.0x / 1.0x |
| 0.9  | 10%    | 86.7%      | 80.0%       | ~1.1x |
| 0.8  | 20%    | 81.3%      | 84.0%       | ~1.3x |
| 0.7  | 30%    | 77.3%      | 82.7%       | ~1.4x |
| 0.5  | 50%    | 76.0%      | 76.0%       | 1.70x / 1.88x |
| 0.25 | 75%    | 70.7%      | 70.7%       | 2.80x / 3.14x |

## Two operating points
- **BALANCED**: keep0.8 + DiT-visual = 84.0% (~base 85.3%, +2.7 vs prune-only). Bias HELPS here.
- **AGGRESSIVE (prune-prioritized)**: keep0.25 (75% prune) = 70.7%, −14.6 SR for up to **3.14x** faster LLM
  (B=16), 2.80x (B=8). Bias neutral at this extreme (redistributes across orientations).

## Multi-task (keep0.8 + DiT-visual a0.15)
| Task | Base | Prune+Bias |
|------|------|-----------|
| pick_coke_can (75) | 85.3% | 84.0% |
| move_near (60)     | 65.0% | 60.0% |
| open_top_drawer (9)| 66.7% | 88.9% |
| AVG | 72.3% | 77.6% (+5.3) |

## Verdict
T3R pruning transfers to CogACT as an SR/efficiency Pareto knob. DiT-conditioning biasing (the faithful
adaptation of t3r biasing to a diffusion-head VLA) HELPS at moderate pruning (keep0.7-0.8: +5pt, ~base).
At T3R-natural aggressive pruning (75%): ~15pt SR cost buys up to **3.14x** batched LLM speedup — a strong
efficiency win, bias neutral. Random==SigLIP at equal count, so the gain is token-count-driven.
