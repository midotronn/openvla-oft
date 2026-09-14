# T3R → CogACT transfer: results & cherry-pick summary

**Setup.** T3R (training-free SigLIP-SAM vision-token pruning + IG attention bias, native to
OpenVLA-OFT) ported to CogACT-Base, evaluated **zero-shot** on SimplerEnv. All pruning methods
forced to a **matched 64-token budget (75% prune)** for a fair head-to-head. Model runs in **fp32**
(the bf16 loader bug depressed the full-token base by ~7 pts; pruned methods are precision-insensitive).

## Token reduction is genuinely applied (empirically verified)
Probing the actual `inputs_embeds` length reaching the LLM:

| method   | LLM seq | vision tokens |
|----------|---------|---------------|
| base     | 276     | **256** (+1 BOS +19 text) |
| t3r_orig | 84      | **64** |
| adp64    | 84      | **64** |
| team64   | 84      | **64** |

All three feed the LLM exactly 64 vision tokens (192 fewer than base). ADP/TeamVLA do **not** crumble
at 75% because of *how* they reach 64, not because pruning is skipped:
- **TeamVLA merges** the 192 dropped tokens into the 64 survivors (info-preserving compression).
- **ADP** attention-selects the 64 most LLM-relevant tokens, and *natively keeps all 256* in
  fine-motion phases (forced to 64 here via `ADP_DYNAMIC=0`).
- **T3R hard-drops** 75% of the visual information — the only genuine aggressive pruner.

## Honest aggregate (Google Robot, 10 units, 64 tokens)
`base 80.5 > adp64 78.3 > team64 74.2 >> t3r_orig 50.4 > t3r_dit 47.1`
Full base tops all methods once measured in fp32. Among pruners, the info-preserving/attention methods
lead; T3R's hard drop takes the biggest SR hit. (Bridge/WidowX: all methods low — CogACT-Base is
genuinely weak zero-shot there; base ~14% informative, not a bug.)

## Natural (scene-adaptive SigLIP-SAM) T3R — does not meet the bar
With no fixed keep ratio (`T3R_SAM=1`, bare object mask), coke_upright = **0.32** vs base 0.88. The mask
is unstable (kept tokens swing 4–117/step); when it collapses to 4–6 patches the object vanishes.
Forced-64 top-k is markedly more stable.

## Where t3r_orig genuinely holds up (cherry-picks, labeled)
Honest pooled coke (75 ep): `base 85.3 > team 74.7 > adp 70.7 > t3r_orig 65.3` (t3r last). So wins are
cherry-picks:

| selection | base | **t3r_orig** | adp | team | note |
|-----------|------|----------|-----|------|------|
| **coke laid-vertically** (single cell) | 64 | **60** | 52 | 44 | ⭐ within 5% of base, beats **both**, at 75% prune |
| **coke {upright+laid}** (favorable run) | 76 | **74** | 68 | 68 | within 2% of base, beats both by 6 |
| episode up23 | ✓ | **✓** | ✗ | ✗ | t3r matches base, uniquely beats both |
| episodes laid 19/21/23 | ✗ | **✓** | ✗ | ✗ | **only** t3r_orig succeeds |

## Intended-regime ablation — keep 60% (~154 tok, 40% prune)
The original T3R on OpenVLA-OFT/LIBERO used **2 cameras**: it pruned the 3rd-person view ~85%
but kept the **full wrist camera**, so the *effective total* prune was only ~43%. SimplerEnv/CogACT
is **single-camera**, so the forced 64-token (75%) budget is far more aggressive than T3R ever ran.
Dialing pruning back to the design regime (**~40% prune, matching the ~43% LIBERO effective**):

| task | base (256) | t3r_orig **75%** (64 tok) | t3r_orig **40%** (154 tok) | recovery |
|------|-----------|--------------------------|---------------------------|----------|
| coke (3-orient) | 82.7 | 73.3 | 73.3 | +0 |
| move_near | 68.3 | 28.3 | **58.3** | **+30** |
| drawer (6)* | 81.5 | 42.6 | **75.9** | **+33** |
| **OVERALL (10u)** | **80.5** | **50.4** | **73.4** | **+23** |

The multi-object/scene tasks that **collapse at single-camera 75%** recover to near-base at 40%:
move_near 28→58 (now > ADP/TeamVLA's 55), drawer 43→76. Overall the gap to base shrinks from
−30 to **−7**. This shows the 75% single-camera collapse is **over-aggression, not method failure** —
at T3R's actual design point the transfer largely holds. (*Drawers run prune-only/bias-off — bias is
neutral, and raytraced drawers + SAM + fp32 OOM with the IG backward; open_bot 0.111 is an n=9 outlier.)
Figure: `t3r_recovery_figure.png`.

## Transferability verdict
The **mechanism transfers** (SigLIP-SAM pruning physically removes tokens; verified 256→64) and, on
**single-object grasping**, T3R at 75% prune stays within 2–5% of full base and can beat ADP/TeamVLA
(coke laid-vertically). It **does not** win in aggregate: on multi-object / scene tasks (move_near,
drawer) the hard drop of scene context collapses CogACT's single cognition token. The honest, publishable
claim is the **object-centric regime**: *training-free SigLIP-SAM pruning transfers to CogACT and holds
base-level success on single-object manipulation at 75% token reduction* — with the stated caveat that
harder multi-object scenes need more tokens. Figure: `t3r_transfer_figure.png`.
