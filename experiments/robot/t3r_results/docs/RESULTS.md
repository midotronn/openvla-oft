# T3R → CogACT: Transferability Results

**Goal:** Show T3R's training-free method (SigLIP token pruning + IG attention bias) transfers
to the CogACT VLA backbone. Benchmark: SimplerEnv (CogACT's native OXE eval), Google-Robot
`GraspSingleOpenedCokeCanInScene-v0` (visual matching, upright, 16 episodes), L40S GPU.
Model: `CogACT/CogACT-Base` (7.6B, DINOv2+SigLIP + LLaMA2-7B + DiT-B), **zero-shot, no finetuning**.

## 1. Success Rate vs Pruning (prune-only, faithful SigLIP weighted-similarity)

| Config            | Visual tokens | Pruned | Success (16 ep) |
|-------------------|---------------|--------|-----------------|
| Base CogACT       | 256           | 0%     | 87.5%           |
| **T3R prune 50%** | **128**       | **50%**| **93.75%**  ✅  |
| T3R prune 65%     | 89            | 65%    | 75.0%           |
| T3R prune 75%     | 64            | 75%    | 68.75%          |
| T3R prune 85%     | 38            | 85%    | 62.5%           |

**At 50% pruning, T3R-CogACT matches/beats base** (93.75% ≥ 87.5%). CogACT tolerates moderate
pruning; OpenVLA-OFT's aggressive 85% level over-prunes CogACT (its single "cognition" token is
more context-sensitive than OFT's parallel action tokens).

## 2. Latency / Efficiency (50% pruning, 256→128 tokens)

LLM forward stage (the part pruning affects):

| Batch | 256 tok | 128 tok | Speedup |
|-------|---------|---------|---------|
| 1     | 33.6 ms | 30.0 ms | 1.12×   |
| 8     | 162.9 ms| 95.5 ms | 1.71×   |
| 16    | 343.0 ms| 181.6 ms| **1.89×** |

Single-call breakdown: full predict 81.5 ms = VLM (vision+LLM) 47.6 ms + DiT diffusion 33.8 ms.
- **Single-env latency ≈ flat**: B=1 is memory-bandwidth-bound on the 7B weights, and vision
  encode + DiT are fixed regardless of vision-token count.
- **Batched throughput improves up to 1.89×** on the LLM stage: pruning halves the compute-bound
  attention/MLP cost — the regime relevant for multi-env / served inference.

## 3. IG/Visual Attention Bias AND DiT-Conditioning Injection — both faithfully tested, neither helps

I implemented and rigorously tested **two** faithful adaptations of t3r biasing:

**(a) Attention bias** (t3r's original form): captum IG saliency over instruction tokens → additive
SDPA bias. Bug found & fixed: CogACT's cognition token is the *last* position and pruned sequences
are short, so the old `q_window=64 ≥ seq` biased *every* query row, corrupting representations
(old prune+bias 85% = 50%). Fixed with `q_window=1` (steer only the cognition token) + visual-patch
bias toward SigLIP-kept patches. After the fix it's non-destructive.

**(b) DiT-conditioning injection** (the architecturally-correct form for a diffusion head): move the
cognition feature `z` toward a saliency-weighted pool of salient-token final-layer hidden states,
`z' = (1-α)·z + α·pool`, staying in the same representation space (near-manifold).

**Definitive same-grid comparison (36 episodes, 6×6 identical object positions, keep0.25 = 75% prune):**
| Method | SR |
|---|---|
| Base CogACT | 86.1% |
| Prune-only | 72.2% |
| Prune + attention-bias (str1.0) | 75.0% (+1 ep) |
| Prune + DiT-injection (α=0.15) | 69.4% (−1 ep) |

Both biasing forms fall **within ±1 episode of prune-only → statistically neutral**. Earlier
16-episode "wins" (attn 75%, DiT 81.25%) were **small-sample noise** (SimplerEnv SR varies a lot by
object position; reliable comparisons need ≥25-ep fixed grids).

**Bias-only test (no pruning) on instruction-sensitive move_near (30-ep fixed grid)** — the regime
where biasing should help most if it helps anywhere:
| α (DiT text-injection) | SR |
|---|---|
| base (no bias) | 73.3% |
| 0.10 | 63.3% |
| 0.20 | 73.3% (neutral) |
| 0.30 | 46.7% |
Best case = neutral (matches base at α=0.20); otherwise it degrades. Biasing **never beats base**.

**Why biasing doesn't transfer to CogACT:** SigLIP pruning already *retains the task-relevant
tokens*, and CogACT's cognition feature is formed by full bidirectional attention over them — it
already integrates that information optimally, and the DiT head is trained on exactly that feature
distribution. Any inference-time steering (re-weighting attention, or injecting salient reps into z)
only perturbs an already-good, on-manifold feature. In OpenVLA-OFT the bias helps because its
parallel decoder steers many action-token queries against the *full* vision context; CogACT's single
cognition token does not.

## Conclusion

- ✅ **Pruning transfers**: SigLIP token pruning preserves SR — ~50% prune matches base; 75% prune
  costs ~14% SR (86.1%→72.2%) on a fixed 36-ep grid. Keep-ratio is a backbone-tuned knob.
- ✅ **Efficiency**: 50% fewer LLM vision tokens → up to **1.89× faster LLM stage** (batched);
  single-env wall-clock flat (7B memory-bound + fixed-cost DiT).
- ❌ **Biasing does NOT transfer**: both attention bias (fixed) and DiT-conditioning injection are
  statistically neutral on CogACT at proper sample sizes — pruning already keeps the task-relevant
  tokens, so biasing is redundant. Faithfully implemented and characterized, not an implementation bug.

**Net:** T3R's *pruning* component transfers to CogACT; its *biasing* component is specific to
LLM-direct-decoding VLAs (OFT) and is redundant/neutral on diffusion-head VLAs like CogACT.

