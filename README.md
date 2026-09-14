# T3R: Training-Free Two-Stage Token Refinement Towards Efficient and Robust VLA Models

Official implementation of **T3R** (ASP-DAC 2027), a training-free inference-time
framework that combines segmentation-guided visual-token pruning with
instruction-saliency attention biasing. The `main` and `t3r` branches contain
the OpenVLA-OFT implementation; transfer experiments are available on the
[`t3r-cogact`](https://github.com/midotronn/openvla-oft/tree/t3r-cogact) and
[`t3r-pi0`](https://github.com/midotronn/openvla-oft/tree/t3r-pi0) branches.

**Project website: https://midotronn.github.io/openvla-oft/**

**Base project website: https://openvla-oft.github.io/**

**Base paper: https://arxiv.org/abs/2502.19645**

## Overview

T3R is applied at inference time on top of a pretrained OpenVLA-OFT checkpoint (no retraining required). It combines two components:

1. **SigLIP-SAM Token Pruning** — Uses SigLIP text-image cosine similarity to identify task-relevant image patches, then generates a segmentation mask via EfficientTAM to physically remove irrelevant vision tokens (~85% pruned from the third-person camera image).

2. **IG Attention Biasing** — Computes Integrated Gradients saliency over text tokens once per episode, then injects an additive bias into the attention mechanism at layers 8–23 to steer action token attention toward high-saliency instruction tokens.

## Results Reported in the Paper

### OpenVLA-OFT on LIBERO-Spatial

Ten tasks with ten episodes per task. Reduction refers to scene-camera tokens
removed before decoder execution.

| Method | Token reduction | Success rate | Runtime |
|---|---:|---:|---:|
| Baseline | 0% | **97%** | 107.5 ms |
| TeamVLA | ~15% | 94% | 102.6 ms |
| FastV | 85% | 14% | — |
| ADP | 85% | 66% | — |
| **T3R** | **85%** | **96%** | **92.5 ms** |

T3R also matches the unpruned baseline within one percentage point on
LIBERO-Goal (98% versus 99%). Under object-position perturbations, it leads the
baseline by up to seven percentage points at 4 cm.

### Cross-Backbone Transfer

CogACT is evaluated zero-shot on single-camera SIMPLER. Each method is shown at
its native operating point.

| Method (keep ratio) | Google Robot | WidowX/Bridge |
|---|---:|---:|
| Baseline (100%) | **73.2%** | 10.4% |
| ADP (75% / ~87%) | 71.0% | **16.7%** |
| FastV (~60%) | 73.3% | 15.6% |
| TeamVLA (31.25%) | 69.5% | 12.5% |
| **T3R (60%)** | 67.7% | 10.4% |

The π0.5 transfer is evaluated zero-shot on five RoboTwin tasks with three
cameras. The unpruned baseline averages 33.4% success.

| Tokens/camera | Keep ratio | T3R | ADP | FastV | TeamVLA |
|---:|---:|---:|---:|---:|---:|
| 64 | 25% | **13.3%** | 13.3% | 11.7% | 3.3% |
| 96 | 37.5% | **25.0%** | 13.3% | 23.3% | 11.7% |
| 128 | 50% | **33.3%** | 19.4% | 18.1% | 13.9% |
| 160 | 62.5% | **38.3%** | 25.0% | 28.3% | 15.0% |
| 192 | 75% | **62.5%** | 33.4% | 37.5% | 20.9% |

The π0.5 branch uses its norm-saliency adaptation with original token positions
preserved. The reported table does not use the OpenVLA-OFT IG attention-bias
stage.

---

## RunPod Setup (Tested & Verified)

These instructions have been tested end-to-end on a RunPod L40S (48GB) pod for
the OpenVLA-OFT/LIBERO-Spatial evaluation in this branch. CogACT and π0.5 use
their linked branches.

### 1. Create the Pod

- **GPU**: 1× L40S 48GB (or A100/H100 — anything with ≥16GB VRAM works for inference)
- **Base image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Disk**: 50GB+ (model checkpoint is ~14GB, downloaded automatically from HuggingFace)

### 2. Clone the Repo

```bash
cd /workspace
git clone https://github.com/midotronn/openvla-oft.git
cd openvla-oft
```

### 3. Install Dependencies

Run these steps **in order**. Each step addresses a specific compatibility issue discovered during testing.

```bash
#!/bin/bash
set -e

# ── Step 1: Transformers fork (CRITICAL) ──────────────────────────────────────
# This fork patches LlamaSdpaAttention with is_causal=False for bidirectional
# attention in OFT's parallel decoder. WITHOUT THIS, performance drops to ~0%.
pip install git+https://github.com/moojink/transformers-openvla-oft.git

# ── Step 2: Uninstall broken TensorFlow ───────────────────────────────────────
# The RunPod base image ships TF 2.15 which is incompatible with numpy 2.x.
# TF is listed in pyproject.toml but is NOT used by any T3R evaluation code.
# If left installed, it crashes on import via numpy.core._multiarray_umath.
pip uninstall tensorflow tensorflow-datasets tensorflow-graphics -y 2>/dev/null || true

# ── Step 3: Fix json_numpy ────────────────────────────────────────────────────
# json_numpy monkey-patches json.loads globally. With numpy 2.x + scipy,
# internal JSON parsing hits json_numpy's object_hook with a SimpleNamespace
# instead of a dict, causing: TypeError: argument of type 'types.SimpleNamespace'
# is not iterable. This one-liner patches it to check isinstance(dct, dict) first.
pip install json-numpy
python3 -c "
import json_numpy, inspect
src_file = inspect.getfile(json_numpy)
content = open(src_file).read()
patched = content.replace(
    '    if \"__numpy__\" in dct:',
    '    if isinstance(dct, dict) and \"__numpy__\" in dct:'
)
open(src_file, 'w').write(patched)
print('Patched json_numpy')
"

# ── Step 4: Install T3R-specific dependencies ─────────────────────────────────
pip install timm==0.9.16 captum scikit-learn draccus

# ── Step 5: Install openvla-oft package ───────────────────────────────────────
# The pyproject.toml pins torch==2.2.0 / torchvision==0.17.0 / tensorflow==2.15.0
# but the pod's pre-installed PyTorch (2.4+) works fine. pip will skip downgrades
# for already-installed packages. Ignore version conflict warnings.
pip install -e .

# ── Step 6: Install LIBERO ────────────────────────────────────────────────────
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r /workspace/openvla-oft/experiments/robot/libero/libero_requirements.txt

# LIBERO prompts interactively for a dataset path on first import.
# Pre-create the config to skip this:
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /workspace/LIBERO/libero/libero
bddl_files: /workspace/LIBERO/libero/libero/bddl_files
init_states: /workspace/LIBERO/libero/libero/init_files
datasets: /workspace/LIBERO/libero/datasets
assets: /workspace/LIBERO/libero/libero/assets
EOF

# ── Step 7: Install EfficientTAM ─────────────────────────────────────────────
cd /workspace
git clone https://github.com/yformer/EfficientTAM.git
pip install hydra-core  # Required by EfficientTAM, not installed by default

# Download the efficienttam_s checkpoint (~131MB)
mkdir -p /workspace/EfficientTAM/checkpoints
wget -q -O /workspace/EfficientTAM/checkpoints/efficienttam_s.pt \
    https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s.pt

# ── Step 8: Install headless rendering dependencies ───────────────────────────
apt-get update -qq && apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf > /dev/null 2>&1

# ── Step 9: Update config paths ──────────────────────────────────────────────
cd /workspace/openvla-oft
sed -i 's|sam_checkpoint: "EfficientTAM/checkpoints"|sam_checkpoint: "/workspace/EfficientTAM/checkpoints"|' \
    experiments/robot/configs/config_full_pipeline.yaml
sed -i 's|efficienttam_base_dir: "EfficientTAM"|efficienttam_base_dir: "/workspace/EfficientTAM"|' \
    experiments/robot/configs/config_full_pipeline.yaml

echo ""
echo "=== Setup complete ==="
```

### 4. Verify the Environment

```bash
python3 -c "
import torch
assert torch.cuda.is_available()
print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')

import transformers
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
import inspect
src = inspect.getsource(LlamaSdpaAttention.forward)
assert 'is_causal=False' in src, 'FATAL: transformers fork missing is_causal=False'
print(f'transformers {transformers.__version__} (fork OK)')

import timm; print(f'timm {timm.__version__}')
import captum; print('captum OK')
from sklearn.cluster import DBSCAN; print('sklearn OK')
import draccus; print('draccus OK')

import sys; sys.path.insert(0, '/workspace/EfficientTAM')
from efficient_track_anything.build_efficienttam import build_efficienttam
print('EfficientTAM OK')

print('\nAll checks passed.')
"
```

---

## Running Evaluations

### Environment Variables

Set these before every evaluation run:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/LIBERO:/workspace/openvla-oft
```

### T3R Full Pipeline (96% SR, ~85% token pruning)

```bash
cd /workspace/openvla-oft
python experiments/robot/run_libero_eval_with_sam.py \
    --config experiments/robot/configs/config_full_pipeline.yaml
```

The model checkpoint (`moojink/openvla-7b-oft-finetuned-libero-spatial`, ~14GB) is downloaded automatically from HuggingFace on the first run. EfficientTAM's torch.compile runs on the first episode (~5–10 min), then all subsequent episodes are fast.

**Expected runtime**: ~2 hours total (10 tasks × 10 episodes) on an L40S.

**Expected output**: Results are saved to `eval_results/full_pipeline/summary_libero_spatial_tasks_0-9.txt`.

### Baseline (97% SR, no pruning)

```bash
cd /workspace/openvla-oft
python experiments/robot/run_libero_eval_with_sam.py \
    --config experiments/robot/configs/config_baseline.yaml
```

**Expected runtime**: ~1.5 hours on an L40S (no SAM overhead).

**Expected output**: Results are saved to `eval_results/baseline/summary_libero_spatial_tasks_0-9.txt`.

### TeamVLA Token Pruning (~94% SR)

```bash
cd /workspace/openvla-oft
python experiments/robot/team/run_libero_eval_team.py \
    --config experiments/robot/configs/config_teamvla.yaml
```

### Run All Three Conditions

```bash
bash run_ablation_3cond.sh
python compare_3cond.py  # Prints per-task comparison table
```

---

## Config Reference

Configs live in `experiments/robot/configs/`. Key settings used by the full
pipeline are:

```yaml
# Model (auto-downloaded from HuggingFace)
model_path: "moojink/openvla-7b-oft-finetuned-libero-spatial"

# Task
task_suite_name: "libero_spatial"
task_id: 0-9
num_episodes: 10
center_crop: true              # CRITICAL: must match training augmentation

# SigLIP-SAM token pruning
use_sam: true
sam_backend: "efficienttam"
sam_model_type: "efficienttam_s"
top_k_points: 8               # Top-8 similar patches as positive prompts
num_neg_points: 3              # 3 negative point prompts
mask_to_patch_threshold: 0.3   # Patch kept if ≥30% covered by mask
use_weighted_similarity: true  # Weight text tokens by IDF
use_spatial_clustering: true   # DBSCAN to cluster similar patches

# IG attention biasing
use_attention_bias: true
bias_strength: 2.0             # Additive bias magnitude
bias_active_layers: "8-23"     # Only bias middle/late layers
saliency_top_k_ratio: 0.3     # Top 30% of text tokens biased
```

The evaluator uses five Integrated Gradients steps, matching the paper.

---

## Known Issues & Workarounds

These issues were discovered during reproduction testing. The setup script above already handles all of them, but they are documented here for reference.

### 1. TensorFlow / numpy incompatibility

**Symptom**: `ImportError: numpy.core.umath failed to import` or `AttributeError: _ARRAY_API not found`

**Cause**: The RunPod base image ships TF 2.15 alongside numpy 2.x. TF 2.15 requires numpy <2.0. The `pyproject.toml` lists `tensorflow==2.15.0` as a dependency because it's needed for training data loading, but it is **not used** by any T3R evaluation code.

**Fix**: `pip uninstall tensorflow tensorflow-datasets tensorflow-graphics -y`

### 2. json_numpy crashes with SimpleNamespace

**Symptom**: `TypeError: argument of type 'types.SimpleNamespace' is not iterable` during environment initialization (robosuite/numba/scipy import chain).

**Cause**: `json_numpy` monkey-patches `json.loads` globally with a custom `object_hook`. When scipy/numba trigger internal JSON parsing, the hook receives a `types.SimpleNamespace` instead of a `dict` and crashes on `if "__numpy__" in dct`.

**Fix**: Patch `json_numpy.py` to add `isinstance(dct, dict)` check (see setup script step 3).

### 3. LIBERO interactive prompt blocks headless execution

**Symptom**: `EOFError: EOF when reading a line` immediately on import.

**Cause**: On first import, LIBERO's `__init__.py` prompts `"Do you want to specify a custom path for the dataset folder? (Y/N)"` via `input()`. This blocks when running non-interactively (e.g., via `nohup`).

**Fix**: Pre-create `~/.libero/config.yaml` with default paths (see setup script step 6).

### 4. Missing `hydra-core` for EfficientTAM

**Symptom**: `ModuleNotFoundError: No module named 'hydra'` — EfficientTAM silently falls back to SAM (which also fails), and evaluation runs **without token pruning** while still logging as if everything is fine.

**Cause**: EfficientTAM uses Hydra for config loading but doesn't list it as a dependency.

**Fix**: `pip install hydra-core`

**How to verify pruning is active**: Look for these log lines:
```
✓ Loaded EfficientTAM image predictor: efficienttam_s
SAM mask computed: 13/256 patches kept (95% pruned)
IG saliency computed in 2.10s (top5=[31, 16, 19, 24, 29])
```
If you instead see `⚠ Warning: Could not load EfficientTAM`, pruning is **NOT active**.

### 5. First episode is very slow (~10 min)

**Symptom**: Episode 1 hangs at `Computing image embeddings for the provided image...` for several minutes with many triton autotuning messages.

**Cause**: EfficientTAM uses `torch.compile` by default. The first forward pass triggers Triton kernel compilation and autotuning. You may see `OutOfMemoryError: out of resource` messages during autotuning — these are expected (Triton is testing large kernel configs and falling back to smaller ones).

**Fix**: This is a one-time cost. All subsequent episodes run at normal speed. No action needed.

---

## Architecture

```
Instruction Text ──→ SigLIP Text Encoder ──→ Text Embeddings
                                                    │
                                                    ├──→ Cosine Similarity ←── Image Patch Embeddings ←── SigLIP Image Encoder ←── Camera Image
                                                    │           │
                                                    │           ▼
                                                    │    Top-K + DBSCAN Clustering
                                                    │           │
                                                    │           ▼
                                                    │    EfficientTAM Point Prompts
                                                    │           │
                                                    │           ▼
                                                    │    Segmentation Mask → patch_mask (keeps ~15% of 256 patches)
                                                    │
                                                    ├──→ Integrated Gradients ──→ Text Token Saliency
                                                    │                                    │
                                                    │                                    ▼
                                                    │                            Attention Bias (layers 8–23)
                                                    │
                                                    ▼
                                            LLM Forward Pass
                                        (pruned vision tokens + biased attention)
                                                    │
                                                    ▼
                                            Action Prediction
```

- **Token pruning** physically removes vision tokens from the sequence (`patch_features[:, patch_mask, :]`), so the LLM processes fewer tokens.
- **Attention biasing** adds a learned bias to the SDPA attention mask at salient text token positions, steering action queries toward important instruction words.
- The SAM mask is computed once at episode start (after warm-up) and recomputed every 5 VLA calls.
- IG saliency is computed once per episode.

---

## File Structure (T3R-specific)

```
experiments/robot/
├── run_libero_eval_with_sam.py     # Main eval script (Full Pipeline & Baseline)
├── team/
│   └── run_libero_eval_team.py     # TeamVLA eval script
├── configs/
│   ├── config_full_pipeline.yaml   # T3R: SigLIP-SAM + IG biasing
│   ├── config_baseline.yaml        # Vanilla OpenVLA-OFT (no pruning)
│   └── config_teamvla.yaml         # TeamVLA token pruning only
├── openvla_utils.py                # get_vla_action() passes patch_mask to model
├── libero/
│   └── libero_utils.py             # LIBERO environment utilities
└── robot_utils.py                  # Shared robot evaluation utilities

prismatic/
├── extern/hf/modeling_prismatic.py # _process_vision_features() applies patch_mask
│                                   # predict_action() orchestrates pruning + inference
├── models/
│   ├── siglip_guided_sam.py        # SigLIP similarity → EfficientTAM mask
│   ├── attention_bias.py           # IG saliency + SDPA monkey-patching
│   └── token_pruning.py            # TEAM algorithm (for TeamVLA condition)

run_ablation_3cond.sh               # Run all 3 conditions sequentially
compare_3cond.py                    # Print per-task comparison table
```

---

## Citation

If you use this code, please cite both T3R and the base OpenVLA-OFT paper:

```bibtex
@inproceedings{hassan2027t3r,
  title={T3R: Training-Free Two-Stage Token Refinement Towards Efficient and Robust VLA Models},
  author={Hassan, Mohammed and Chen, Zhenyang and Wang, Zheng and Zhu, Zhixin and Chen, Tianlong and Lin, Yingyan and Li, Chaojian},
  booktitle={Proceedings of the 32nd Asia and South Pacific Design Automation Conference (ASP-DAC)},
  year={2027}
}

@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
