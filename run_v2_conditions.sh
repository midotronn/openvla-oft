#!/bin/bash
# Run A+B experimental conditions (moderate pruning + visual highlighting)
# Conditions:
#   1. Baseline (6cm OOD, no SAM, no bias, no highlight)
#   2. Prune-v2 (50% pruning, no bias, no highlight)
#   3. Full-v2 (50% pruning + visual highlighting, no bias)
#
# All conditions use the same OOD seed (42) and 6cm perturbation.
# The current v1 results are preserved in their original output dirs.

set -e

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export TORCHDYNAMO_DISABLE=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/LIBERO:/workspace/openvla-oft

cd /workspace/openvla-oft

echo "============================================"
echo "CONDITION 1/3: Baseline (6cm OOD)"
echo "============================================"
# Save to a new dir so we don't overwrite the 10cm baseline
mkdir -p eval_results/baseline_6cm rollouts/baseline_6cm
# Use a temporary config with 6cm baseline output path
python -c "
import yaml
with open('experiments/robot/configs/config_baseline.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['output_dir'] = './eval_results/baseline_6cm'
cfg['rollout_dir'] = './rollouts/baseline_6cm'
cfg['ood_position_noise_cm'] = 6.0
with open('/tmp/config_baseline_6cm.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
python experiments/robot/run_libero_eval_with_sam.py --config /tmp/config_baseline_6cm.yaml
echo "Baseline 6cm complete."

echo ""
echo "============================================"
echo "CONDITION 2/3: Prune-v2 (50% pruning)"
echo "============================================"
mkdir -p eval_results/prune_v2 rollouts/prune_v2
python experiments/robot/run_libero_eval_with_sam.py --config experiments/robot/configs/config_prune_v2.yaml
echo "Prune-v2 complete."

echo ""
echo "============================================"
echo "CONDITION 3/3: Full-v2 (50% prune + highlight)"
echo "============================================"
mkdir -p eval_results/full_v2 rollouts/full_v2
python experiments/robot/run_libero_eval_with_sam.py --config experiments/robot/configs/config_full_v2.yaml
echo "Full-v2 complete."

echo ""
echo "============================================"
echo "ALL CONDITIONS COMPLETE"
echo "============================================"
echo "Results saved to:"
echo "  eval_results/baseline_6cm/"
echo "  eval_results/prune_v2/"
echo "  eval_results/full_v2/"
