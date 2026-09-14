#!/bin/bash
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

echo "=== 0cm SANITY v3: crop_scale=0.95, cfg.center_crop=False ==="
echo "Started: $(date)"
python experiments/robot/run_libero_eval_base.py \
  --config experiments/robot/configs/config_base_0cm_sanity.yaml
echo "Finished: $(date)"
