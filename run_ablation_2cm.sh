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
SCRIPT="experiments/robot/run_libero_eval_base.py"
CONFIGS="experiments/robot/configs"

echo "============================================="
echo "  BASE OPENVLA ABLATION (2cm, 5 tasks x 10 eps)"
echo "  Order: full -> baseline -> prune-only"
echo "  Started: $(date)"
echo "============================================="
echo ""

echo "[1/3] FULL PIPELINE (prune + bias)"
echo "============================================="
python "$SCRIPT" --config "$CONFIGS/config_base_full.yaml" 2>&1
echo "FULL DONE: $(date)"
echo ""

echo "[2/3] BASELINE"
echo "============================================="
python "$SCRIPT" --config "$CONFIGS/config_base_baseline.yaml" 2>&1
echo "BASELINE DONE: $(date)"
echo ""

echo "[3/3] PRUNE ONLY"
echo "============================================="
python "$SCRIPT" --config "$CONFIGS/config_base_prune_only.yaml" 2>&1
echo "PRUNE_ONLY DONE: $(date)"
echo ""

echo "============================================="
echo "  ALL DONE: $(date)"
echo "============================================="
