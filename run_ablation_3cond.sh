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

echo '============================================================'
echo ' ABLATION STUDY: 3 CONDITIONS x 10 TASKS x 10 EPISODES'
echo " Started: $(date)"
echo '============================================================'

rm -rf eval_results/ablation rollouts/ablation
mkdir -p eval_results/ablation rollouts/ablation

# ---- Condition 1: Baseline OFT ----
echo ''
echo '============================================================'
echo ' CONDITION 1/3: BASELINE OFT (no pruning, no biasing)'
echo " Started: $(date)"
echo '============================================================'
python experiments/robot/run_libero_eval_with_sam.py \
    --config experiments/robot/configs/config_ablation_baseline.yaml
echo " Baseline completed: $(date)"

# ---- Condition 2: Full Pipeline (SigLIP-SAM + IG Bias) ----
echo ''
echo '============================================================'
echo ' CONDITION 2/3: FULL PIPELINE (SigLIP-SAM + IG Bias)'
echo " Started: $(date)"
echo '============================================================'
python experiments/robot/run_libero_eval_with_sam.py \
    --config experiments/robot/configs/config_ablation_full.yaml
echo " Full pipeline completed: $(date)"

# ---- Condition 3: TeamVLA ----
echo ''
echo '============================================================'
echo ' CONDITION 3/3: TEAMVLA TOKEN PRUNING'
echo " Started: $(date)"
echo '============================================================'
python experiments/robot/team/run_libero_eval_team.py \
    --config experiments/robot/configs/config_ablation_teamvla.yaml
echo " TeamVLA completed: $(date)"

# ---- Summary ----
echo ''
echo '============================================================'
echo ' ALL CONDITIONS COMPLETE'
echo " Finished: $(date)"
echo '============================================================'
echo ''
echo 'JSON summaries:'
for f in eval_results/ablation/*/summary_libero_spatial.json; do
    echo "--- $f ---"
    cat "$f"
    echo ''
done
