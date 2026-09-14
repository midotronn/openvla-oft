#!/bin/bash
# run_sweep.sh TASK CFGFILE RESULTS
# Iterates configs (name|json per line), writing /workspace/t3r_runtime.json before each eval.
# One running pi0.5 server serves all configs (config switched via the runtime json file).
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd /workspace/RoboTwin
source /workspace/rtenv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DISPLAY=""
export PYTHONWARNINGS=ignore
export EVAL_TEST_NUM=${EVAL_TEST_NUM:-8}

TASK=$1
CFGFILE=$2
RESULTS=$3
echo "[sweep] task=$TASK testnum=$EVAL_TEST_NUM cfgfile=$CFGFILE"

while IFS='|' read -r name json; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  echo "$json" > /workspace/t3r_runtime.json
  echo "=== RUN task=$TASK cfg=$name json=$json testnum=$EVAL_TEST_NUM $(date +%H:%M:%S) ==="
  logf=/workspace/sweep_${TASK}_${name}.log
  python -u script/eval_policy.py --config policy/pi0/deploy_policy.yml --overrides \
    --task_name $TASK --task_config demo_clean \
    --train_config_name x --model_name x --ckpt_setting x --seed 0 --policy_name pi0 > $logf 2>&1
  sr=$(grep -a 'Success rate' $logf | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
  tok=$(grep -a 't3r_pi0\] method' /workspace/pi05_server.log | tail -1 | sed 's/.*img_tokens=/tok=/; s/ prefix.*//')
  echo "RESULT $TASK | $name | $sr | $tok"
  echo "$TASK,$name,\"$json\",$sr,$tok" >> $RESULTS
done < $CFGFILE
echo "SWEEP_DONE $TASK"
