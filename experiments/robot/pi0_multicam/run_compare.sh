#!/bin/bash
# run_compare.sh — resumable multi-camera method comparison on pi0.5/RoboTwin.
# 5 conditions (base/t3r/adp/fastv/team) x TASKS, matched budget from cfg_compare.txt.
# Resumable: skips (task,name) cells already present in the results CSV (survives pod suspend/restart).
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd /workspace/RoboTwin
source /workspace/rtenv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DISPLAY=""
export PYTHONWARNINGS=ignore
export EVAL_TEST_NUM=${EVAL_TEST_NUM:-12}
RES=${RES:-/workspace/compare_results.csv}
touch $RES
TASKS="${TASKS:-click_bell press_stapler click_alarmclock}"
echo "[compare] tasks=$TASKS testnum=$EVAL_TEST_NUM"
for t in $TASKS; do
  while IFS='|' read -r name json; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    if grep -q "^$t,$name," $RES; then echo "SKIP $t $name (already done)"; continue; fi
    echo "$json" > /workspace/t3r_runtime.json
    echo "=== RUN $t | $name | $json | $(date +%H:%M:%S) ==="
    logf=/workspace/cmp_${t}_${name}.log
    python -u script/eval_policy.py --config policy/pi0/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name x --model_name x --ckpt_setting x --seed 0 --policy_name pi0 > $logf 2>&1
    sr=$(grep -a 'Success rate' $logf | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
    tok=$(grep -a 't3r_pi0\] method' /workspace/pi05_server.log | tail -1 | sed 's/.*img_tokens=/tok=/; s/ prefix.*//')
    echo "RESULT $t | $name | $sr | $tok"
    echo "$t,$name,\"$json\",$sr,$tok" >> $RES
  done < ${CFG:-/workspace/cfg_compare.txt}
done
echo "==== COMPARE SUMMARY ===="
cat $RES
echo "COMPARE_DONE"
