#!/bin/bash
#SBATCH --job-name=vla_eval
#SBATCH --partition=h100,h200,a100
#SBATCH --qos=jhu
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%A_%a.out
# 一个 array task 评一个 suite。默认评官方 checkpoint。
#   sbatch --array=0-3 scripts/eval_suite.sbatch                    # 四个 suite
#   sbatch --array=3 scripts/eval_suite.sbatch                      # 只跑 libero_10
#   TRIALS=10 REPLAN=10 OUT=results/ablation/replan10 sbatch --array=3 scripts/eval_suite.sbatch
# 其余参数（CKPT CONFIG NUM_STEPS TASKS INIT_IDS REPEATS VIDEOS TAG SEED）直接作为环境变量传给 run_eval.sh。
set -x
# --- weka guard（内联，不能 source /weka 上的文件：掉挂载时它自己也不在）---
ROOT=/weka/projects/rzhao36/lwang/codes/learnVLA
JID=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; JID=${JID:-$SLURM_JOB_ID}
for i in $(seq 1 30); do timeout 10 test -d $ROOT/logs && break; sleep 10; done
if ! timeout 10 test -d $ROOT/logs; then
  echo "WEKA_ABSENT on $(hostname) at $(date +%T): requeue with this node excluded"
  cur=$(scontrol show job $JID -o 2>/dev/null | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p'); [ -z "$cur" -o "$cur" = "(null)" ] && cur=$(hostname) || cur="$cur,$(hostname)"
  scontrol requeue $JID || { echo REQUEUE_FAILED; exit 1; }; for i in $(seq 1 15); do sleep 1; scontrol update JobId=$JID ExcNodeList=$cur 2>/dev/null && break; done; exit 0
fi
LOGF=$(scontrol show job $JID -o 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p'); trap '[ -n "$LOGF" ] && cp -f "$LOGF" $ROOT/logs/ 2>/dev/null' EXIT
echo "host=$(hostname) job=$JID start=$(date)"
# --- end weka guard ---
source $ROOT/scripts/env.sh
module load cuda/12.6.3
SUITES=(libero_spatial libero_object libero_goal libero_10)
export SUITE=${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}
export OUT=${OUT:-$ROOT/results/eval_official}/$SUITE
export TAG=${TAG:-official}
bash $ROOT/scripts/run_eval.sh
