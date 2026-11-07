#!/bin/bash
#SBATCH --job-name=vla_stoch
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
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%j.out
# §1.5 随机性分离：固定初始状态，重复 K 次（采样噪声每次不同，仿真本身确定）。
# 默认 libero_10 全部 10 任务 × 前 20 个初始状态 × 10 次 = 2000 episode。
# 读法：python3 scripts/summarize.py --by init results/stochasticity/episodes.csv
#   双峰（全成/全败）占比高 → 失败由初始状态决定（数据侧）；散在中间 → 策略采样随机（模型侧）。
#   sbatch scripts/stochasticity.sbatch
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
export SUITE=${SUITE:-libero_10} TRIALS=${TRIALS:-20} REPEATS=${REPEATS:-10} VIDEOS=${VIDEOS:-none}
export TAG=stoch_k${REPEATS} OUT=$ROOT/results/stochasticity
bash $ROOT/scripts/run_eval.sh
