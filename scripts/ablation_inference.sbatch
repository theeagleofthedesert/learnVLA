#!/bin/bash
#SBATCH --job-name=vla_abl
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
# §1.3 推理期消融：官方 checkpoint，libero_10，只改推理参数。
# array 索引 → (replan_steps, num_steps)：
#   0: (5,10) 基线   1: (1,10)   2: (2,10)   3: (10,10) 整 chunk 开环
#   4: (5,3)         5: (5,20)
# 区分的假设：失败来自"预测错"（开环与闭环接近）还是"不重新观测、不纠偏"（开环大跌）；
#             动作质量是否受采样精度限制（num_steps 3 vs 10 vs 20）。
#   sbatch --array=0-5 scripts/ablation_inference.sbatch
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
REPLANS=(5 1 2 10 5 5)
NSTEPS=(10 10 10 10 3 20)
i=${SLURM_ARRAY_TASK_ID:-0}
export REPLAN=${REPLANS[$i]} NUM_STEPS=${NSTEPS[$i]}
export SUITE=${SUITE:-libero_10} TRIALS=${TRIALS:-50}
export TAG=replan${REPLAN}_nsteps${NUM_STEPS}
export OUT=$ROOT/results/ablation_inference/$TAG
bash $ROOT/scripts/run_eval.sh
