#!/bin/bash
#SBATCH --job-name=vla_stall
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
# notes/07 停滞来源对照：官方 checkpoint，libero_10，只改"相邻两次采样之间的关系"。
# array 索引 → 条件：
#   0: k=5 原样（带轨迹和 probe 的基线）   1: k=5 噪声×0（确定性）      2: k=5 一集内复用同一份噪声
#   3: k=5 RTC                             4: k=1 RTC（阳性对照）       5: k=5 噪声×0.5
#   6: k=1 一集内复用同一份噪声            7: k=1 噪声×0                8: k=1 原样（带轨迹）
# 区分的假设：最优 k 下剩余的停滞，是"相邻采样互相推翻"（RTC / 复用噪声能救），
#             还是"单次采样本身落在等待上"（降噪声能救），还是都不是（状态本身已经走偏）。
#   sbatch --array=0-4,6 scripts/stall_study.sbatch
#   冒烟：TASKS="0 8" TRIALS=3 OUT_BASE=results/stall_study_smoke sbatch --array=0,3 --time=1:00:00 scripts/stall_study.sbatch
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
REPLANS=(5 5 5 5 1 5 1 1 1)
SCALES=(1 0 1 1 1 0.5 1 0 1)
MODES=(fresh fresh episode fresh fresh fresh episode fresh fresh)
RTCS=(0 0 0 1 1 0 0 0 0)
NAMES=(k5_base k5_noise0 k5_epnoise k5_rtc k1_rtc k5_noise0.5 k1_epnoise k1_noise0 k1_base)
i=${SLURM_ARRAY_TASK_ID:-0}
export REPLAN=${REPLANS[$i]} NOISE_SCALE=${SCALES[$i]} NOISE_MODE=${MODES[$i]} RTC=${RTCS[$i]} NUM_STEPS=10
export SUITE=${SUITE:-libero_10} TRIALS=${TRIALS:-50} VIDEOS=${VIDEOS:-none}
export LOG_TRAJ=1 PROBE_EVERY=${PROBE_EVERY:-25} PROBE_N=16
export TAG=${NAMES[$i]}
export OUT=$ROOT/${OUT_BASE:-results/stall_study}/$TAG
bash $ROOT/scripts/run_eval.sh
