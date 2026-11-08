#!/bin/bash
#SBATCH --job-name=vla_plus_eval
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
# §1.4 LIBERO-Plus 扰动评测。每个 array task 一个 suite；每类扰动抽 N 个变体任务，各跑 1 个 episode（官方协议）。
#   sbatch --array=0-3 scripts/eval_plus.sbatch                     # 粗扫：7 类 × 30 个 = 210 episode / suite
#   N=0 CATEGORY="Camera Viewpoints" sbatch --array=3 scripts/eval_plus.sbatch   # 某一类全量
#   N=0 RESUME=1 sbatch --array=3 scripts/eval_plus.sbatch           # 超时后续跑：跳过 OUT/episodes.csv 里已有的任务
# 耗时参考（a100，全量 N=0）：spatial 8.3h / object 10.3h / goal 9.5h / libero_10 >12h（超时，需 RESUME）
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
set -x
source $ROOT/scripts/env.sh
module load cuda/12.6.3
SUITES=(libero_spatial libero_object libero_goal libero_10)
export PLUS=1 SUITE=${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}
N=${N:-30}; CATEGORY=${CATEGORY:-all}; SEL_SEED=${SEL_SEED:-0}
# 选任务要在 plus venv 里跑
export LIBERO_CONFIG_PATH=$ROOT/libero_config_plus PYTHONPATH=$ROOT/external/LIBERO-plus
export MAGICK_HOME=/weka/projects/rzhao36/lwang/miniforge3/envs/imagemagick
export LD_LIBRARY_PATH=$MAGICK_HOME/lib   # 不能和上一行合并：同一行里 $MAGICK_HOME 在赋值前展开，会变成 /lib
export TRIALS=1 TAG=${TAG:-plus_official} VIDEOS=${VIDEOS:-failure}
export OUT=${OUT:-$ROOT/results/libero_plus}/$SUITE
EXCL=${RESUME:+--exclude-csv $OUT/episodes.csv}
SEL=$($ROOT/external/LIBERO-plus/.venv/bin/python $ROOT/scripts/plus_select.py --suite $SUITE --category "$CATEGORY" --n $N --seed $SEL_SEED $EXCL) || { echo "SELECT_FAILED"; exit 1; }
export TASKS=$(echo "$SEL" | tail -1)
echo "selected $(echo $TASKS | wc -w) tasks"
[ -z "$TASKS" ] && { echo "nothing to run (all tasks already in $OUT/episodes.csv)"; exit 0; }
bash $ROOT/scripts/run_eval.sh
