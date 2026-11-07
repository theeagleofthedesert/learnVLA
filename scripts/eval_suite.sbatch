#!/bin/bash
#SBATCH --job-name=vla_eval
#SBATCH --partition=l40s
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%A_%a.out
# 一个 array task 评一个 suite。默认评官方 checkpoint。
#   sbatch --array=0-3 scripts/eval_suite.sbatch                    # 四个 suite
#   sbatch --array=3 scripts/eval_suite.sbatch                      # 只跑 libero_10
#   TRIALS=10 REPLAN=10 OUT=results/ablation/replan10 sbatch --array=3 scripts/eval_suite.sbatch
# 其余参数（CKPT CONFIG NUM_STEPS TASKS INIT_IDS REPEATS VIDEOS TAG SEED）直接作为环境变量传给 run_eval.sh。
set -x
source /weka/projects/rzhao36/lwang/codes/learnVLA/scripts/weka_guard.sh
source $ROOT/scripts/env.sh
module load cuda/12.6.3
SUITES=(libero_spatial libero_object libero_goal libero_10)
export SUITE=${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}
export OUT=${OUT:-$ROOT/results/eval_official}/$SUITE
export TAG=${TAG:-official}
bash $ROOT/scripts/run_eval.sh
