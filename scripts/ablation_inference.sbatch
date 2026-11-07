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
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%A_%a.out
# §1.3 推理期消融：官方 checkpoint，libero_10，只改推理参数。
# array 索引 → (replan_steps, num_steps)：
#   0: (5,10) 基线   1: (1,10)   2: (2,10)   3: (10,10) 整 chunk 开环
#   4: (5,3)         5: (5,20)
# 区分的假设：失败来自"预测错"（开环与闭环接近）还是"不重新观测、不纠偏"（开环大跌）；
#             动作质量是否受采样精度限制（num_steps 3 vs 10 vs 20）。
#   sbatch --array=0-5 scripts/ablation_inference.sbatch
set -x
source /weka/projects/rzhao36/lwang/codes/learnVLA/scripts/weka_guard.sh
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
