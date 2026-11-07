#!/bin/bash
#SBATCH --job-name=vla_stoch
#SBATCH --partition=l40s
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%j.out
# §1.5 随机性分离：固定初始状态，重复 K 次（采样噪声每次不同，仿真本身确定）。
# 默认 libero_10 全部 10 任务 × 前 20 个初始状态 × 10 次 = 2000 episode。
# 读法：python3 scripts/summarize.py --by init results/stochasticity/episodes.csv
#   双峰（全成/全败）占比高 → 失败由初始状态决定（数据侧）；散在中间 → 策略采样随机（模型侧）。
#   sbatch scripts/stochasticity.sbatch
set -x
source /weka/projects/rzhao36/lwang/codes/learnVLA/scripts/weka_guard.sh
source $ROOT/scripts/env.sh
module load cuda/12.6.3
export SUITE=${SUITE:-libero_10} TRIALS=${TRIALS:-20} REPEATS=${REPEATS:-10} VIDEOS=${VIDEOS:-none}
export TAG=stoch_k${REPEATS} OUT=$ROOT/results/stochasticity
bash $ROOT/scripts/run_eval.sh
