#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --partition=h100,h200,a100
#SBATCH --qos=jhu
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%j.out
# 第 2 周 LoRA 微调（计划 §2）。配置在 scripts/train_configs.py。
#   CONFIG=pi05_libero10_lora_n10 EXP=run1 sbatch scripts/train.sbatch
#   CONFIG=pi05_libero10_lora_n10 EXP=smoke SMOKE=1 sbatch scripts/train.sbatch        # 50 步、batch 8，只验证跑通
#   RESUME=1 → 从 checkpoint 续跑（requeue 后也自动用它）；否则 --overwrite
# norm stats 不存在会先算（compute_norm_stats_ext），结果在 assets/<CONFIG>/<asset_id>/，进 git。
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
CONFIG=${CONFIG:?set CONFIG}; EXP=${EXP:?set EXP}
cd $OPENPI
ASSET=$(uv run python - <<PY
import sys; sys.path.insert(0, "$ROOT/scripts"); import train_configs, openpi.training.config as c
cfg = c.get_config("$CONFIG"); print(cfg.data.create(cfg.assets_dirs, cfg.model).asset_id)
PY
)
ASSET=$(echo "$ASSET" | tail -1)
if [ ! -f $ROOT/assets/$CONFIG/$ASSET/norm_stats.json ]; then
  uv run python $ROOT/scripts/compute_norm_stats_ext.py --config-name $CONFIG || exit 1
fi
CKPT=/weka/projects/rzhao36/lwang/checkpoints/learnVLA/$CONFIG/$EXP
if { [ -n "$RESUME" ] || [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; } && [ -d "$CKPT" ]; then MODE=--resume; else MODE=--overwrite; fi
EXTRA=""; [ -n "$SMOKE" ] && EXTRA="--num-train-steps 50 --batch-size 8 --save-interval 25 --log-interval 5"
uv run python $ROOT/scripts/train_lora.py $CONFIG --exp-name $EXP $MODE $EXTRA
RC=$?
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
exit $RC
