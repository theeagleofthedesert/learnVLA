#!/bin/bash
# 在同一个节点上起 server + client 跑一次评测。被 sbatch 脚本或 setup 调用，也可在交互 GPU 节点直接跑。
# 通过环境变量传参（都有默认值）：
#   CONFIG=pi05_libero  CKPT=gs://openpi-assets/checkpoints/pi05_libero  NUM_STEPS=10
#   SUITE=libero_spatial  TASKS="0 1 2"(空=全部)  TRIALS=50  INIT_IDS="0 3"(空=前 TRIALS 个)  REPEATS=1  REPLAN=5
#   OUT=results/xxx  VIDEOS=failure|all|none  TAG=任意字符串  SEED=7
set -o pipefail
source /weka/projects/rzhao36/lwang/codes/learnVLA/scripts/env.sh
# PLUS=1 → 用 LIBERO-Plus（独立 venv + 独立路径配置 + ImageMagick），否则原版 LIBERO
if [ "${PLUS:-0}" = "1" ]; then
  LIBERO_VENV=$ROOT/external/LIBERO-plus/.venv
  export LIBERO_CONFIG_PATH=$ROOT/libero_config_plus
  export PYTHONPATH=$ROOT/external/LIBERO-plus
  export MAGICK_HOME=/weka/projects/rzhao36/lwang/miniforge3/envs/imagemagick
  export LD_LIBRARY_PATH=$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
CONFIG=${CONFIG:-pi05_libero}; CKPT=${CKPT:-gs://openpi-assets/checkpoints/pi05_libero}; NUM_STEPS=${NUM_STEPS:-10}
SUITE=${SUITE:-libero_spatial}; TRIALS=${TRIALS:-50}; REPEATS=${REPEATS:-1}; REPLAN=${REPLAN:-5}
OUT=${OUT:-$ROOT/results/eval}; VIDEOS=${VIDEOS:-failure}; TAG=${TAG:-$CONFIG}; SEED=${SEED:-7}
PORT=${PORT:-$((20000 + RANDOM % 20000))}   # 随机端口，避免同节点两个作业撞上
mkdir -p $OUT

# --- server（openpi 环境）---
cd $OPENPI
uv run python $ROOT/scripts/serve_policy_ext.py --config $CONFIG --dir $CKPT --port $PORT --num-steps $NUM_STEPS \
  > $OUT/server_${SLURM_JOB_ID:-local}.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
echo "server pid=$SERVER_PID port=$PORT log=$OUT/server_${SLURM_JOB_ID:-local}.log"
# 等端口打开（加载 + JIT 编译要几分钟）
for i in $(seq 1 120); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo SERVER_DIED; tail -30 $OUT/server_${SLURM_JOB_ID:-local}.log; exit 1; }
  sleep 10
done
(echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo SERVER_TIMEOUT; exit 1; }
echo "server up after ~$((i*10))s"

# --- client（LIBERO 环境）---
source $LIBERO_VENV/bin/activate
ARGS="--host 127.0.0.1 --port $PORT --task-suite-name $SUITE --num-trials-per-task $TRIALS --repeats $REPEATS \
      --replan-steps $REPLAN --out-dir $OUT --save-videos $VIDEOS --tag $TAG --seed $SEED"
[ -n "$TASKS" ] && ARGS="$ARGS --task-ids $TASKS"
[ -n "$INIT_IDS" ] && ARGS="$ARGS --init-state-ids $INIT_IDS"
echo "client args: $ARGS"
python $ROOT/scripts/eval_libero.py $ARGS 2>&1 | tee -a $OUT/client_${SLURM_JOB_ID:-local}.log
RC=${PIPESTATUS[0]}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
exit $RC
