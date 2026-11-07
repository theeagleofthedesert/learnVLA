#!/bin/bash
#SBATCH --job-name=vla_setup
#SBATCH --partition=h100,h200,a100
#SBATCH --qos=jhu
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%j.out
# 建两个环境并做端到端冒烟测试：
#   1. openpi (Python 3.11, uv sync)        —— 跑策略 server
#   2. examples/libero/.venv (Python 3.8)   —— 跑 LIBERO 仿真 client
# 然后下载 pi05_libero checkpoint，起 server，跑 1 任务 x 2 episode。
# 提交：sbatch scripts/setup_envs.sbatch
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
nvidia-smi -L; module load cuda/12.6.3
uv --version

# ---------- 1. openpi 环境 ----------
cd $OPENPI
GIT_LFS_SKIP_SMUDGE=1 uv sync || { echo OPENPI_SYNC_FAIL; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . || { echo OPENPI_PIP_FAIL; exit 1; }
uv run python -c "
import jax; print('jax devices', jax.devices())
import openpi.training.config as c; cfg = c.get_config('pi05_libero'); print(cfg.model)
" || { echo OPENPI_IMPORT_FAIL; exit 1; }
echo OPENPI_ENV_OK

# ---------- 2. LIBERO 环境 ----------
[ -d $LIBERO_VENV ] || uv venv --python 3.8 $LIBERO_VENV || { echo LIBERO_VENV_FAIL; exit 1; }
source $LIBERO_VENV/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match || { echo LIBERO_SYNC_FAIL; exit 1; }
uv pip install -e packages/openpi-client || exit 1
uv pip install -e third_party/libero || exit 1
python -c "
from libero.libero import benchmark, get_libero_path
print('suites', list(benchmark.get_benchmark_dict().keys()))
print('bddl', get_libero_path('bddl_files'))
" || { echo LIBERO_IMPORT_FAIL; exit 1; }
# EGL 渲染测试：建一个环境，渲染一帧
python - <<'PY' || { echo LIBERO_RENDER_FAIL; exit 1; }
import pathlib, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(7); env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
print("agentview", obs["agentview_image"].shape, obs["agentview_image"].dtype, "mean", obs["agentview_image"].mean())
print("state keys", [k for k in obs if not k.endswith("image")])
env.close()
PY
deactivate
echo LIBERO_ENV_OK

# ---------- 3. 下载 checkpoint ----------
cd $OPENPI
uv run python -c "
import openpi.shared.download as d
p = d.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'); print('ckpt at', p)
import os; [print(r, len(f), 'files') for r, _, f in os.walk(p)]
" || { echo CKPT_DOWNLOAD_FAIL; exit 1; }
du -sh $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero

# ---------- 4. 冒烟评测：1 任务 x 2 episode ----------
SUITE=libero_spatial TASKS=0 TRIALS=2 VIDEOS=all OUT=$ROOT/results/smoke \
  bash $ROOT/scripts/run_eval.sh || { echo SMOKE_FAIL; exit 1; }
cat $ROOT/results/smoke/episodes.csv
echo SETUP_ALL_OK
