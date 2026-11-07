#!/bin/bash
#SBATCH --job-name=vla_plus
#SBATCH --partition=h100,h200,a100
#SBATCH --qos=jhu
#SBATCH --account=rzhao36_vifinav
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jhu/lwang231/vla_logs/%x_%j.out
# LIBERO-Plus 环境：独立 py3.8 venv（robosuite 钉 1.4.0，多了 wand/scikit-image/usd-core）
# + conda-forge 的 ImageMagick（wand 需要 libMagickWand，系统没有、也没 apt）
# + 6.4 GB assets.zip（新物体、纹理、场景）解压到 LIBERO-plus/libero/libero/assets
#   sbatch scripts/setup_libero_plus.sbatch
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
PLUS=$ROOT/external/LIBERO-plus

# ---------- 1. ImageMagick (conda) ----------
source /weka/projects/rzhao36/lwang/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^imagemagick " || conda create -y -n imagemagick -c conda-forge imagemagick || { echo MAGICK_FAIL; exit 1; }
export MAGICK_HOME=/weka/projects/rzhao36/lwang/miniforge3/envs/imagemagick
export LD_LIBRARY_PATH=$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ls $MAGICK_HOME/lib/libMagickWand* | head -2 || { echo MAGICK_LIB_MISSING; exit 1; }

# ---------- 2. venv ----------
cd $PLUS
[ -d .venv ] || uv venv --python 3.8 .venv || exit 1
source .venv/bin/activate
# 复用 openpi 给 LIBERO 钉好的版本（torch cu113 等），再叠 LIBERO-plus 自己的 requirements 与 extra
uv pip install "cmake<4"   # egl-probe (robosuite 1.4.0 依赖) 从源码编；它的 CMakeLists 要求 cmake_minimum_required<3.5，cmake 4 拒绝
uv pip install -r $OPENPI/examples/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match || exit 1
uv pip install -r requirements.txt -r extra_requirements.txt --index-strategy=unsafe-best-match || { echo PLUS_REQ_FAIL; exit 1; }
uv pip install -e $OPENPI/packages/openpi-client || exit 1
uv pip install -e . || { echo PLUS_INSTALL_FAIL; exit 1; }
python -c "import wand.image; print('wand ok')" || { echo WAND_FAIL; exit 1; }

# ---------- 3. assets ----------
if [ ! -d libero/libero/assets/textures ]; then
  mkdir -p /weka/projects/rzhao36/lwang/datasets/libero_plus && cd /weka/projects/rzhao36/lwang/datasets/libero_plus
  for i in 1 2 3; do
    uv run --no-project --with huggingface_hub python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Sylvest/LIBERO-plus', 'assets.zip', repo_type='dataset', local_dir='.'))" && break
    echo "download attempt $i failed; retry in 60s"; sleep 60
  done
  [ -f assets.zip ] || { echo ASSETS_DOWNLOAD_FAIL; exit 1; }
  unzip -q -o assets.zip -d $PLUS/libero/libero/ || { echo UNZIP_FAIL; exit 1; }
  cd $PLUS
fi
ls libero/libero/assets | head; du -sh libero/libero/assets

# ---------- 4. 测试：建一个扰动任务的环境并渲染 ----------
export LIBERO_CONFIG_PATH=$ROOT/libero_config_plus PYTHONPATH=$PLUS
python - <<'PY' || { echo PLUS_RENDER_FAIL; exit 1; }
import pathlib, json, collections
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
print("n_tasks", suite.n_tasks)
cls = json.load(open(pathlib.Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"))["libero_spatial"]
bycat = collections.defaultdict(list)
for t in cls: bycat[t["category"]].append(t["name"])
for cat, names in bycat.items():
    i = [k for k in range(suite.n_tasks) if suite.get_task(k).name == names[0]][0]
    task = suite.get_task(i)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7); env.reset()
    obs = env.set_init_state(suite.get_task_init_states(i)[0])
    print(f"{cat:22s} task {i:5d} {task.name[:60]:60s} img mean {obs['agentview_image'].mean():.1f}  lang: {task.language[:60]}")
    env.close()
PY
echo PLUS_ALL_OK
