# 所有 sbatch 脚本 source 这个文件。只放路径和环境变量，不做任何重活。
export ROOT=/weka/projects/rzhao36/lwang/codes/learnVLA
export OPENPI=$ROOT/external/openpi
export LIBERO_VENV=$OPENPI/examples/libero/.venv
export LIBERO_CONFIG_PATH=$ROOT/libero_config
export PYTHONPATH=$OPENPI/third_party/libero${PYTHONPATH:+:$PYTHONPATH}

# uv 的缓存和托管 Python 都放 weka，别占 $HOME 配额
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/weka/projects/rzhao36/lwang/.uv_cache
export UV_PYTHON_INSTALL_DIR=/weka/projects/rzhao36/lwang/.uv_python

# openpi 的 checkpoint / tokenizer 缓存（默认 ~/.cache/openpi）
export OPENPI_DATA_HOME=/weka/projects/rzhao36/lwang/datasets/models/openpi
export HF_HOME=/weka/projects/rzhao36/lwang/datasets/models/huggingface
export HF_HUB_DISABLE_XET=1

# 无头渲染
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# JAX 显存：默认只拿 75%，训练时要 0.9；评测用默认即可
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
