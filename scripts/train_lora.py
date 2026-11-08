"""openpi scripts/train.py 的包装：先注册 scripts/train_configs.py 里的配置，再调原来的 main。
  cd $OPENPI && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python $ROOT/scripts/train_lora.py pi05_libero10_lora_n10 --exp-name run1 --overwrite
其余命令行参数和 openpi 一样（tyro），比如 --num-train-steps 50 --batch-size 8 做冒烟。
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_configs  # noqa: F401  注册配置 + 子集 patch
import openpi.training.config as _config
OPENPI = pathlib.Path(__file__).resolve().parents[1] / "external/openpi"
sys.path.insert(0, str(OPENPI / "scripts"))
import train  # openpi/scripts/train.py

if __name__ == "__main__":
    train.main(_config.cli())
