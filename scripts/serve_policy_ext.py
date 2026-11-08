"""策略 server。改自 openpi/scripts/serve_policy.py，只加了一件事：
--num-steps 控制 flow matching 去噪步数（官方脚本没暴露，§1.3 要扫它）。
在 openpi 的 uv 环境里跑：cd $OPENPI && uv run python $ROOT/scripts/serve_policy_ext.py ...
"""
import dataclasses
import logging
import socket

import tyro
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_configs  # noqa: F401  让 --config 也能用第 2 周的微调配置
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    dir: str = "gs://openpi-assets/checkpoints/pi05_libero"
    port: int = 8000
    num_steps: int = 10        # flow matching 去噪步数；pi0.py sample_actions 默认 10
    default_prompt: str | None = None


def main(args: Args) -> None:
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )
    logging.info("serving %s from %s on %s:%d num_steps=%d", args.config, args.dir, socket.gethostname(), args.port, args.num_steps)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port,
        metadata={**policy.metadata, "config": args.config, "dir": args.dir, "num_steps": args.num_steps},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
