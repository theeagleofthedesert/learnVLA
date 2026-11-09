"""第 2 周的训练配置，注册进 openpi 的配置表，不改 external/openpi 里的任何文件。

用法：任何要用这些配置的脚本（train_lora.py / compute_norm_stats_ext.py / serve_policy_ext.py）先 `import train_configs`。
配置名：
  pi05_libero10_lora_n50     libero_10 全部示教（每任务 29–49 条，共 379 条）      槽位 A
  pi05_libero10_lora_n10     每任务 10 条，seed 0                                   槽位 B
  pi05_libero10_lora_n2      每任务 2 条，seed 0                                    槽位 C
  pi05_libero10_lora_n10_s1  每任务 10 条，seed 1（训练方差）                       槽位 D
和官方 pi05_libero 的差别：LoRA（paligemma gemma_2b_lora + action expert gemma_300m_lora）、只用 libero_10、batch 32、
默认学习率表（warmup 1k，peak 2.5e-5）、ema 关、每 10k 步存一个 checkpoint 且都保留。其余（pi05、horizon 10、不读 state、
不做 delta 变换、quantile 归一化、从 pi05_base 起）与官方一致。

示教子采样怎么做：数据集还是 physical-intelligence/libero 这一份；LiberoSubsetDataConfig.create() 算出要用的 episode 索引，
塞进 DataConfig 的 episodes 字段；这里 monkeypatch 了 openpi 的 create_torch_dataset，把 episodes 传给 LeRobotDataset(episodes=...)。
norm stats 也只在子集上算（compute_norm_stats 走同一个函数）。
"""
import dataclasses
import json
import pathlib
import random

import torch

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from lerobot.common.datasets import lerobot_dataset
import openpi.transforms as _transforms

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT_BASE = "/weka/projects/rzhao36/lwang/checkpoints/learnVLA"
LIBERO_REPO = "physical-intelligence/libero"
LIBERO_META = pathlib.Path("/weka/projects/rzhao36/lwang/datasets/lerobot") / LIBERO_REPO / "meta"

# libero_10 的 10 条指令（数据集里 task 字符串就是指令本身）
LIBERO_10_TASKS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]


def select_episodes(tasks: list[str], n_per_task: int | None, seed: int) -> list[int]:
    """按任务分层抽样。n_per_task=None → 该任务全部。只读 meta/episodes.jsonl，不碰 parquet。"""
    eps = [json.loads(l) for l in open(LIBERO_META / "episodes.jsonl")]
    rng = random.Random(seed)
    out = []
    for t in tasks:
        ids = sorted(e["episode_index"] for e in eps if e["tasks"][0] == t)
        out += ids if n_per_task is None or n_per_task >= len(ids) else rng.sample(ids, n_per_task)
    return sorted(out)


@dataclasses.dataclass(frozen=True)
class SubsetDataConfig(_config.DataConfig):
    episodes: tuple[int, ...] | None = None


@dataclasses.dataclass(frozen=True)
class LiberoSubsetDataConfig(_config.LeRobotLiberoDataConfig):
    n_per_task: int | None = None   # None = 全部
    subset_seed: int = 0

    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        eps = select_episodes(LIBERO_10_TASKS, self.n_per_task, self.subset_seed)
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        return SubsetDataConfig(**fields, episodes=tuple(eps))


# --- monkeypatch：让 openpi 的数据集构造接受 episodes ---
_orig_create_torch_dataset = _data_loader.create_torch_dataset


def _create_torch_dataset(data_config, action_horizon, model_config):
    episodes = getattr(data_config, "episodes", None)
    if episodes is None or data_config.repo_id in (None, "fake"):
        return _orig_create_torch_dataset(data_config, action_horizon, model_config)
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    ds = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        episodes=list(episodes),
        delta_timestamps={k: [t / meta.fps for t in range(action_horizon)] for k in data_config.action_sequence_keys},
    )
    _fix_episode_data_index(ds, episodes, meta.total_episodes)
    if data_config.prompt_from_task:
        ds = _data_loader.TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    return ds


def _fix_episode_data_index(ds, episodes, total_episodes):
    """绕开这一版 LeRobot 的子集 bug（notes/00 错误 7）。
    传 episodes 时 episode_data_index 的第 j 行是"子集里第 j 条"的帧范围，但 __getitem__ 用原始 episode 编号去查。
    编号 >= 子集大小时越界报错；编号更小时不报错，却拿到另一条 episode 的范围，动作序列被悄悄截错。
    这里把表摊开成按原始编号索引的全长表（没选中的行是 0，不会被查到）。
    """
    idx = torch.as_tensor(list(episodes))
    full = {}
    for key, val in ds.episode_data_index.items():
        full[key] = torch.zeros(total_episodes, dtype=val.dtype)
        full[key][idx] = val
    ds.episode_data_index = full


_data_loader.create_torch_dataset = _create_torch_dataset


def _lora_config(name: str, n_per_task: int | None, subset_seed: int, train_seed: int = 42) -> _config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    )
    tag = f"libero_10_n{n_per_task if n_per_task else 'all'}_s{subset_seed}"
    return _config.TrainConfig(
        name=name,
        model=model,
        data=LiberoSubsetDataConfig(
            repo_id=LIBERO_REPO,
            assets=_config.AssetsConfig(asset_id=tag),           # norm stats 存在 assets/<name>/<tag>/
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            n_per_task=n_per_task, subset_seed=subset_seed,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
        seed=train_seed,
        wandb_enabled=False,
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=CKPT_BASE,
        num_workers=4,
    )


MY_CONFIGS = [
    _lora_config("pi05_libero10_lora_n50", None, 0),
    _lora_config("pi05_libero10_lora_n10", 10, 0),
    _lora_config("pi05_libero10_lora_n2", 2, 0),
    _lora_config("pi05_libero10_lora_n10_s1", 10, 1, train_seed=43),
]
for c in MY_CONFIGS:
    _config._CONFIGS_DICT[c.name] = c
