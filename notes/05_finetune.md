# notes/05 微调：LoRA 挂在哪、数据怎么抽、怎么评（提纲，随第 2 周填）

> 读者：lwang。目的：第 2 周最多 4 次 LoRA 微调（计划 §2）。这篇先把"改了什么、在哪改、怎么跑"定下来，结果出来后补 §4。

## 0. 四个槽位

| 槽位 | 配置名 | 示教 / 任务 | 共 | 目的 |
|---|---|---|---|---|
| A | `pi05_libero10_lora_n50` | 全部（29–49） | 379 | 复现；和官方 checkpoint 在 libero_10 上比 |
| B | `pi05_libero10_lora_n10` | 10（seed 0） | 100 | 数据效率第一点 |
| C | `pi05_libero10_lora_n2` | 2（seed 0） | 20 | 数据效率第二点 |
| D | `pi05_libero10_lora_n10_s1` | 10（seed 1，训练 seed 43） | 100 | 训练方差 |

只用 libero_10。注意官方 `pi05_libero` 是在 4 个 suite 共 1693 条示教上**全量**微调的，A 不是严格复现，是"同一方法、只用 1/4 数据、LoRA"的对照。

## 1. 配置在哪、和官方差在哪

配置不改 `external/openpi`，全部在 [`scripts/train_configs.py`](../scripts/train_configs.py)，`import train_configs` 后注册进 openpi 的配置表。与 `pi05_libero`（[`config.py:744`](../external/openpi/src/openpi/training/config.py#L744)）逐项对比：

| 项 | 官方 pi05_libero | 我们的 LoRA 配置 | 为什么 |
|---|---|---|---|
| `paligemma_variant` | `gemma_2b` | `gemma_2b_lora` | 全量 >70 GB 显存；LoRA 22.5 GB |
| `action_expert_variant` | `gemma_300m` | `gemma_300m_lora` | 同上 |
| `freeze_filter` | 无 | `Pi0Config(...).get_freeze_filter()`：冻结所有非 `.*lora.*` 参数 | [`pi0_config.py:88-113`](../external/openpi/src/openpi/models/pi0_config.py#L88) |
| `ema_decay` | 0.999 | None | openpi 的 LoRA 示例都关 EMA |
| `batch_size` | 256 | 32 | 单卡 |
| `lr_schedule` | warmup 10k, peak 5e-5 | warmup 1k, peak 2.5e-5, decay 30k → 2.5e-6 | openpi 默认值 |
| 数据 | 4 suite 1693 条 | libero_10 子集 | 计划 §2 |
| `discrete_state_input` | False | False | 保持一致；候选 3 的对照再打开 |
| 其余 | pi05=True, horizon 10, quantile 归一化, 不做 delta, 从 pi05_base 起 | 同 | |

LoRA 具体挂在哪（[`openpi/models/gemma.py:88-107, 177-242, 319-323`](../external/openpi/src/openpi/models/gemma.py#L88)，[`openpi/models/lora.py:12-24`](../external/openpi/src/openpi/models/lora.py#L12)）：

| 半边 | 变体 | 挂在哪些矩阵 | rank / alpha |
|---|---|---|---|
| VLM 骨干（PaliGemma 的 Gemma 2B） | `gemma_2b_lora` | 每层注意力的 q/k/v 投影和输出投影（`attn`）、每层 FFN 的三个矩阵（`ffn`） | 16 / 16 |
| 动作专家（Gemma 300M） | `gemma_300m_lora` | 同上 | 32 / 32 |

SigLIP 视觉编码器、embedding、`action_in_proj` / `action_out_proj` / 时间步 MLP 这些不在 LoRA 里，被 `freeze_filter` 冻住（它冻结一切名字不含 `lora` 的参数）。也就是说**微调能改的只有两个 transformer 里的注意力和 FFN 的低秩增量**；视觉特征完全是 pi05_base 的。这对候选 3（起手姿势脆弱）有含义：如果问题在视觉编码器没见过新姿势的手臂，LoRA 微调也救不了。

## 2. 示教子采样怎么做

数据集是 `physical-intelligence/libero`（LeRobot v2.0，1693 集、273k 帧、10 fps、图 256×256，35 GB，放 `$HF_LEROBOT_HOME=/weka/.../datasets/lerobot`）。不复制数据：`LiberoSubsetDataConfig` 读 `meta/episodes.jsonl`，按任务分层抽 n 条，把 episode 索引放进 DataConfig，monkeypatch 过的 `create_torch_dataset` 传给 `LeRobotDataset(episodes=...)`。抽样是确定的（任务内按 index 排序后 `random.Random(seed).sample`），用到的索引写在 `assets/<config>/<asset_id>/episodes.json`。

libero_10 每任务示教数不等：**task 8（两只壶）只有 29 条，是 10 个任务里最少的，而且平均 407 帧是最长的**；其他任务 33–49 条、186–294 帧。这本身是 task 8 成功率 66% 的一个数据侧解释候选，和 notes/03 §2.3 "失败是采样随机"不矛盾：示教少 → 策略方差大。

## 3. norm stats

`compute_norm_stats_ext.py --config-name <配置>` 在子集上算分位数（`use_quantile_norm=True`，因为 model_type 是 PI05），存 `assets/<config>/<asset_id>/norm_stats.json`，进 git。训练时 [`train.sbatch`](../scripts/train.sbatch) 发现没有就先算。换数据源时最容易错的一环：stats 必须和训练数据一致，评测时从 checkpoint 的 assets 里读。

## 4. 怎么跑

```bash
CONFIG=pi05_libero10_lora_n10 EXP=smoke SMOKE=1 sbatch scripts/train.sbatch   # 50 步冒烟
CONFIG=pi05_libero10_lora_n50 EXP=A sbatch scripts/train.sbatch
# 评测某个 checkpoint（server 端加 --config/--dir）：
CONFIG=pi05_libero10_lora_n50 CKPT=/weka/.../checkpoints/learnVLA/pi05_libero10_lora_n50/A/10000 SUITE=libero_10 TAG=A_10k OUT=results/finetune/A_10k bash scripts/run_eval.sh
```

每 10k 步存一个 checkpoint，三个都评：10 任务 × 50 集 + 10 个固定初始状态 × 5 次 + 失败分类（计划 §2）。

## 5. 结果（待填）

## 6. 换数据源 / 真机时这里会踩什么（待填）
