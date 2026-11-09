# notes/02 评测协议：LIBERO 的数字是怎么来的，以及怎么读

> 读者：lwang。目的：知道一个"96.85%"背后固定了什么、没固定什么，以及多大的差距才算差距。

## 1. LIBERO 的四个 suite

| suite | 任务数 | 考什么 | 官方 π0.5 | 最大步数 |
|---|---|---|---|---|
| libero_spatial | 10 | 同样的物体，不同空间布局；指令里有"左边的碗"这类空间关系 | 98.8 | 220 |
| libero_object | 10 | 同一个布局，不同物体；考物体识别 | 98.2 | 280 |
| libero_goal | 10 | 同一个场景，不同目标；考语言理解 | 98.0 | 300 |
| libero_10 (long) | 10 | 两阶段长时程任务 | 92.4 | 520 |

官方数字来自 [`external/openpi/examples/libero/README.md`](../external/openpi/examples/libero/README.md)。libero_90 是预训练用的 90 个任务，评测通常不报。

每个任务 50 条示教（LIBERO 原始数据），训练集就是 4 suite × 10 任务 × 50 条 = 2000 条轨迹。注意 **libero_90 不在 `physical-intelligence/libero` 这个训练集里**（它是 4 个评测 suite 的合集），所以官方 checkpoint 是"训练集和测试集同分布、只有初始状态不同"的设定。这是 LIBERO 被批评"考记忆"的根源（LIBERO-PRO、LIBERO-Plus 都在说这事）。

## 2. 一个 episode 固定了什么

读 [`scripts/eval_libero.py`](../scripts/eval_libero.py)：
1. `env.seed(7)`，然后 `env.reset()`。
2. `env.set_init_state(init_states[i])`：把 suite 自带的第 i 个 MuJoCo 状态向量写进去。每任务 50 个，是 benchmark 发布时录好的（[`third_party/libero/libero/libero/init_files/`](../external/openpi/third_party/libero/libero/libero/init_files/)），**所有论文的 50 episode 用的是同一批初始状态**。
3. 先执行 10 步空动作等物体落定。
4. 循环：观测 → 每 `replan_steps` 步问一次策略 → 执行。
5. `done`（BDDL goal 谓词全部满足）→ 成功；到 `max_steps` 没 done → 失败。

没固定的：策略的采样噪声（server 端 rng 每次调用都变）。所以同一初始状态跑两次，结果可能不同。这正是 §1.5 要量的东西。

MuJoCo 本身是确定性的：同样的状态 + 同样的动作序列 = 同样的结果。所以"随机性"只有策略这一个来源。

## 3. 多大的差距算差距

成功率是二项分布。n 个 episode、成功率 p 的标准误 ≈ sqrt(p(1-p)/n)：

| n | p=0.95 | p=0.90 | p=0.70 |
|---|---|---|---|
| 50（单任务） | ±3.1 | ±4.2 | ±6.5 |
| 500（单 suite） | ±1.0 | ±1.3 | ±2.0 |
| 2000（全部） | ±0.5 | ±0.7 | ±1.0 |

规则：
- **单任务的数字不比较**。50 个 episode 在 p=0.9 处 ±4 个点，两个任务差 5 个点是噪声。
- suite 级用 Wilson 区间（[`scripts/summarize.py`](../scripts/summarize.py)），两个区间不重叠才说"有差距"。
- 复现官方 checkpoint 时，suite 级落在官方数字 ±2 个点内就算复现成功。更大的差距先查：图像翻转、resize 方式、norm stats、replan_steps、`num_steps_wait`。

Wilson 区间比正态近似好在 p 接近 1 时不会超过 100%，且 n 小时更准。公式在 `summarize.py:wilson`。

## 4. 失败和"超时"

LIBERO 里没有"部分完成"。两阶段任务做完第一阶段卡住，和一开始就没动，都是 0。所以成功率掩盖了失败的位置，这就是为什么 §3 要看视频分类。`episodes.csv` 的 `steps` 列能粗分：steps 接近 max_steps 是超时（卡住、反复尝试），steps 很短而失败是出了异常（撞翻、掉落触发终止）。

## 5. 评测成本（冒烟测试 1678271，H100 实测）

- server 起到 healthz 通：40 秒。第一次 `infer` 触发 JIT 编译，第一个 episode 平均 820 ms/次、共 23 秒；之后 **42 ms/次**。
- 一个 spatial episode（约 100 步、20 次推理）：**3.3 秒**，仿真和渲染占大头。
- 估算：spatial 500 episode 约 30 分钟；libero_10 最长 520 步，约 1.5 小时。整个 4 suite 一张卡 4 小时以内。
- 显存：JAX 预留 75%，实际远小于此。

## 6. 怎么跑

```bash
sbatch --array=0-3 scripts/eval_suite.sbatch                        # 官方 checkpoint，4 suite
python3 scripts/summarize.py results/eval_official/*/episodes.csv   # suite 级汇总
python3 scripts/summarize.py --by task results/eval_official/libero_10/episodes.csv
```
