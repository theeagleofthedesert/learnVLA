# 官方 π0.5 checkpoint 在 LIBERO 上的复现（2026-11-07）

checkpoint `gs://openpi-assets/checkpoints/pi05_libero`，openpi main（clone 于 2026-11-06），
每 suite 10 任务 × 50 个官方初始状态 = 500 集，`replan_steps=5`，`num_steps=10`，seed 7。
作业 1678298[0-3]，节点 gh102/gh123/gh132（H100），每 suite 一张卡。

## 1. suite 级结果

| suite | 本次 succ/n | 本次 % | Wilson 95% CI | 官方 % | 官方在 CI 内 | 耗时 |
|---|---|---|---|---|---|---|
| libero_spatial | 491/500 | 98.2 | [96.6, 99.1] | 98.8 | 是 | 54 min |
| libero_object  | 494/500 | 98.8 | [97.4, 99.4] | 98.2 | 是 | 36 min |
| libero_goal    | 488/500 | 97.6 | [95.9, 98.6] | 98.0 | 是 | 33 min |
| libero_10      | 468/500 | 93.6 | [91.1, 95.4] | 92.4 | 是 | 65 min |
| 平均 | | 97.05 | | 96.85 | | 3.1 h 卡时 |

官方数字来自 [`external/openpi/examples/libero/README.md`](../external/openpi/examples/libero/README.md)。四个 suite 的官方值都落在本次的置信区间内，差异都 ≤ 1.2 个百分点，和 500 集的采样噪声同量级（见 [`notes/02_eval_protocol.md`](../notes/02_eval_protocol.md) §3）。**复现成立。**

注意 README 没说官方跑了多少集、用了什么 replan；这里按 openpi [`examples/libero/main.py`](../external/openpi/examples/libero/main.py) 的默认值跑。

## 2. 任务级：失分在哪

只列不是 50/50 的任务（完整表 `python3 scripts/summarize.py --by task results/eval_official/*/episodes.csv`）：

| suite / task | succ/50 | % | CI |
|---|---|---|---|
| libero_10 / 8 put both moka pots on the stove | 33 | 66.0 | [52.2, 77.6] |
| libero_10 / 4 white mug on left plate, yellow mug on right plate | 46 | 92.0 | [81.2, 96.8] |
| libero_10 / 0 alphabet soup + tomato sauce in basket | 47 | 94.0 | [83.8, 97.9] |
| libero_goal / 6 cream cheese in the bowl | 47 | 94.0 | [83.8, 97.9] |
| 其余 36 个任务 | 48–50 | 96–100 | |

libero_10 的 32 条失败里 17 条来自 task 8。去掉它，libero_10 是 435/450 = 96.7%。
任务级 CI 很宽（50 集 ±8 个点），**除了 task 8，其他任务之间的差异都不显著**。

## 3. 失败形态

59 条失败全部出了 6 帧拼图并逐条标注（[`results/failure_gallery/official/`](failure_gallery/official/)，分类标准见 [`notes/06_failure_taxonomy.md`](../notes/06_failure_taxonomy.md)）：

| 类 | 条数 | 说明 |
|---|---|---|
| D 悬停不放 | 35 | 物体抓到、到了目标上方、不松手直到超时；libero_10 两阶段任务的第二阶段几乎全是这个 |
| A 接近停滞 | 8 | 手悬在目标附近，没合夹爪 |
| C 放置偏离 | 8 | 松手了但掉在篮子/盘子外；集中在 libero_object |
| E 物体/目标混淆 | 3 | 抓了相邻任务的物体 |
| F 看似完成 | 2 | 画面上目标达成但 `done` 未触发，待查 BDDL 谓词 |
| G 碰撞 | 2 | 撞倒别的物体 |
| B 抓取失败 | 1 | |

所有失败的 `steps` 都等于 max_steps——LIBERO 只在成功或超时时结束。

## 4. 成本

| 项 | 值 |
|---|---|
| server 启动到 healthz | ~40 s |
| 首次推理（JIT） | ~0.8 s |
| 之后每次推理 | 39–43 ms（H100） |
| 每集推理次数 | 22（spatial）~55（libero_10） |
| 每集 wall | 3–8 s，被 MuJoCo 渲染 + 视频编码主导，不是模型 |
| 显存 | 34.5 GB（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` 预分配；真实占用更小） |

4 suite 全量 = 3.1 卡时。以后每次微调后的评测按这个预算算。

## 5. 复跑命令

```bash
sbatch --array=0-3 scripts/eval_suite.sbatch          # 结果 results/eval_official/<suite>/
python3 scripts/summarize.py results/eval_official/*/episodes.csv
```
