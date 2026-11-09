# 从上往下看 VLA：新手地图

> 读者：第一次接触 VLA 的人。目的：在读任何代码之前，先知道整件事由哪几块组成、每块回答什么问题、这个仓库里每块对应哪个文件。其他 notes 都是这张地图上某一格的放大。

## 0. 一句话

VLA（Vision-Language-Action）模型是一个函数：**输入几张相机图 + 一句自然语言指令，输出机器人接下来要执行的一小段动作**。它用预训练的视觉语言模型（VLM）当"脑子"，再接一个输出动作的"手"。

我们这两周做的事是：拿一个别人训好的 VLA（π0.5），在一个标准仿真考场（LIBERO）上让它干活，看它什么时候失败、为什么失败。

## 1. 问题怎么定义

把"机器人做一件事"拆成这几个词，后面所有文档都在用：

| 词 | 意思 | 在 LIBERO 里的具体样子 |
|---|---|---|
| **observation（观测）** | 每个时刻机器人能看到/感到的东西 | 第三人称图 + 腕上相机图（各 224×224）+ 8 维本体状态 + 指令文本 |
| **state（状态）** | 观测里的"本体感知"部分：机器人自己报的数字，不是看出来的 | 末端位置 3 + 姿态 3 + 夹爪 2 = 8 维 |
| **action（动作）** | 每个时刻给机器人的指令 | 7 维：末端位移 3 + 转动 3 + 夹爪开合 1，20 Hz |
| **action chunk / horizon** | 模型一次吐出未来 H 步动作，而不是 1 步 | H = 10（0.5 秒） |
| **replan** | 执行 chunk 里前 k 步后再问一次模型 | k = 5 |
| **policy（策略）** | 观测 → 动作 的那个函数，即模型 + 前后处理 | π0.5 + 归一化 + 图像处理 |
| **episode（一集）** | 从一个初始状态开始到成功或超时 | 最多 220–520 步 |
| **task（任务）** | 一句指令 + 一个场景 | "把黑碗放到盘子上" |
| **suite** | 一组任务 | 4 个 suite × 10 任务 |
| **success rate** | 成功的 episode 比例 | 每任务 50 个初始状态 |

一张图：

```
          指令 "put the bowl on the plate"
                     │
  相机图 ×2 ──────►  策略 π  ──► 未来 10 步动作 ──► 执行前 5 步 ──► 环境变化
  state(8) ─(本配置不用)┘                                     │
       ▲                                                      │
       └──────────────── 新观测 ◄─────────────────────────────┘
                   循环，直到 BDDL 目标谓词满足（成功）或超时（失败）
```

## 2. 整个系统的五层

从上到下，每层回答一个问题。括号里是这个仓库对应的位置。

```
┌─ 5 评测与分析   这个策略好不好？失败在哪？        scripts/summarize.py, contact_sheet.py, notes/02, 06
├─ 4 环境/仿真    世界怎么变、怎么判成功？          LIBERO (MuJoCo), external/openpi/third_party/libero
├─ 3 推理部署     怎么把模型跑起来接到环境上？      serve_policy_ext.py ⇄ eval_libero.py (websocket)
├─ 2 模型         观测→动作 的网络长什么样？        external/openpi/src/openpi/models/pi0.py, notes/01
└─ 1 数据与训练   网络是怎么学会的？                LIBERO 示教数据 + openpi 训练配置, 第 2 周
```

**第 1 周只碰 3、4、5 层**，模型和数据都用现成的。因为训练是最贵、最慢、最难判断对错的一层；先把"怎么量、量出来是什么意思"搞清楚，再去动模型，才知道改动有没有用。

## 3. π0.5 一页纸

模型分两半：

- **VLM 骨干（PaliGemma，约 3B）**：看图 + 读字。图像经 SigLIP 变成 token，文字经 Gemma 的 tokenizer 变成 token，一起过 Gemma 语言模型。这一半继承了互联网规模的预训练，是"知道什么是碗、什么叫左边"的来源。
- **动作专家（action expert，约 300M）**：一个小 transformer，和 VLM 共享注意力，但有自己的权重。它不是一步算出动作，而是用 **flow matching**：从随机噪声开始，分 10 步把噪声"流"成一段合理的动作。所以同一个观测问两次，答案略有不同（这就是第 1 周"随机性实验"要量的东西）。

输入输出的张量：

| | 形状 | 说明 |
|---|---|---|
| 图 | 2 × (224, 224, 3) | 第三人称 + 腕部；LIBERO 的图要翻转 180° |
| 指令 | ≤ 200 token | 文字 |
| state | (8,) | **`pi05_libero` 配置下不进网络**（见 notes/01 §3） |
| 输出动作 | (10, 7) | 10 步 × 7 维，先归一化到 [-1, 1] 再反归一化 |

"预训练 → 微调"的关系：`pi05_base` 是 π 公司在大量真机数据上训的通用模型；`pi05_libero` 是在 LIBERO 的 2000 条示教上全量微调 3 万步得到的。我们评测的是后者。第 2 周自己微调时，起点也是 `pi05_base`。

## 4. 评测到底在量什么

- 一个数字 "98.2%" = 500 集里 491 集在限定步数内触发了成功判定。
- 判定是 LIBERO 用 BDDL 写的谓词（"碗 On 盘子"），物理上够不够"放上去"由谓词说了算，不由人眼。
- 500 集的随机误差约 ±1 个百分点（Wilson 区间），所以两个模型差 1 个点不算差。任务级 50 集误差 ±8 个点，更不能比。
- 失败在 CSV 里只有一个 0，原因要看视频。所以我们把失败视频抽成 6 帧拼图并分成 7 类（notes/06）。

## 5. 现在手里有什么

| 产物 | 位置 | 回答的问题 |
|---|---|---|
| 官方复现报告 | [`results/repro.md`](../results/repro.md) | 我们的环境和论文一致吗 |
| 失败画廊 + 标签 | [`results/failure_gallery/official/`](../results/failure_gallery/official/) | 失败长什么样 |
| 成功/失败对照视频 | [`results/showcase/`](../results/showcase/) | 同一个任务成功和失败差在哪 |
| 推理消融 | `results/inference_ablation.csv/.png`, notes/03 | replan、去噪步数影响多大：开环=闭环，replan 越频繁越差，去噪 3 步够 |
| 随机性实验 | [`results/stochasticity_by_init.csv`](../results/stochasticity_by_init.csv), [`stochasticity_hist.png`](../results/stochasticity_hist.png), notes/03 §2.3 | 失败是"这个初始状态必败"还是"运气"：200 个初始状态 0 个必败 |
| LIBERO-Plus | [`results/libero_plus_by_category.csv`](../results/libero_plus_by_category.csv), [`libero_plus.png`](../results/libero_plus.png), notes/04 | 换光照/视角/措辞还行不行：视角和起手姿势最伤（等级 5 约 45%），光照纹理免疫 |

## 6. 读文档的顺序

1. 这篇（地图）。
2. [`notes/02_eval_protocol.md`](02_eval_protocol.md)：一个数字是怎么来的。只要你会看结果，就能跟上后面所有讨论。
3. [`results/repro.md`](../results/repro.md) + [`notes/06_failure_taxonomy.md`](06_failure_taxonomy.md)：结果和失败。配合视频看。
4. [`notes/03_inference_ablation.md`](03_inference_ablation.md) + [`notes/04_libero_plus.md`](04_libero_plus.md)：不训练的三组实验。每组开头有"区分哪两个假设"，先读那张表再看数字。
5. [`notes/01_code_walkthrough.md`](01_code_walkthrough.md)：顺着一次推理调用读代码。需要开着编辑器对照行号，是最"具体"的一篇，不必一次读完，§0 总览和 §4 模型先看。
6. [`notes/00_setup.md`](00_setup.md)：环境怎么装的、踩了什么坑。要自己改脚本时再读。
7. [`vla-onboarding-plan.md`](../vla-onboarding-plan.md)：两周计划本身，看第 2 周要做什么。

## 7. 术语表

- **VLM**：视觉语言模型，输入图+文字，输出文字。VLA 把它改成输出动作。
- **PaliGemma**：Google 的 VLM，SigLIP 视觉编码器 + Gemma 语言模型。π0 系列的骨干。
- **flow matching**：一种生成模型，学"从噪声到数据的速度场"，采样时沿速度场走几步。和扩散模型是近亲，步数更少。
- **action expert**：专门输出动作的那一半网络；参数和 VLM 分开，但注意力互通。
- **proprioception / state**：本体感知，机器人自己测得的关节/末端/夹爪数字。
- **BDDL**：LIBERO 用来描述任务目标的小语言，写的是谓词（On、In、Open）。
- **init state**：一集开始时所有物体的精确位置，LIBERO 每任务预录 50 个，所有论文共用。
- **OSC_POSE**：robosuite 的控制器，把"末端往哪挪"的 7 维动作换算成关节力矩。
- **quantile normalization**：用训练集的 1% 和 99% 分位把每一维缩放到 [-1, 1]，让模型不用关心单位。
- **LoRA**：微调时只训练插入的低秩小矩阵，省显存；第 2 周要用。
- **Wilson 区间**：比例的置信区间，样本小、比例接近 100% 时比"±1.96√(p(1-p)/n)"更靠谱。
