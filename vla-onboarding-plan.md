# VLA Manipulation 入门计划 v2：用 π0.5 + LIBERO 建立失败直觉、找下一个 idea

> 日期：2026-11-07（v2，修订自 11-06 版，旧版存为 `vla-onboarding-plan.v1-20261106.md`）。
> 执行者：cluster 上的 agent。读者：lwang（入门）。时间预算 2 周。
>
> v2 相对 v1 的改动：
> 1. 砍掉观测侧消融与标签噪声（原 §2.3 / §2.4）。它们回答的是 Active Observer 的问题，与"不复活 Active Observer"冲突。
> 2. 微调次数从 25+ 压到 ≤ 4。单卡 H100 一次 30k 步微调约一天，两周只够这么多。
> 3. 第 1 周全部改成不训练的实验：推理期消融、LIBERO-Plus 扰动、随机性分离。信息量最大、算力最便宜。
> 4. 方差分解改成"固定初始状态、换采样 seed 重复"的设计。原设计在确定性仿真里残差恒为零，测不出东西。
> 5. 加入代码阅读与教学文档要求（§7）、集群规范（§8）。

---

## 0. 目的

**第一目的**：建立对当前 VLA 微调流程"在哪里、为什么坏"的第一手直觉，产出一份排序过的瓶颈备忘录 `bottleneck_memo.md`，作为下一轮找 idea 的输入。

**第二目的**：lwang 入门。每一步都要产出讲解文档（§7），让人能顺着文档和代码走一遍。

不是目的的事：
- 不是复现一个 98% 的 LIBERO 数字。
- 不是改进微调算法。
- 不是在仿真里长期迭代。

成功标准：第 2 周结束时，备忘录里有 **3 条**瓶颈，每条附带（a）复现它的最小实验、（b）证据数字与置信区间、（c）"现有 pipeline 为什么在结构上解决不了它"、（d）相邻文献里谁试过、没解决的是什么。3 条有足够样本量，比 5 条靠猜强。

---

## 0.5 仿真器：MuJoCo，不是 Isaac

- LIBERO 建在 robosuite 上，robosuite 的物理引擎是 MuJoCo。openpi 的 LIBERO 评测直接起 MuJoCo 环境。本计划全程不碰 Isaac。
- 原因：VLA 社区的主流仿真 benchmark（LIBERO、SimplerEnv、RoboCasa）都是 MuJoCo 系。Isaac Lab 的优势在大规模并行 RL 和 locomotion，换过去意味着重做数据与评测协议，与本计划目的无关。
- 之前 habitat / isaac 的经验在这里不直接迁移。需要新学的是 robosuite 的环境定义、相机配置和控制器（OSC_POSE，末端位姿增量控制）。
- 集群无头渲染：`MUJOCO_GL=egl`。

---

## 1. 第 1 周：跑通 + 不训练的实验

### 1.1 Day 1–2：跑通官方评测

- 模型：**π0.5**，openpi 的 `pi05_libero` 官方 checkpoint。先不微调。
- 环境：openpi 要 Python 3.11，LIBERO 要 Python 3.8。openpi 用 client/server 模式解决：server 进程跑策略，client 进程跑 LIBERO 仿真，两者 websocket 通信。集群没有 docker，所以建两个 conda 环境（`openpi`、`libero`），在同一个 sbatch 作业里先后台起 server，再起 client。这是 Day 1 最可能卡住的地方，比显存更要紧。
- 评测：4 个 suite（Spatial / Object / Goal / 10）× 10 任务 × 50 episode，固定 seed。记录每个 episode 的成败、步数、初始状态 id。
- 统计提醒：50 个 episode 在 p≈0.9 附近标准误约 ±4 个百分点，单任务数字没有意义；suite 级 500 个 episode 约 ±1.3。**比较只在 suite 级做，差距超过 2 倍标准误才算差距。**
- 记录：显存峰值、评测耗时、每 episode 耗时。写入 `results/repro.md`。
- 守门：不许超过 2 天。跑不通就记录卡点，转 OpenVLA-OFT。

### 1.2 Day 2：代码阅读（人学习的部分，与 1.1 并行）

顺着 openpi 从观测到动作读一遍，写 `notes/01_code_walkthrough.md`。具体文件路径以仓库当前版本为准，下面是要找的东西：

1. **配置**：`pi05_libero` 这个 TrainConfig 里定义了什么。模型类、数据集、action horizon（chunk 长度）、batch、LoRA 设置、norm stats 路径。
2. **数据变换**：LIBERO 的 policy transform。图像怎么 resize、state 怎么 pad 到统一维度、prompt 怎么拼。动作是 7 维末端增量 + 夹爪，归一化到 [-1, 1]。
3. **归一化**：norm stats 怎么算（均值方差还是分位数）、训练时在哪里用、推理时在哪里反归一化。换数据源时最容易错的一环。
4. **模型**：PaliGemma 作为 VLM 前缀，action expert 用 flow matching 从噪声去噪出一个 action chunk。看清楚：去噪步数、时间步嵌入、prefix 和 suffix 的 attention mask。
5. **推理**：`serve_policy.py` 怎么封装，LIBERO 的 client 脚本里 `replan_steps` 是什么。策略每次预测 50 步 chunk，但只执行前 `replan_steps` 步就重新观测。这个参数是 1.3 的实验对象。

笔记格式：每一项"概念一段 → 关键代码摘录（文件:行号，10–30 行）→ 解释 → 换数据源时会在这里踩什么坑"。

### 1.3 Day 3：推理期消融（不训练）

同一个官方 checkpoint，只改推理参数，在 LIBERO-10（失败最多的 suite）上评测。

| 实验 | 取值 | 区分的两个假设 |
|---|---|---|
| `replan_steps` | 1 / 5（默认）/ 10 / 25 / 50（整段开环） | 失败来自"预测错"还是"执行中不重新观测、不纠偏" |
| 去噪步数 | 3 / 10（默认）/ 20 | 动作质量是否受采样精度限制 |

开环（50）成功率若与闭环接近，说明策略基本不依赖反馈，失败在预测侧；若大跌，说明闭环纠偏在承担很多，恢复能力是关键。

### 1.4 Day 3–4：LIBERO-Plus 扰动（不训练）

- 在 LIBERO-Plus 的 7 类扰动（物体布局、相机视角、机器人初始状态、语言、光照、背景纹理、传感器噪声）上评官方 checkpoint。
- 先每类扰动每任务 10 episode 粗扫，找出成功率大跌的扰动类，再对这些类补到 50 episode。
- **不重做分析**：LIBERO-Plus 论文已经报告视角和机器人初始状态扰动把成功率从 95% 打到 30% 以下。这一节的目的是**收集失败样本**供 §3 分类，并亲眼看到"扰动下策略怎么坏"。
- LIBERO-X 可选，有时间再跑。

### 1.5 Day 4：随机性分离（替代原方差分解）

- 在 LIBERO-10 上选 20 个初始状态（含成功和失败过的），每个初始状态固定不变，换 K=10 个采样 seed 重复。
- 看每个初始状态的成功率分布：
  - 集中在 0 或 1（双峰）→ 失败由初始状态决定，是可预测的，数据侧可解。
  - 散在 0 到 1 之间 → 失败来自策略采样随机性，是模型侧问题。
- 这一个数字直接决定下一个 idea 落在数据侧还是模型侧。它比原来的线性方差分解干净，因为仿真是确定性的，残差只能来自策略采样。

### 1.6 Day 5：失败分类 + 周报

- 把 1.1 / 1.3 / 1.4 / 1.5 的所有失败 episode 按 §3 标注，每类存 3–5 个代表视频到 `results/failure_gallery/`。
- 写 `progress_2026-11-xx.md`，并决定第 2 周用哪个 suite 微调（默认 LIBERO-10；若 Goal 的失败更有意思则换 Goal）。

---

## 2. 第 2 周：最多 4 次微调

只在一个 suite 上做。每次微调 LoRA、30k 步、单卡 H100 或 H200。每 10k 步存 checkpoint，三个 checkpoint 都评测，这是免费的训练曲线。

| 槽位 | 示教数 / 任务 | 目的 |
|---|---|---|
| A | 50（官方全量） | 复现。与官方 checkpoint 在 suite 级差距应在 2 倍标准误内，否则查归一化或 chunk 设置 |
| B | 10 | 数据效率曲线第一点 |
| C | 2 | 数据效率曲线第二点，看哪类失败先出现 |
| D | 10，第二个 seed | 训练方差。没有这个数，B 和 C 的差距不知道是不是噪声 |

每个 checkpoint 的评测：10 任务 × 50 episode + 对 10 个固定初始状态做 K=5 的随机性重复 + 失败分类。

要看的问题：
- 示教减少时成功率怎么掉，失败类分布怎么变（抓不到 / 放错 / 顺序错）。
- 10k / 20k / 30k 步的曲线：少数据时是否过拟合，在原 LIBERO 上更好但在 LIBERO-Plus 上更差。对 B、C 各跑一次 LIBERO-Plus 中最致命的那一类扰动。
- 随机性分离的结论是否随示教数变化：少数据时失败是变得更"可预测"还是更"随机"。

算力预算：4 次训练约 4 个 GPU 日，评测约 1 个 GPU 日。集群排队另算，所以第 2 周周一就把 A 和 B 提交上去。

---

## 3. 失败模式分类（看视频时用）

每条失败 episode 标一个主类、可选一个次类：

| 类 | 定义 | 典型信号 |
|---|---|---|
| G-reach | 没到达物体附近 | 轨迹偏、停住、振荡 |
| G-grasp | 到了但没抓住 / 抓滑 | 夹爪闭合时机或位置错 |
| T-transport | 抓住后掉落 | 中途开夹爪、碰撞 |
| P-place | 放置位置/姿态错 | 放偏、没放进去 |
| L-language | 操作了错误物体或错误目标 | 多物体场景下选错 |
| S-sequence | 长时程任务阶段顺序错或卡在某阶段 | LIBERO-10 |
| R-norecover | 失败后无任何重试动作，停住或重复同一动作 | 可与上面任一类并标 |
| X-other | 其他 | 记录描述 |

R-norecover 是新加的。示教里没有恢复数据是 VLA 最常被指出的结构性问题，要单独统计"失败后有没有重试"。

**同一个成功率下降，失败类分布不同，含义完全不同。**

---

## 4. 分析工具

- 所有成功率附 Wilson 95% 置信区间，不报裸数字。
- 失败类 × 实验条件的交叉表，报占比。
- 随机性分离：每初始状态成功率的直方图，报双峰程度（在 0 或 1 的初始状态占比）。
- 可选：对 (任务, 扰动, 示教数) 做 logistic 回归看主效应，但样本量小时别过度解读。

---

## 5. 第 2 周周五：写瓶颈备忘录

`bottleneck_memo.md`，每条瓶颈按以下模板：

```
## 瓶颈 N：<一句话>
- 证据：<实验 + 数字 + 置信区间 + 失败类占比>
- 复现：<最小实验命令>
- 为什么现有 pipeline 在结构上解决不了：<分析>
- 谁试过：<相邻文献，他们解决到什么程度，剩下什么>
- 上限问题：如果完美解决，收益多大？有没有人用更便宜的办法达到同样效果？
- 社区会在意吗：<一句话判断>
- 一人可执行吗：<硬件/算力/数据需求>
```

按"结构性 × 社区在意 × 一人可做"排序。预期 3 条。

---

## 6. 边界（不要做的事）

1. 不改微调算法、不加新 loss、不换架构。
2. 不在仿真里超过 2 周。
3. 不为了提高某个数字调超参。
4. 不做观测侧消融、不做标签噪声注入。这两个是 Active Observer 的问题，Active Observer 的结论已经清楚。
5. 不读文献找空格；文献只用来回答"谁试过"。
6. 微调不超过 4 次。想加第 5 次，先写出它区分的两个假设，并说明砍掉哪一次来换。

---

## 7. 教学文档要求

目录 `notes/`，每个阶段一份，格式统一为：**概念 → 关键代码（文件:行号 + 10–30 行摘录 + 逐行解释）→ 实验怎么跑（完整命令）→ 结果怎么读 → 换数据源 / 换真机时这里会踩什么坑**。

| 文件 | 内容 |
|---|---|
| `notes/00_setup.md` | 两个 conda 环境怎么建、client/server 怎么起、EGL 渲染、第一次跑通的完整命令与常见报错 |
| `notes/01_code_walkthrough.md` | §1.2 的五项：配置、数据变换、归一化、模型、推理 |
| `notes/02_eval_protocol.md` | LIBERO 的初始状态怎么定义、seed 怎么固定、成功判定在哪里、置信区间怎么算 |
| `notes/03_inference_ablation.md` | replan_steps 和去噪步数在代码里改哪、结果怎么解读 |
| `notes/04_libero_plus.md` | 扰动怎么注入（改了 robosuite 的什么）、怎么挑扰动类 |
| `notes/05_finetune.md` | LoRA 挂在哪些层、norm stats 怎么重算、示教子采样怎么做、checkpoint 结构 |
| `notes/06_failure_taxonomy.md` | 分类标准、每类代表视频、容易混淆的边界案例 |

**重点代码清单**（lwang 必读，agent 在 notes 里逐段讲解）：

1. `pi05_libero` 的 TrainConfig 定义。
2. LIBERO 的输入/输出 transform（图像、state、prompt、动作反归一化）。
3. Normalize / Unnormalize 变换与 norm stats 的计算脚本。
4. π0.5 模型的 forward：prefix（图像 + 文本 token）、suffix（state + 噪声动作 + 时间步）、attention mask、flow matching loss 与采样循环。
5. LoRA 配置：挂在哪些矩阵、rank 多少。
6. LIBERO client 主循环：观测取出、预处理、调策略、执行 `replan_steps` 步、成功判定。
7. robosuite 的 OSC_POSE 控制器：7 维动作怎么变成关节力矩。

---

## 8. 集群规范（沿用 SmartXray / wifiVLN 的做法）

- sbatch 头：`--partition=h100,h200,a100 --account=rzhao36_vifinav --qos=jhu --comment=accept_cost --gres=gpu:1 --cpus-per-task=8 --mem=64G --requeue`。评测作业 CPU 可以多给，MuJoCo 渲染吃 CPU。
- 分区固定用 `h100,h200,a100`（qos=jhu）。l40s 留给 wifiVLN 项目，不要用。冒烟测试实测评测占 34.5 GB 显存，三种卡都够。`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。
- conda：`source /weka/projects/rzhao36/lwang/miniforge3/etc/profile.d/conda.sh`，先 `module load cuda/12.6.3`。建环境本身也用 sbatch（参考 `SmartXray/scripts/setup_env.sbatch`）。
- **weka guard**：计算节点间歇性掉 `/weka` 挂载。每个 sbatch 开头抄 `SmartXray/scripts/run_episodes.sbatch` 的那段：等 30 次不行就 requeue 并排除该节点；日志先写 `$HOME` 再拷回。
- HF 缓存：`HF_HOME=/weka/projects/rzhao36/lwang/datasets/models/huggingface`，`HF_HUB_DISABLE_XET=1`。
- 存储：项目盘 39T 已用 80%。checkpoint 只留每次训练的 10k/20k/30k 三个，评测视频只留失败样本和 gallery。
- 重活一律 sbatch，不在会话节点跑。所有 sbatch 脚本、seed、配置进 git。

---

## 9. 预期产出清单

| 文件 | 内容 |
|---|---|
| `results/repro.md` | 官方 checkpoint 评测记录、关键配置、资源消耗 |
| `results/eval_official.csv` | 4 suite × 10 任务 × 50 episode 的逐 episode 结果 |
| `results/inference_ablation.csv` + 图 | replan_steps × 去噪步数 |
| `results/libero_plus.csv` + 图 | 7 类扰动下的成功率与失败类分布 |
| `results/stochasticity.csv` + 图 | 固定初始状态 × 采样 seed 的成功率直方图 |
| `results/data_efficiency.csv` + 图 | 示教数 × checkpoint 步数 × seed 的成功率与失败类分布 |
| `results/failure_gallery/` | 每类失败 3–5 个代表视频/帧 |
| `notes/00–06` | 教学文档（§7） |
| `bottleneck_memo.md` | 最终交付 |

---

## 10. 给 agent 的工作方式要求

- 每个实验先写一句"这个实验要区分的两个假设是什么"，再跑。区分不了任何假设的实验不跑。
- 先抽样小跑（1 任务 × 10 episode）确认 pipeline 正确，再全量。
- 每完成一个阶段，先写对应的 `notes/` 文档再进入下一阶段。文档是交付物，不是附带品。
- 结果表用 csv，报告用 markdown。每天结束写 `progress_<date>.md`。
- 发现与本计划预期相反的结果，不要圆，直接写出来；相反的结果比符合预期的更有价值。
- 所有成功率附置信区间。单任务级的比较不下结论。
