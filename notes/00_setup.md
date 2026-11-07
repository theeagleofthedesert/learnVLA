# notes/00 环境搭建与第一次跑通

> 读者：lwang。目的：看完能自己在集群上起一次 π0.5 的 LIBERO 评测，并知道每个环节为什么这样做。
> 状态：首次搭建作业 `sbatch scripts/setup_envs.sbatch`（job 1677024）。"常见报错"一节在它跑完后补。

## 1. 概念：为什么是两个 Python 环境 + 一个 websocket

openpi 要 Python 3.11（JAX 0.5、flax 0.10），LIBERO 要 Python 3.8（它依赖旧版 robosuite 1.4 和 mujoco 2.x，升不上去）。两边装不进一个解释器，所以 openpi 把策略做成一个**服务**：

```
[libero venv, py3.8]                         [openpi venv, py3.11]
eval_libero.py                                serve_policy_ext.py
  MuJoCo 仿真 → 观测(图像+state+prompt) ──websocket──▶ Policy.infer()
  执行动作 ◀─────────────────────── action chunk (10, 7) ──┘
```

通信用 msgpack 序列化 numpy，见 `external/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py:45-55`。真机上也是同一套：机器人进程是 client，GPU 机器是 server。所以这个结构不是仿真专用的 workaround。

集群上没有 docker，官方的 `docker compose` 方案用不了。两个环境在同一个 sbatch 作业里先后起：server 后台起，等端口打开，再起 client。这一段在 `scripts/run_eval.sh`。

## 2. 目录

```
learnVLA/
  external/openpi/                 上游仓库（depth 1 clone + submodule）
    .venv/                         openpi 环境（uv sync 生成）
    examples/libero/.venv/         LIBERO 环境（uv venv --python 3.8）
    third_party/libero/            LIBERO 源码（submodule），含 bddl / init_files / assets(405MB)
  libero_config/config.yaml        LIBERO 路径配置，见 §4
  scripts/
    env.sh                         所有路径和环境变量，每个脚本都 source 它
    weka_guard.sh                  计算节点掉 /weka 挂载时自动 requeue
    setup_envs.sbatch              建环境 + 下载 checkpoint + 冒烟测试
    run_eval.sh                    起 server + client 跑一次评测
    serve_policy_ext.py            server（加了 --num-steps）
    eval_libero.py                 client（逐 episode 写 CSV）
  results/                         CSV、视频、日志
  notes/                           本系列文档
```

## 3. 环境变量（`scripts/env.sh`）

| 变量 | 值 | 为什么 |
|---|---|---|
| `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` | weka 下 | uv 会下载托管 Python 3.8/3.11 和几 GB 的 wheel，别占 `$HOME` |
| `OPENPI_DATA_HOME` | `/weka/.../datasets/models/openpi` | openpi 把 `gs://openpi-assets/...` 的 checkpoint 和 tokenizer 缓存到这里，默认是 `~/.cache/openpi`。代码在 `src/openpi/shared/download.py:18-28` |
| `LIBERO_CONFIG_PATH` | `learnVLA/libero_config` | 见 §4 |
| `PYTHONPATH` | 加 `third_party/libero` | 官方 README 要求，否则 `from libero.libero import benchmark` 找不到 |
| `MUJOCO_GL=egl` | | 无头渲染。计算节点没有 X，glx 不可用；osmesa 是纯 CPU，慢 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 评测 0.75，训练 0.9 | JAX 启动时一次性预留这个比例的显存 |

## 4. LIBERO 的一个坑：首次 import 会交互式提问

`third_party/libero/libero/libero/__init__.py:62-95`：如果 `~/.libero/config.yaml` 不存在，它会 `input("Do you want to specify a custom path ...")`。sbatch 里没有 stdin，会直接卡死到超时。解决：预先写好 `libero_config/config.yaml`，用 `LIBERO_CONFIG_PATH` 指过去。文件内容是五个路径（bddl、init_states、assets、benchmark_root、datasets），评测只用前三个。

## 5. 安装步骤（`setup_envs.sbatch` 做的事）

```bash
# openpi
cd external/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync            # 读 uv.lock，装进 .venv；LFS 跳过是因为 lerobot 仓库里有大文件
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# libero
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
   --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client   # client 只依赖这个小包，不依赖 openpi 本体
uv pip install -e third_party/libero
```

`uv run <cmd>` 会自动用 `.venv`，不需要 activate。LIBERO 那边用传统 activate。

## 6. 第一次评测

```bash
# 冒烟：1 任务 x 2 episode，存全部视频
SUITE=libero_spatial TASKS=0 TRIALS=2 VIDEOS=all OUT=results/smoke bash scripts/run_eval.sh
# 正式：一个 suite 全量
SUITE=libero_10 OUT=results/eval_official/libero_10 TAG=official bash scripts/run_eval.sh
```

`run_eval.sh` 做的事：随机选一个端口 → 后台起 server → 每 10 秒探一次端口，最多 20 分钟（模型加载 + JIT 编译要几分钟）→ 起 client → 结束时 kill server。server 日志在 `$OUT/server_<jobid>.log`，client 日志在 `$OUT/client_<jobid>.log`，逐 episode 结果在 `$OUT/episodes.csv`。

## 7. episodes.csv 字段

| 列 | 含义 |
|---|---|
| task_id, task | LIBERO suite 内的任务序号和语言指令 |
| init_state_id | 用了 suite 自带的第几个固定初始状态（每任务 50 个，`task_suite.get_task_init_states()`） |
| repeat | 同一初始状态第几次重复（§1.5 随机性实验用） |
| replan_steps | 每个 chunk 执行几步就重新观测 |
| success | 1/0。LIBERO 的 `done` 就是成功判定，超过最大步数没 done 就是失败 |
| steps | 实际执行步数（不含开头等物体落定的 10 步） |
| n_infer, mean_infer_ms | 调了几次策略、每次平均耗时（server 端计时，不含网络） |
| video | 失败视频文件名；左半主视角、右半腕部相机 |

## 8. 换数据源 / 真机时这里会踩什么坑

- **图像翻转**：LIBERO 训练数据是 180° 旋转存的，所以 client 里 `obs["agentview_image"][::-1, ::-1]`（`eval_libero.py`）。换数据源要先确认存的方向。
- **state 的定义**：8 维 = 末端位置 3 + 末端轴角 3 + 夹爪两指 2。四元数转轴角的函数是从 robosuite 抄的。换机器人就是另一套。
- **控制频率 20 Hz**（`third_party/libero/libero/libero/envs/env_wrapper.py:27`），控制器 OSC_POSE（同文件 :17）：7 维动作 = 末端位姿增量 6 + 夹爪 1，范围 [-1, 1]。这决定了 chunk 10 步 = 0.5 秒。
- **max_steps**：每个 suite 一个上限（最长示教 + 余量）。超时算失败，不是"还没做完"。

## 9. 常见报错

（setup 作业跑完后补）
