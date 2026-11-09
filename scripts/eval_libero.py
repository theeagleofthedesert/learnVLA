"""LIBERO 评测客户端。改自 openpi/examples/libero/main.py，差异：

1. 每个 episode 一行写进 CSV（任务、初始状态 id、成败、步数、推理次数、推理耗时），而不是只打日志。
2. 可以只跑部分任务 / 部分初始状态，可以对同一初始状态重复 K 次（§1.5 随机性分离）。
3. 视频只存失败（默认），文件名带任务 id 和初始状态 id，方便做 failure gallery。
4. replan_steps 从命令行给（§1.3 推理期消融）。
5. --log-traj 把每步动作、末端位置、夹爪开度和每次推理的整段 chunk 存成 traj/task{id}.npz；
   --probe-every N 每隔约 N 步让 server 对同一观测多采 probe_n 条 chunk 一起存下（notes/07）。

必须在 examples/libero/.venv (Python 3.8) 里跑，所以不能用 3.10+ 语法。
用法见 scripts/run_eval.sh。
"""
import collections
import csv
import dataclasses
import logging
import math
import pathlib
import time
from typing import List, Optional

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]   # 6 维末端增量 = 0，夹爪 -1 = 张开
LIBERO_ENV_RESOLUTION = 256                 # 训练数据渲染分辨率；之后 resize 到 224

# 每个 suite 的最大步数 = 最长示教步数 + 余量（沿用官方值）
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # --- 策略 server ---
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5          # 每次只执行 chunk 的前 replan_steps 步就重新观测

    # --- 任务选择 ---
    task_suite_name: str = "libero_spatial"
    task_ids: Optional[List[int]] = None       # None = suite 里全部任务
    init_state_ids: Optional[List[int]] = None # None = 前 num_trials_per_task 个
    num_trials_per_task: int = 50
    repeats: int = 1               # 同一初始状态重复次数（采样噪声由 server 端 rng 决定，每次不同）
    num_steps_wait: int = 10       # 仿真开头等物体落定
    max_steps: Optional[int] = None  # None = 用 MAX_STEPS 表

    # --- 输出 ---
    out_dir: str = "results/eval"
    save_videos: str = "failure"   # failure | all | none
    tag: str = ""                  # 写进 CSV 的自由标签，例如 ckpt 名
    seed: int = 7
    log_traj: bool = False         # 存逐步轨迹和每次推理的 chunk
    probe_every: int = 0           # >0：距上次 probe 至少这么多步后，下一次推理顺带采 probe_n 条
    probe_n: int = 16


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episodes.csv"
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow([
            "tag", "suite", "task_id", "task_name", "task", "init_state_id", "repeat", "replan_steps",
            "success", "steps", "n_infer", "mean_infer_ms", "wall_s", "video",
        ])

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))
    max_steps = args.max_steps or MAX_STEPS[args.task_suite_name]
    logging.info("suite=%s tasks=%s max_steps=%d replan=%d", args.task_suite_name, task_ids, max_steps, args.replan_steps)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("server metadata: %s", client.get_server_metadata())

    total, succ = 0, 0
    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)   # (50, state_dim) 固定初始状态
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        init_ids = args.init_state_ids if args.init_state_ids is not None else list(range(args.num_trials_per_task))
        traj = {}   # 本任务所有 episode 的轨迹，任务结束时写一个 npz

        for init_id in tqdm.tqdm(init_ids, desc=f"task{task_id}", leave=False):
            for rep in range(args.repeats):
                t0 = time.time()
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(init_states[init_id])
                t, done = 0, False
                n_infer, infer_ms = 0, 0.0
                frames = []
                log = collections.defaultdict(list)
                last_probe = -10 ** 9
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    # 训练数据是旋转 180 度存的，所以这里也翻
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                    if args.save_videos != "none":
                        frames.append(np.concatenate([img, wrist], axis=1))  # 左主视角，右腕部

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            # state = 末端位置(3) + 末端轴角(3) + 夹爪两指位置(2) = 8 维
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                            "prompt": str(task_description),
                        }
                        probe = args.probe_n if args.probe_every > 0 and t - last_probe >= args.probe_every else 0
                        # server 端 PolicyExt 用：一集的边界、上一段执行了几步、是否顺带多采几条
                        element["ctrl"] = {"reset": n_infer == 0, "executed": args.replan_steps if n_infer else 0, "probe": probe}
                        out = client.infer(element)
                        chunk = out["actions"]            # (action_horizon, 7)
                        if args.log_traj:
                            log["chunk_t"].append(t - args.num_steps_wait)
                            log["chunk"].append(np.asarray(chunk, dtype=np.float32))
                            if "rtc_diag" in out:
                                log["rtc_diag"].append(np.asarray(out["rtc_diag"], dtype=np.float32))
                        if probe:
                            last_probe = t
                            log["probe_t"].append(t - args.num_steps_wait)
                            log["probe"].append(np.asarray(out["probe_actions"], dtype=np.float32))
                        n_infer += 1
                        infer_ms += out.get("policy_timing", {}).get("infer_ms", 0.0)
                        assert len(chunk) >= args.replan_steps, f"chunk {len(chunk)} < replan {args.replan_steps}"
                        action_plan.extend(chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    if args.log_traj:   # 执行这一步之前的状态 + 这一步的动作
                        log["act"].append(np.asarray(action, dtype=np.float32))
                        log["eef"].append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                        log["grip"].append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
                    obs, reward, done, info = env.step(action.tolist())   # done == 任务成功判定
                    if done:
                        break
                    t += 1

                total += 1
                succ += int(done)
                for key, val in log.items():
                    traj["i%d_r%d/%s" % (init_id, rep, key)] = np.asarray(val)
                if log:
                    traj["i%d_r%d/success" % (init_id, rep)] = np.asarray(int(done))
                video = ""
                if args.save_videos == "all" or (args.save_videos == "failure" and not done):
                    video = f"{args.task_suite_name}_t{task_id}_s{init_id}_r{rep}_{'ok' if done else 'fail'}.mp4"
                    imageio.mimwrite(out_dir / "videos" / video, frames, fps=10)
                writer.writerow([
                    args.tag, args.task_suite_name, task_id, task.name, task_description, init_id, rep, args.replan_steps,
                    int(done), t - args.num_steps_wait, n_infer, round(infer_ms / max(n_infer, 1), 1),
                    round(time.time() - t0, 1), video,
                ])
                csv_f.flush()
                logging.info("task %d init %d rep %d: %s (%d/%d = %.1f%%)", task_id, init_id, rep,
                             "OK" if done else "FAIL", succ, total, 100 * succ / total)
        env.close()
        if traj:
            (out_dir / "traj").mkdir(exist_ok=True)
            np.savez_compressed(out_dir / "traj" / ("task%d.npz" % task_id), **traj)

    csv_f.close()
    logging.info("DONE %s: %d/%d = %.1f%%  -> %s", args.task_suite_name, succ, total, 100 * succ / max(total, 1), csv_path)


def _get_libero_env(task, resolution, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # 官方注释：即使用固定初始状态，seed 也会影响物体位置
    return env, task.language


def _quat2axisangle(quat):
    # 从 robosuite 抄的：四元数 (x,y,z,w) -> 轴角向量
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    eval_libero(tyro.cli(Args))   # 官方 main.py 用 tyro.cli(eval_libero)，参数会变成 --args.xxx；这里直接解析 Args
