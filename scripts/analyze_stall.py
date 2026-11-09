"""notes/07 的分析：从 stall_study 各条件的 episodes.csv + traj/task*.npz 里量三件事。

1. 成功率（Wilson 95% CI）和按 (task, init) 配对的翻转数。
2. 失败的 episode 结尾是不是停滞、停在什么夹爪状态、停了多久。
3. 接缝（相邻两次采样重叠部分）的不一致程度；停滞状态下 probe 出来的动作分布。

  external/LIBERO-plus/.venv/bin/python scripts/analyze_stall.py results/stall_study [--base k5_base]
输出：打印表格，并写 <dir>/summary.csv、<dir>/episodes_metrics.csv。
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

STALL_RADIUS = 0.03   # 末端停在终点 3 cm 球内算"没动"
STALL_MIN = 100       # 至少持续 100 步（5 秒）才算停滞
K_HEAD = 5            # 看 chunk 的前几步


def wilson(s, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def load(cond_dir):
    """→ {(task, init, rep): {act, eef, grip, chunk_t, chunk, probe_t, probe, success}}"""
    eps = {}
    for f in sorted(glob.glob(os.path.join(cond_dir, "traj", "task*.npz"))):
        task = int(os.path.basename(f)[4:-4])
        z = np.load(f)
        for key in z.files:
            ep, name = key.split("/")
            i, r = ep.split("_")
            eps.setdefault((task, int(i[1:]), int(r[1:])), {})[name] = z[key]
    return eps


def episode_metrics(e, k):
    act, eef = e["act"], e["eef"]
    T = len(act)
    m = {"steps": T, "success": int(e["success"])}
    # --- 结尾停滞：从哪一步起末端再没离开终点 3 cm ---
    dist = np.linalg.norm(eef - eef[-1], axis=1)
    moved = np.nonzero(dist > STALL_RADIUS)[0]
    onset = int(moved[-1]) + 1 if len(moved) else 0
    m["stall_len"] = T - onset
    m["stalled"] = int(m["stall_len"] >= STALL_MIN and not m["success"])
    tail = act[max(onset, T - STALL_MIN):]
    m["tail_grip_cmd"] = float(np.mean(np.sign(tail[:, 6])))        # +1 一直在合拢（抓着），-1 一直张开
    m["tail_move"] = float(np.mean(np.linalg.norm(tail[:, :3], axis=1)))  # 结尾每步平移指令的平均幅度（归一化单位）
    # --- 执行动作的高频成分 ---
    d = np.diff(act, axis=0)
    m["jerk_pos"] = float(np.mean(np.sum(d[:, :3] ** 2, axis=1)))
    m["grip_flips"] = int(np.sum(np.sign(act[1:, 6]) != np.sign(act[:-1, 6])))
    # --- 接缝：上一段没执行的部分 vs 新一段的开头 ---
    ch = e["chunk"]
    H = ch.shape[1]
    if len(ch) > 1 and k < H:
        prev, new = ch[:-1, k:], ch[1:, : H - k]
        m["seam_pos"] = float(np.mean(np.linalg.norm(new[:, 0, :3] - prev[:, 0, :3], axis=1)))
        m["seam_grip_disagree"] = float(np.mean(np.sign(new[:, :, 6]) != np.sign(prev[:, :, 6])))
        inner = np.linalg.norm(np.diff(ch[:, :, :3], axis=1), axis=2)
        m["inner_pos"] = float(np.mean(inner))   # chunk 内部相邻步的差，作为接缝的参照
    if "rtc_diag" in e and len(e["rtc_diag"]):
        m["_diag"] = e["rtc_diag"].mean(axis=0)
    # --- probe：同一观测下 16 条采样 ---
    if "probe" in e:
        pt, pr = e["probe_t"], e["probe"]                       # (P,), (P, N, H, 7)
        g_open = np.mean(pr[:, :, :K_HEAD, 6], axis=2) < 0      # 每条样本前 5 步的夹爪倾向：张开？
        p_open = g_open.mean(axis=1)
        mag = np.linalg.norm(pr[:, :, :K_HEAD, :3], axis=3).mean(axis=2)   # (P, N) 每条样本想移动多快
        head = pr[:, :, :K_HEAD, :3].mean(axis=2)               # (P, N, 3)
        spread = np.linalg.norm(head - head.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
        m["_probe"] = {"t": pt, "p_open": p_open, "mag": mag.mean(axis=1), "mag_max": mag.max(axis=1), "spread": spread}
        if m["stalled"]:
            sel = pt >= onset + 20     # 停滞开始 1 秒之后的 probe
            if sel.any():
                m["stall_p_open"] = float(p_open[sel].mean())
                m["stall_split"] = float(np.minimum(p_open[sel], 1 - p_open[sel]).mean())
                m["stall_mag"] = float(mag[sel].mean())
                m["stall_mag_max"] = float(mag[sel].max(axis=1).mean())
                m["stall_spread"] = float(spread[sel].mean())
        m["all_mag"] = float(mag.mean())
        m["all_spread"] = float(spread.mean())
        m["all_split"] = float(np.minimum(p_open, 1 - p_open).mean())
    return m


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="k5_base")
    args = ap.parse_args()

    conds, rows, per_ep = {}, [], []
    for d in sorted(glob.glob(os.path.join(args.root, "*", ""))):
        name = os.path.basename(d.rstrip("/"))
        csv_f = os.path.join(d, "episodes.csv")
        if not os.path.exists(csv_f):
            continue
        ep_rows = list(csv.DictReader(open(csv_f)))
        k = int(ep_rows[0]["replan_steps"])
        eps = load(d)
        ms = {key: episode_metrics(e, k) for key, e in eps.items()}
        conds[name] = {"k": k, "csv": ep_rows, "m": ms}

    for name, c in conds.items():
        rs = c["csv"]
        n, s = len(rs), sum(int(r["success"]) for r in rs)
        rate, lo, hi = wilson(s, n)
        t8 = [int(r["success"]) for r in rs if r["task_id"] == "8"]
        ms = list(c["m"].values())
        fails = [m for m in ms if not m["success"]]
        stalled = [m for m in fails if m["stalled"]]
        row = {
            "cond": name, "k": c["k"], "n": n, "succ": s, "rate": round(rate, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
            "t8_succ": sum(t8), "t8_n": len(t8), "non_t8_fail": (n - s) - (len(t8) - sum(t8)),
            "infer_ms": round(mean([float(r["mean_infer_ms"]) for r in rs]), 1),
            "fail_logged": len(fails), "fail_stalled": len(stalled),
            "stall_holding": sum(m["tail_grip_cmd"] > 0.5 for m in stalled),
            "stall_open": sum(m["tail_grip_cmd"] < -0.5 for m in stalled),
            "stall_len_med": float(np.median([m["stall_len"] for m in stalled])) if stalled else float("nan"),
            "seam_pos": round(mean([m.get("seam_pos") for m in ms]), 4),
            "inner_pos": round(mean([m.get("inner_pos") for m in ms]), 4),
            "seam_grip_disagree": round(mean([m.get("seam_grip_disagree") for m in ms]), 4),
            "jerk_pos": round(mean([m["jerk_pos"] for m in ms]), 4),
            "grip_flips_succ": round(mean([m["grip_flips"] for m in ms if m["success"]]), 2),
            "grip_flips_fail": round(mean([m["grip_flips"] for m in fails]), 2),
            "stall_p_open": round(mean([m.get("stall_p_open") for m in stalled]), 3),
            "stall_split": round(mean([m.get("stall_split") for m in stalled]), 3),
            "stall_mag": round(mean([m.get("stall_mag") for m in stalled]), 3),
            "stall_mag_max": round(mean([m.get("stall_mag_max") for m in stalled]), 3),
            "all_mag": round(mean([m.get("all_mag") for m in ms]), 3),
            "stall_spread": round(mean([m.get("stall_spread") for m in stalled]), 3),
            "all_spread": round(mean([m.get("all_spread") for m in ms]), 3),
        }
        rows.append(row)
        for (task, init, rep), m in c["m"].items():
            per_ep.append({"cond": name, "task_id": task, "init_state_id": init, "repeat": rep,
                           **{a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items() if not a.startswith("_")}})

    def table(cols):
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join(str(r[c]) for c in cols))
        print()

    for name, c in conds.items():
        dg = [m["_diag"] for m in c["m"].values() if "_diag" in m]
        if dg:
            d = np.mean(dg, axis=0)
            print("rtc_diag %s: overlap rmse guided %.4f vs plain %.4f | first-step rmse guided %.4f vs plain %.4f | grip agree guided %.3f vs plain %.3f\n" % (name, *d))
    table(["cond", "k", "n", "succ", "rate", "ci_lo", "ci_hi", "t8_succ", "t8_n", "non_t8_fail", "infer_ms"])
    table(["cond", "fail_logged", "fail_stalled", "stall_holding", "stall_open", "stall_len_med", "grip_flips_succ", "grip_flips_fail"])
    table(["cond", "seam_pos", "inner_pos", "seam_grip_disagree", "jerk_pos"])
    table(["cond", "stall_p_open", "stall_split", "stall_mag", "stall_mag_max", "all_mag", "stall_spread", "all_spread"])

    # --- 配对：同一个 (task, init) 在基线和各条件下的成败 ---
    if args.base in conds:
        base = {(r["task_id"], r["init_state_id"]): int(r["success"]) for r in conds[args.base]["csv"]}
        print("paired vs %s:  cond | both_ok | base_only | cond_only | both_fail | sign-test p" % args.base)
        for name, c in conds.items():
            if name == args.base:
                continue
            cur = {(r["task_id"], r["init_state_id"]): int(r["success"]) for r in c["csv"]}
            keys = [x for x in base if x in cur]
            a = sum(base[x] and cur[x] for x in keys); b = sum(base[x] and not cur[x] for x in keys)
            cc = sum(cur[x] and not base[x] for x in keys); dd = sum(not base[x] and not cur[x] for x in keys)
            nn = b + cc
            p = min(1.0, 2 * sum(math.comb(nn, j) for j in range(0, min(b, cc) + 1)) / 2 ** nn) if nn else 1.0
            print("  %s | %d | %d | %d | %d | %.3f" % (name, a, b, cc, dd, p))

    for fn, data in (("summary.csv", rows), ("episodes_metrics.csv", per_ep)):
        if data:
            cols = sorted({c for r in data for c in r}, key=lambda c: (c not in data[0], list(data[0]).index(c) if c in data[0] else 0, c))
            with open(os.path.join(args.root, fn), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(data)


if __name__ == "__main__":
    main()
