"""汇总 episodes.csv：按 (tag, suite[, task]) 算成功率和 Wilson 95% 置信区间。纯标准库，任何 python3 都能跑。

用法：
  python3 scripts/summarize.py results/eval_official/*/episodes.csv            # suite 级
  python3 scripts/summarize.py --by task results/eval_official/libero_10/episodes.csv
  python3 scripts/summarize.py --by init results/stochasticity/episodes.csv    # 每个初始状态的成功率（§1.5）
"""
import argparse
import csv
import math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--by", choices=["suite", "task", "init", "replan"], default="suite")
    a = ap.parse_args()

    groups = defaultdict(lambda: [0, 0, 0.0])  # k, n, sum_steps
    for path in a.csvs:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["tag"], r["suite"])
                if a.by == "task":
                    key += (int(r["task_id"]), r["task"][:50])
                elif a.by == "init":
                    key += (int(r["task_id"]), int(r["init_state_id"]))
                elif a.by == "replan":
                    key += (int(r["replan_steps"]),)
                g = groups[key]
                g[0] += int(r["success"]); g[1] += 1; g[2] += float(r["steps"])

    print(f"{'group':<70} {'succ':>5} {'n':>4} {'rate':>6} {'95% CI':>14} {'steps':>6}")
    for key in sorted(groups):
        k, n, s = groups[key]
        p, lo, hi = wilson(k, n)
        name = " / ".join(map(str, key))
        print(f"{name:<70} {k:>5} {n:>4} {100*p:>5.1f}% [{100*lo:>5.1f},{100*hi:>5.1f}] {s/n:>6.0f}")
    if a.by == "init":
        # 随机性分离：有多少初始状态是"全成"或"全败"
        bim = sum(1 for k_, n_, _ in groups.values() if k_ == 0 or k_ == n_)
        print(f"\n初始状态总数 {len(groups)}，其中全成或全败（双峰）的 {bim} 个 = {100*bim/len(groups):.0f}%")


if __name__ == "__main__":
    main()
