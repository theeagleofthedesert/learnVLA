"""在 LIBERO-Plus 的某个 suite 里按扰动类别抽任务，打印 suite 内的任务索引（空格分隔），供 eval_libero.py --task-ids 用。
要在 LIBERO-plus 的 venv 里跑（需要 import libero）。

  python scripts/plus_select.py --suite libero_10 --category "Camera Viewpoints" --n 30 --seed 0
  python scripts/plus_select.py --suite libero_10 --category all --n 30      # 每类各 30 个
  python scripts/plus_select.py --suite libero_10 --list-categories
  python scripts/plus_select.py --suite libero_10 --n 0 --exclude-csv results/libero_plus/libero_10/episodes.csv  # 续跑：去掉已有结果的任务
"""
import argparse, csv, json, pathlib, random, collections
from libero.libero import benchmark, get_libero_path

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)
ap.add_argument("--category", default="all")
ap.add_argument("--n", type=int, default=30, help="每类抽几个；0 = 全部")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--list-categories", action="store_true")
ap.add_argument("--exclude-csv", default=None, help="已有的 episodes.csv；其中出现过的 task_name 不再选（作业超时后续跑用）")
a = ap.parse_args()

cls = json.load(open(pathlib.Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"))[a.suite]
suite = benchmark.get_benchmark_dict()[a.suite]()
name2idx = {suite.get_task(i).name: i for i in range(suite.n_tasks)}
done = set()
if a.exclude_csv and pathlib.Path(a.exclude_csv).exists():
    done = {r["task_name"] for r in csv.DictReader(open(a.exclude_csv))}
bycat = collections.defaultdict(list)
for t in cls:
    if t["name"] in name2idx and t["name"] not in done:
        bycat[t["category"]].append(name2idx[t["name"]])
if a.list_categories:
    for c, ids in sorted(bycat.items()):
        print(f"{c:22s} {len(ids)}")
    raise SystemExit
cats = sorted(bycat) if a.category == "all" else [a.category]
rng = random.Random(a.seed)
out = []
for c in cats:
    ids = sorted(bycat[c])
    out += ids if a.n == 0 else rng.sample(ids, min(a.n, len(ids)))
print(" ".join(map(str, sorted(out))))
