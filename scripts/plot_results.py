"""把 episodes.csv 汇总成小 csv + 图。用 LIBERO-plus venv 跑（有 matplotlib 3.5）：
  external/LIBERO-plus/.venv/bin/python scripts/plot_results.py
产出：
  results/inference_ablation.csv / .png        §1.3 replan × 去噪步数
  results/stochasticity_by_init.csv / .png     §1.5 每初始状态 10 次重复的成功数直方图
  results/libero_plus_by_category.csv / _by_difficulty.csv / libero_plus.png   §1.4 扰动类 × suite
"""
import csv, glob, json, math, collections, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from summarize import wilson

# 配色：单系列用 slot1 蓝，第二系列橙；文字用 ink，网格淡灰。
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

def rows(path):
    return list(csv.DictReader(open(path)))

def grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---------- §1.3 推理期消融 ----------
abl = []
for f in sorted(glob.glob(str(R / "ablation_inference/*/episodes.csv"))):
    rs = rows(f); n = len(rs); k = sum(int(r["success"]) for r in rs)
    p, lo, hi = wilson(k, n)
    abl.append(dict(tag=rs[0]["tag"], replan=int(rs[0]["replan_steps"]), num_steps=int(rs[0]["tag"].split("nsteps")[1]),
                    succ=k, n=n, rate=round(100 * p, 1), ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1),
                    n_infer_per_ep=round(sum(float(r["n_infer"]) for r in rs) / n, 1),
                    infer_ms=round(sum(float(r["mean_infer_ms"]) for r in rs) / n, 1),
                    wall_s_per_ep=round(sum(float(r["wall_s"]) for r in rs) / n, 1),
                    t8_succ=sum(int(r["success"]) for r in rs if r["task_id"] == "8")))
abl.sort(key=lambda d: (d["num_steps"] != 10, d["replan"], d["num_steps"]))
with open(R / "inference_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(abl[0])); w.writeheader(); w.writerows(abl)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, key, fixed, xlabel in [(axes[0], "replan", ("num_steps", 10), "replan_steps (10 = whole chunk, open loop)"),
                               (axes[1], "num_steps", ("replan", 5), "flow-matching denoising steps")]:
    pts = sorted([d for d in abl if d[fixed[0]] == fixed[1]], key=lambda d: d[key])
    x = [str(d[key]) for d in pts]; y = [d["rate"] for d in pts]
    err = [[d["rate"] - d["ci_lo"] for d in pts], [d["ci_hi"] - d["rate"] for d in pts]]
    ax.errorbar(x, y, yerr=err, color=BLUE, lw=2, marker="o", ms=6, capsize=3, mfc="#fcfcfb", mew=2)
    for xi, d in zip(x, pts):
        ax.annotate(f"{d['rate']:.1f}", (xi, d["ci_hi"]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK2)
    ax.set_xlabel(xlabel); grid(ax)
axes[0].set_ylabel("libero_10 success %\n(500 episodes, Wilson 95% CI)"); axes[0].set_ylim(78, 100)
fig.suptitle("Inference-time ablation: official pi05_libero, only inference parameters changed", x=0.02, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(R / "inference_ablation.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ---------- §1.5 随机性 ----------
st = collections.defaultdict(list)
for r in rows(R / "stochasticity/episodes.csv"):
    st[(int(r["task_id"]), int(r["init_state_id"]))].append(int(r["success"]))
off = {(int(r["task_id"]), int(r["init_state_id"])): int(r["success"]) for r in rows(R / "eval_official/libero_10/episodes.csv")}
out = []
for (t, i), v in sorted(st.items()):
    k, n = sum(v), len(v); p, lo, hi = wilson(k, n)
    out.append(dict(task_id=t, init_state_id=i, succ=k, n=n, rate=round(100 * p, 1), ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1),
                    official_success=off.get((t, i), "")))
with open(R / "stochasticity_by_init.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

ks = list(range(11))
h8 = [sum(1 for d in out if d["task_id"] == 8 and d["succ"] == k) for k in ks]
ho = [sum(1 for d in out if d["task_id"] != 8 and d["succ"] == k) for k in ks]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.bar(ks, ho, color=BLUE, width=0.8, label="tasks 0-7, 9 (180 init states)", edgecolor="#fcfcfb", lw=1)
ax.bar(ks, h8, bottom=ho, color=ORANGE, width=0.8, label="task 8, both moka pots (20 init states)", edgecolor="#fcfcfb", lw=1)
for k, (a, b) in enumerate(zip(ho, h8)):
    if a + b: ax.annotate(str(a + b), (k, a + b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK2)
ax.set_xticks(ks); ax.set_xlabel("successes out of 10 repeats of the same init state"); ax.set_ylabel("number of init states"); grid(ax)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Stochasticity: libero_10, 10 tasks x 20 init states x 10 sampling repeats", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(R / "stochasticity_hist.png", dpi=160); plt.close(fig)

# ---------- §1.4 LIBERO-Plus ----------
cls = json.load(open(ROOT / "external/LIBERO-plus/libero/libero/benchmark/task_classification.json"))
cat = {(s, t["name"]): (t["category"], t.get("difficulty_level")) for s, ts in cls.items() for t in ts}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Objects Layout", "Sensor Noise", "Background Textures", "Light Conditions"]
g = collections.defaultdict(lambda: [0, 0]); gd = collections.defaultdict(lambda: [0, 0])
for s in SUITES:
    f = R / f"libero_plus/{s}/episodes.csv"
    if not f.exists(): continue
    for r in rows(f):
        c, d = cat[(s, r["task_name"])]; k = int(r["success"])
        g[(s, c)][0] += k; g[(s, c)][1] += 1; g[("all", c)][0] += k; g[("all", c)][1] += 1
        gd[(c, d)][0] += k; gd[(c, d)][1] += 1
with open(R / "libero_plus_by_category.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["suite", "category", "succ", "n", "rate", "ci_lo", "ci_hi"])
    for s in SUITES + ["all"]:
        for c in CATS:
            if (s, c) in g:
                k, n = g[(s, c)]; p, lo, hi = wilson(k, n)
                w.writerow([s, c, k, n, round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)])
with open(R / "libero_plus_by_difficulty.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["category", "difficulty_level", "succ", "n", "rate", "ci_lo", "ci_hi"])
    for c in CATS:
        for d in [1, 2, 3, 4, 5, None]:
            if (c, d) in gd:
                k, n = gd[(c, d)]; p, lo, hi = wilson(k, n)
                w.writerow([c, d if d is not None else "", k, n, round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)])

fig = plt.figure(figsize=(9.6, 5.4))
gs = fig.add_gridspec(2, 7, height_ratios=[1.15, 1], hspace=0.7, wspace=0.15)
ax = fig.add_subplot(gs[0, :])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#e8f0fb", "#2a78d6", "#0f3f7a"])
norm = matplotlib.colors.Normalize(40, 100)
for i, s in enumerate(SUITES):
    for j, c in enumerate(CATS):
        if (s, c) in g:
            k, n = g[(s, c)]; p = 100 * k / n
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=cmap(norm(p))))
            ax.text(j + 0.5, i + 0.5, f"{p:.0f}", ha="center", va="center", fontsize=9, color="white" if p < 80 else INK)
        else:
            ax.text(j + 0.5, i + 0.5, "running", ha="center", va="center", fontsize=8, color=INK2)
ax.set_xlim(0, len(CATS)); ax.set_ylim(len(SUITES), 0)
ax.set_xticks([j + 0.5 for j in range(len(CATS))]); ax.set_xticklabels([c.replace(" ", "\n") for c in CATS], fontsize=8)
ax.set_yticks([i + 0.5 for i in range(len(SUITES))]); ax.set_yticklabels([s.replace("libero_", "") for s in SUITES])
ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
ax.set_title("LIBERO-Plus success % by perturbation (official pi05_libero, 1 episode per variant task; plain LIBERO: 98.2 / 98.8 / 97.6 / 93.6)", loc="left", fontsize=9.5)
axes = [fig.add_subplot(gs[1, j]) for j in range(7)]
for ax, c in zip(axes, CATS):
    xs = [d for d in [1, 2, 3, 4, 5] if (c, d) in gd]; ys = [100 * gd[(c, d)][0] / gd[(c, d)][1] for d in xs]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=4); ax.set_ylim(35, 102); ax.set_xticks([1, 3, 5]); ax.set_xlim(0.7, 5.3)
    ax.set_title(c.replace(" ", "\n"), fontsize=8); grid(ax)
    if ax is not axes[0]: ax.set_yticklabels([])
    ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-2, -11), ha="center", fontsize=7.5, color=INK2)
axes[0].set_ylabel("success %"); axes[3].set_xlabel("perturbation difficulty_level 1-5 (LIBERO-Plus, all 4 suites pooled)", fontsize=8.5)
fig.savefig(R / "libero_plus.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("ok")
