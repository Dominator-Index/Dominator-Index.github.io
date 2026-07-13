"""第 02 篇实验图。数据:../results/ddp.csv"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"

rows = list(csv.DictReader(open(HERE.parent / "results" / "ddp.csv")))
single_ms = float(next(r for r in rows if r["mode"] == "single")["step_ms"])
sweep = [(float(r["bucket_cap_mb"]), float(r["step_ms"]), float(r["ktok_per_s"]))
         for r in rows if r["mode"] == "ddp" and r["gas"] == "1"]

apply()

# ---- fig 3: bucket_cap_mb 扫描 -------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.8))
xs = [b for b, _, _ in sweep]
ys = [t for _, t, _ in sweep]
ax.plot(xs, ys, marker="o", color=SERIES[0], label="DDP step time (8 GPUs)", zorder=3)
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"{b:g}" for b in xs])
ax.minorticks_off()
ax.axhline(single_ms, color=TEXT2, ls=":", lw=1.4)
ax.text(1.05, single_ms - 4, f"single-GPU compute time ({single_ms:.0f} ms) — the floor", fontsize=9.5, color=TEXT2)
full_ar = 49.8
ax.axhline(single_ms + full_ar, color=GLOW, ls="--", lw=1.4)
ax.text(1.05, single_ms + full_ar + 1.5,
        f"floor + one FULL un-overlapped all-reduce (+{full_ar:.0f} ms, priced from post #1)",
        fontsize=9.5, color=GLOW)
best = min(sweep, key=lambda r: r[1])
ax.annotate(f"sweet spot: {best[0]:g} MB\n(hides half the comm)",
            xy=(best[0], best[1]), xytext=(4.5, 138), fontsize=10, color=SERIES[3],
            arrowprops=dict(arrowstyle="->", color=SERIES[3], lw=1.2))
ax.annotate("one giant bucket:\nzero overlap — lands exactly\non the un-overlapped line",
            xy=(500, sweep[-1][1]), xytext=(60, 176), fontsize=10, color=SERIES[1],
            arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.2))
ax.set_xlabel("bucket_cap_mb")
ax.set_ylabel("time per step (ms)")
ax.set_ylim(115, 185)
ax.set_title("The bucket-size U-curve: too small pays latency, too big loses overlap")
ax.legend(loc="lower right")
save(fig, FIG / "fig-3-bucket-sweep")

# ---- fig 4: 吞吐对比条形图 -------------------------------------------------
gas4 = {r["no_sync"]: float(r["ktok_per_s"]) for r in rows if r["gas"] == "4"}
single_tps = float(next(r for r in rows if r["mode"] == "single")["ktok_per_s"])
ideal = single_tps * 8
bars = [
    ("ideal\n8 × single", ideal, TEXT2),
    ("DDP gas=1\n(sync every step)", max(t for _, _, t in sweep), SERIES[0]),
    ("DDP gas=4\nsync every micro", gas4["0"], SERIES[2]),
    ("DDP gas=4\n+ no_sync", gas4["1"], SERIES[3]),
]
fig, ax = plt.subplots(figsize=(8.0, 4.6))
xpos = range(len(bars))
for i, (lab, v, c) in enumerate(bars):
    ax.bar(i, v, width=0.62, color=c, alpha=0.4 if i == 0 else 0.92,
           edgecolor=c if i == 0 else "none", linestyle=":" if i == 0 else "-")
    eff = v / ideal * 100
    ax.text(i, v + 12, f"{v:.0f}" + ("" if i == 0 else f"  ({eff:.0f}%)"),
            ha="center", fontsize=10.5,
            color=c if i else TEXT2, fontweight="bold")
ax.set_xticks(list(xpos))
ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
ax.set_ylabel("throughput (ktok/s)")
ax.set_ylim(0, 900)
ax.set_title("Gradient accumulation + no_sync buys back most of the comm tax")
save(fig, FIG / "fig-4-throughput")

print("done")
