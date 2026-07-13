"""Result plots for post 01. Data: ../results/collectives_n{2,4,8}.csv"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, bytes_ticks, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"


def load(n):
    rows = []
    with open(RES / f"collectives_n{n}.csv") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "op" else float(v)) for k, v in r.items()})
    return rows


apply()

# ---- fig 4: six primitives, busbw vs S (N=8) -----------------------------
data = load(8)
ops = [  # (op, label, color) semantic colors: the protagonist all_reduce = s1
    ("all_reduce", "all-reduce", SERIES[0]),
    ("all_gather", "all-gather", SERIES[3]),
    ("reduce_scatter", "reduce-scatter", SERIES[2]),
    ("broadcast", "broadcast", SERIES[4]),
    ("scatter", "scatter", SERIES[5]),
    ("gather", "gather", SERIES[1]),
]
fig, ax = plt.subplots(figsize=(8.6, 5))
sizes = sorted({int(r["bytes"]) for r in data})
for op, label, c in ops:
    xs = [int(r["bytes"]) for r in data if r["op"] == op]
    ys = [r["busbw_GBps"] for r in data if r["op"] == op]
    ax.plot(xs, ys, marker="o", color=c, label=label)
bytes_ticks(ax, sizes)
ax.set_ylim(0, 56)
ax.set_xlabel("message size S (logical tensor bytes)")
ax.set_ylabel("bus bandwidth (GB/s)")
ax.set_title("Ring collectives hit one shared ceiling; root-P2P ops don't (8 GPUs, PCIe)")
ax.annotate("ring ops: all links busy both ways,\nthrottled by the slowest (UPI) crossing",
            xy=(2**30, 18), xytext=(2**23.5, 30), fontsize=9.5, color=TEXT2,
            arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.annotate("scatter/gather: one-way flows\nout of / into the root",
            xy=(2**30, 50), xytext=(2**21.6, 44.5), fontsize=9.5, color=TEXT2,
            arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.legend(ncol=2, loc="upper left")
save(fig, FIG / "fig-4-primitives-busbw")

# ---- fig 5: all-reduce algbw vs busbw, N=2/4/8 ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
for i, (key, title, ylab) in enumerate([
        ("algbw_GBps", "what you feel: algorithm bandwidth S/t", "algorithm bandwidth (GB/s)"),
        ("busbw_GBps", "what the wires do: bus bandwidth = S/t &#183; 2(N-1)/N".replace("&#183;", "·"), "bus bandwidth (GB/s)")]):
    ax = axes[i]
    for j, n in enumerate([2, 4, 8]):
        rows = [r for r in load(n) if r["op"] == "all_reduce"]
        ax.plot([int(r["bytes"]) for r in rows], [r[key] for r in rows],
                marker="o", color=SERIES[j], label=f"N = {n}")
    bytes_ticks(ax, [x for x in sizes if x != 256 * 2**20])
    ax.set_ylim(0, 24)
    ax.set_xlabel("message size S")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11.5)
axes[0].legend()
axes[1].annotate("collapses to one curve:\nthe 2(N−1)/N factor is real",
                 xy=(2**28, 19), xytext=(2**21.2, 6.5), fontsize=10, color=GLOW,
                 arrowprops=dict(arrowstyle="->", color=GLOW, lw=1.2))
fig.suptitle("All-reduce across 2 / 4 / 8 GPUs", fontsize=13, fontweight="bold")
save(fig, FIG / "fig-5-allreduce-scaling")

# ---- fig 6: the latency floor (time vs S, log-log) ------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for j, n in enumerate([2, 4, 8]):
    rows = [r for r in load(n) if r["op"] == "all_reduce"]
    ax.plot([int(r["bytes"]) for r in rows], [r["time_ms"] for r in rows],
            marker="o", color=SERIES[j], label=f"N = {n}")
bytes_ticks(ax, sizes)
ax.set_yscale("log")
ax.set_xlabel("message size S")
ax.set_ylabel("time per all-reduce (ms, log)")
ax.set_title("Below ~1 MiB the wire time vanishes — you are paying pure latency")
ax.axvspan(sizes[0] * 0.7, 2**20, color=SERIES[1], alpha=0.08)
ax.text(2**14.5, 20, "latency-bound\n(flat: ~10-70 µs,\ngrows with N, not with S)", fontsize=9.5, color=TEXT2)
ax.text(2**26.5, 0.06, "bandwidth-bound\n(time ∝ S)", fontsize=9.5, color=TEXT2)
ax.legend(loc="center right")
save(fig, FIG / "fig-6-latency-floor")

print("all figures done")
