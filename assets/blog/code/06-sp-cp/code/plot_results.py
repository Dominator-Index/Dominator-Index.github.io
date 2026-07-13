"""Part 06 experiment figure: ring attention breakdown for CP=2/4/8 (same layout as the part 05 TP figure, read them side by side)."""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
rows = list(csv.DictReader(open(HERE.parent / "results" / "ringattn.csv")))

apply()
fig, ax = plt.subplots(figsize=(8.2, 4.8))
x = range(len(rows))
comp = [float(r["compute_ms"]) for r in rows]
comm = [max(0.0, float(r["comm_ms"])) for r in rows]  # the -0.06 at cp=2 is measurement noise, clamp to 0
ax.bar(x, comp, width=0.5, color=SERIES[0], label="compute (blockwise attention)")
ax.bar(x, comm, width=0.5, bottom=comp, color=SERIES[4], label="KV ring exchange (p2p)")
for i, r in enumerate(rows):
    ax.text(i, comp[i] / 2, f"{comp[i]:.2f}", ha="center", va="center", fontsize=10, color="#0b0f19", fontweight="bold")
    if comm[i] > 0.3:
        ax.text(i, comp[i] + comm[i] / 2, f"{comm[i]:.2f}", ha="center", va="center", fontsize=10, color="#0b0f19", fontweight="bold")
    tot = comp[i] + comm[i]
    ax.text(i, tot + 0.15, f"total {tot:.1f} ms", ha="center", fontsize=10.5, color=SERIES[3], fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels([f"CP = {r['cp']}" for r in rows])
ax.set_ylabel("time per attention forward (ms)")
ax.set_ylim(0, 10.2)
ax.set_title("Same PCIe box where TP flatlined — CP scales (s = 8192, non-causal)")
ax.annotate("compute is O(s²/N) — quadratic —\nwhile the ring moves only O(s/N):\nthe square buys what TP couldn't afford",
            xy=(1.28, 4.2), xytext=(0.68, 7.2), fontsize=9.5, color=GLOW)
ax.annotate("exactness: max err ≈ 4e-7 (fp32 rounding)\nat every CP — online softmax is not an approximation",
            xy=(0, 8.6), xytext=(-0.35, 9.5), fontsize=9, color=TEXT2)
ax.legend(loc="center right")
save(fig, FIG / "fig-3-cp-scaling")
print("done")
