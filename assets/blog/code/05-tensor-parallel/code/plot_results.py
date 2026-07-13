"""第 05 篇实验图:TP=2/4/8 的计算/通信分解(一层 MLP 前向)。"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
rows = list(csv.DictReader(open(HERE.parent / "results" / "tp.csv")))

apply()
fig, ax = plt.subplots(figsize=(8.2, 4.8))
x = range(len(rows))
comp = [float(r["compute_ms"]) for r in rows]
comm = [float(r["comm_ms"]) for r in rows]
ax.bar(x, comp, width=0.5, color=SERIES[0], label="compute (matmuls + GeLU)")
ax.bar(x, comm, width=0.5, bottom=comp, color=SERIES[2], label="all-reduce (the g operator)")
for i, r in enumerate(rows):
    ax.text(i, comp[i] / 2, f"{comp[i]:.2f}", ha="center", va="center", fontsize=10, color="#0b0f19", fontweight="bold")
    ax.text(i, comp[i] + comm[i] / 2, f"{comm[i]:.2f}", ha="center", va="center", fontsize=10, color="#0b0f19", fontweight="bold")
    ax.text(i, comp[i] + comm[i] + 0.08, f"comm {r['comm_pct']}%", ha="center", fontsize=10.5,
            color=SERIES[2], fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels([f"TP = {r['tp']}" for r in rows])
ax.set_ylabel("time per MLP forward (ms)")
ax.set_ylim(0, 3.2)
ax.set_title("TP on PCIe does not scale: compute halves perfectly, the wire eats the winnings")
ax.annotate("compute: 1.32 → 0.66 → 0.34 ms\n(ideal ÷N scaling — the math works)",
            xy=(2, 0.17), xytext=(1.28, 1.1), fontsize=9.5, color=SERIES[0],
            arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=1.2))
ax.annotate("total: 2.39 → 2.30 → 2.43 ms\nflat — this is why TP stays\ninside NVLink nodes",
            xy=(2.25, 2.43), xytext=(1.62, 2.75), fontsize=9.5, color=GLOW)
ax.legend(loc="upper left")
save(fig, FIG / "fig-3-tp-scaling")
print("done")
