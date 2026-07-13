"""Result plots for post 04. Data: ../results/fsdp.csv"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
rows = {r["mode"]: r for r in csv.DictReader(open(HERE.parent / "results" / "fsdp.csv"))}

apply()

modes = ["ddp", "fsdp2", "fsdp2-noreshard"]
labels = ["DDP", "FSDP2\nreshard=True\n(ZeRO-3)", "FSDP2\nreshard=False\n(ZeRO-2)"]
colors = [SERIES[1], SERIES[0], SERIES[3]]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.25})

# left: memory
ax = axes[0]
res = [float(rows[m]["resident_GiB"]) for m in modes]
pk = [float(rows[m]["peak_GiB"]) for m in modes]
x = range(3)
ax.bar([i - 0.19 for i in x], res, width=0.36, color=colors, label="resident after step")
ax.bar([i + 0.19 for i in x], pk, width=0.36, color=colors, alpha=0.45, label="peak during step")
for i in x:
    ax.text(i - 0.19, res[i] + 0.5, f"{res[i]:.1f}", ha="center", fontsize=10.5, color=colors[i], fontweight="bold")
    ax.text(i + 0.19, pk[i] + 0.5, f"{pk[i]:.1f}", ha="center", fontsize=10.5, color=colors[i])
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("per-GPU memory (GiB)")
ax.set_ylim(0, 34)
ax.set_title("FSDP2 vs DDP memory: 11.8 → 1.3 GiB resident (9.3×)", fontsize=11.5)
ax.annotate("peak gap = 1.3 GiB\n≈ one full bf16 model\nkept un-resharded",
            xy=(2.19, 14.17), xytext=(1.32, 24), fontsize=9, color=TEXT2,
            arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.legend(fontsize=9)

# right: step time
ax = axes[1]
ms = [float(rows[m]["step_ms"]) for m in modes]
ax.bar(x, ms, width=0.5, color=colors, alpha=0.92)
for i in x:
    ax.text(i, ms[i] + 8, f"{ms[i]:.0f} ms", ha="center", fontsize=10.5, color=colors[i], fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("time per step (ms)")
ax.set_ylim(0, 560)
ax.set_title("step time: skipping the backward re-AG saves 65 ms\n≈ one bf16-model all-gather, priced from post #1", fontsize=10.5)
ax.annotate("", xy=(2, 273), xytext=(2, 338), arrowprops=dict(arrowstyle="->", color=GLOW, lw=1.6))
ax.text(1.62, 296, "−65 ms", fontsize=10, color=GLOW, fontweight="bold")

save(fig, FIG / "fig-3-fsdp-vs-ddp")
print("done")
