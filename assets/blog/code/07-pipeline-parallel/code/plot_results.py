"""Part 07 experiment figure: GPipe bubble, measured vs theory."""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
rows = list(csv.DictReader(open(HERE.parent / "results" / "gpipe.csv")))
m = [int(r["m"]) for r in rows]
slot = [float(r["slot_ms"]) for r in rows]
step = [float(r["step_ms"]) for r in rows]
bt = [float(r["bubble_theory"]) for r in rows]
bm = [float(r["bubble_measured"]) for r in rows]
p = 4

apply()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.24})

# left: U-shaped step time
ax = axes[0]
theory = [(mi + p - 1) * s for mi, s in zip(m, slot)]
ax.plot(m, step, marker="o", color=SERIES[0], label="measured step time")
ax.plot(m, theory, marker="s", ms=5, ls="--", color=GLOW, label="bubble-only theory (m+p−1)·t_slot")
ax.set_xscale("log", base=2)
ax.set_xticks(m)
ax.set_xticklabels([str(x) for x in m])
ax.minorticks_off()
ax.set_xlabel("microbatch count m (total batch fixed)")
ax.set_ylabel("time per step (ms)")
ax.set_title("The other U-curve: bubble shrinks, per-microbatch\noverhead grows — optimum in the middle", fontsize=11)
ax.annotate("m=1 = naive model parallel:\ntheory and measurement agree", xy=(1, 91.2), xytext=(1.4, 100),
            fontsize=9, color=TEXT2, arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.annotate("sweet spot m=8", xy=(8, 48.6), xytext=(11, 68), fontsize=9.5, color=SERIES[3],
            arrowprops=dict(arrowstyle="->", color=SERIES[3], lw=1.2))
ax.annotate("tiny GEMMs + per-hop latency\ndominate", xy=(32, 86.2), xytext=(7.5, 88), fontsize=9,
            color=SERIES[1], arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1))
ax.set_ylim(0, 140)
ax.legend(fontsize=8.5, loc="lower left")

# right: bubble fraction
ax = axes[1]
ax.plot(m, [b * 100 for b in bt], marker="s", ms=5, ls="--", color=GLOW, label="theory (p−1)/(m+p−1)")
ax.plot(m, [b * 100 for b in bm], marker="o", color=SERIES[2], label="measured idle fraction")
ax.set_xscale("log", base=2)
ax.set_xticks(m)
ax.set_xticklabels([str(x) for x in m])
ax.minorticks_off()
ax.set_xlabel("microbatch count m")
ax.set_ylabel("bubble / idle fraction (%)")
ax.set_title("At m=1 the formula is exact (75.9% vs 75%);\nbeyond, overheads set a floor the formula ignores", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=8.5)
save(fig, FIG / "fig-2-bubble-measured")
print("done")
