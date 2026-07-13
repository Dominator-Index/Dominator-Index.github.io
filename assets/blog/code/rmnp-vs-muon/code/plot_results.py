"""RMNP vs Muon 独立篇实验图:fig-3 (a) GPT-2 Large 全集四方案 (b) 单矩阵尺寸扫描。"""

import csv
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
import matplotlib.pyplot as plt
import numpy as np
from plot_style import GLOW, SERIES, TEXT, TEXT2, apply, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

LABEL = {
    "rmnp_local": "RMNP, row shard\n(FSDP2 native)",
    "rmnp_colcut": "RMNP, column-cut TP\n(O(m) vector all-reduce)",
    "muon_sc": "Muon, gather +\nredundant NS",
    "muon_rr": "Muon, gather + owner NS\n+ broadcast (round-robin)",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    apply()
    opt = read_csv(os.path.join(RES, "dist_opt.csv"))
    sweep = read_csv(os.path.join(RES, "sweep.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    schemes = [r["scheme"] for r in opt]
    ms = [float(r["ms_per_step"]) for r in opt]
    mib = [float(r["mib_per_step"]) for r in opt]
    colors = [SERIES[3], SERIES[4], SERIES[1], SERIES[1]]
    y = np.arange(len(schemes))[::-1]
    bars = ax1.barh(y, ms, color=colors, height=0.6, alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABEL[s] for s in schemes], fontsize=8.5)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 2000)
    ax1.set_xlabel("optimizer precondition step (ms, log scale)")
    for yi, v, b in zip(y, ms, mib):
        note = f"{v:.1f} ms · {b:.0f} MiB/GPU moved" if b >= 1 else f"{v:.1f} ms · 0 comm"
        if v > 100:  # 长条:标注画进条内,右对齐
            ax1.annotate(note, (v * 0.9, yi), va="center", ha="right", fontsize=9,
                         color="#0b0f19", fontweight="bold")
        else:
            ax1.annotate(note, (v * 1.15, yi), va="center", fontsize=9, color=TEXT)
    ax1.set_title("(a) GPT-2 Large matrix set (708M params), 8 GPUs:\nrow-local vs full-matrix is 67×", fontsize=11.5)
    ax1.grid(axis="y", visible=False)

    sizes = [int(r["size"]) for r in sweep]
    t_r = [float(r["rmnp_ms"]) for r in sweep]
    t_m = [float(r["muon_ms"]) for r in sweep]
    ax2.plot(sizes, t_m, color=SERIES[1], marker="o", label="Muon: all-gather + NS")
    ax2.plot(sizes, t_r, color=SERIES[3], marker="o", label="RMNP: local row-normalize")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f"{s//1024}k²" for s in sizes])
    ax2.minorticks_off()
    ax2.set_xlabel("square matrix size")
    ax2.set_ylabel("per-matrix precondition (ms, log)")
    ax2.set_ylim(top=max(t_m) * 6)
    gap0 = t_m[0] / t_r[0]
    gap1 = t_m[-1] / t_r[-1]
    ax2.annotate(f"{gap0:.0f}×", (sizes[0], t_m[0] * 2.0), color=TEXT2, fontsize=9, ha="center")
    ax2.annotate(f"{gap1:.0f}×", (sizes[-1], t_m[-1] * 2.0), color=TEXT2, fontsize=9, ha="right")
    ax2.legend(fontsize=9)
    ax2.set_title("(b) Per-matrix cost vs size:\nthe gap itself grows with the matrix", fontsize=11.5)

    save(fig, os.path.join(FIG, "fig-3-measured"))


if __name__ == "__main__":
    main()
