"""第 08 篇实验图:fig-3 三精度训练对照 + swallow 对账,fig-4 GEMM 吞吐 + 梯度直方图。"""

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

C = {"fp32": SERIES[2], "mixed": SERIES[0], "bf16": SERIES[1]}  # 紫 / 青(主角) / 红(警示)
LABEL = {"fp32": "fp32 (baseline)", "mixed": "bf16 mixed (fp32 master)", "bf16": "pure bf16 (no master)"}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def fig3():
    curves = {s: read_csv(os.path.join(RES, f"curve_{s}.csv")) for s in ["fp32", "mixed", "bf16"]}
    lowlr = {s: read_csv(os.path.join(RES, f"curve_{s}_lowlr.csv")) for s in ["fp32", "mixed", "bf16"]}
    swallow = read_csv(os.path.join(RES, "swallow.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    for s, lw, a in [("fp32", 4.5, 0.55), ("mixed", 1.8, 1.0), ("bf16", 1.8, 1.0)]:
        steps = [int(r["step"]) for r in curves[s]]
        tr = [float(r["train_ema"]) for r in curves[s]]
        ax1.plot(steps, tr, color=C[s], label=LABEL[s], lw=lw, alpha=a)
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss (EMA 0.98)")
    ax1.set_title("(a) Full run: all three track\n(lr 1e-3 cosine — disease invisible)", fontsize=11)
    ax1.legend(fontsize=8.5)

    for s, lw, a in [("fp32", 4.5, 0.55), ("mixed", 1.8, 1.0), ("bf16", 1.8, 1.0)]:
        steps = [int(r["step"]) for r in lowlr[s]]
        tr = [float(r["train_ema"]) for r in lowlr[s]]
        ax2.plot(steps, tr, color=C[s], label=LABEL[s], lw=lw, alpha=a)
    ax2.set_xlabel("step")
    ax2.set_ylabel("train loss (EMA 0.98)")
    ax2.set_title("(b) Updates below |w|/512 (lr 3e-5):\npure bf16 stalls, mixed = fp32", fontsize=11)
    ax2.legend(fontsize=8.5)

    st = [int(r["step"]) for r in swallow]
    fr = [100 * float(r["swallow_frac"]) for r in swallow]
    ax3.plot(st, fr, color=SERIES[1], marker="o", ms=3.5, lw=2)
    ax3.set_xlabel("step")
    ax3.set_ylabel("% of updates a bf16 weight would swallow")
    ax3.set_ylim(0, 100)
    ax3.set_title("(c) Swallow audit of run (a): LR decay\ndrives swallowed updates to 87%", fontsize=11)
    ax3.annotate("cosine decay →\nsmaller updates →\nmore swallowed", xy=(st[-8], fr[-8]),
                 xytext=(0.35, 0.18), textcoords="axes fraction", color=TEXT2, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
    save(fig, os.path.join(FIG, "fig-3-three-precisions"))


def fig4():
    gemm = read_csv(os.path.join(RES, "gemm.csv"))
    hist = read_csv(os.path.join(RES, "gradhist.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    names = [r["dtype"] for r in gemm]
    tf = [float(r["tflops"]) for r in gemm]
    colors = [SERIES[0] if n == "bf16" else SERIES[0] for n in names]
    alphas = [1.0 if n == "bf16" else 0.5 for n in names]
    bars = ax1.bar(names, tf, color=colors, width=0.62)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    for b, v in zip(bars, tf):
        ax1.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                     va="bottom", fontsize=10, color=TEXT)
    ax1.set_ylabel("TFLOPS (8192³ GEMM, measured)")
    ax1.set_title("The same matmul, 5× faster in bf16\n(8192³ GEMM, RTX PRO 6000)", fontsize=12)
    ax1.grid(axis="x", visible=False)

    lo = np.array([float(r["log10_bin_lo"]) for r in hist])
    cnt = np.array([float(r["count"]) for r in hist])
    binw = lo[1] - lo[0]
    pct = 100 * cnt / cnt.sum()
    ax2.bar(lo + binw / 2, pct, width=binw * 0.92, color=SERIES[0], alpha=0.85)
    fp16_min = np.log10(6.1e-5)
    fp16_sub = np.log10(5.96e-8)
    below = 100 * cnt[lo + binw <= fp16_min].sum() / cnt.sum()
    ax2.axvline(fp16_min, color=SERIES[1], ls="--", lw=1.8)
    ax2.axvline(fp16_sub, color=SERIES[1], ls=":", lw=1.4)
    ymax = pct.max()
    ax2.annotate(f"fp16 min normal 6.1e-5\n{below:.0f}% of grad entries\nsit below this line", xy=(fp16_min, ymax * 0.72),
                 xytext=(fp16_min + 1.2, ymax * 0.72), color=SERIES[1], fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1))
    ax2.annotate("fp16 subnormal floor 6e-8\n(flushed to 0)", xy=(fp16_sub, ymax * 0.35),
                 xytext=(fp16_sub + 1.2, ymax * 0.38), color=SERIES[1], fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1))
    ax2.text(0.02, 0.60, "bf16/fp32 min normal: 1.2e-38\n— 26 decades left of this chart:\nnothing here ever underflows",
             transform=ax2.transAxes, color=SERIES[3], fontsize=9, va="top")
    ax2.set_xlabel("log10 |gradient entry|  (all params, step 500, mixed run)")
    ax2.set_ylabel("% of entries")
    ax2.set_title("Real gradients vs fp16's floor:\nthe gap loss scaling papers over", fontsize=12)
    save(fig, os.path.join(FIG, "fig-4-gemm-and-gradhist"))


if __name__ == "__main__":
    apply()
    fig3()
    fig4()
