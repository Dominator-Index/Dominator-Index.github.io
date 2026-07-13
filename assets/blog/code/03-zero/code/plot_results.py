"""第 03 篇实验图。数据:../results/zero.csv + 单卡基线 152.9 ms(single_large_baseline.py)"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from plot_style import apply, save, SERIES, GLOW, TEXT2, TEXT  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
rows = list(csv.DictReader(open(HERE.parent / "results" / "zero.csv")))
PSI_GIB = 774090240 / 2**30  # 1Ψ 字节数换算成 GiB 的系数(×每参数字节数)

apply()

# ---- fig 3: 每卡显存(常驻 + 峰值) ---------------------------------------
stages = [r["stage"] for r in rows]
resident = [float(r["resident_GiB"]) for r in rows]
peak = [float(r["peak_GiB"]) for r in rows]
# 论文账本预测的"step 后常驻"(梯度不驻留口径): 14Ψ, 3.5Ψ, 3.5Ψ, 1.75Ψ
ledger = [14 * PSI_GIB, 3.5 * PSI_GIB, 3.5 * PSI_GIB, 1.75 * PSI_GIB]

fig, ax = plt.subplots(figsize=(8.6, 4.8))
x = range(4)
ax.bar([i - 0.19 for i in x], resident, width=0.36, color=SERIES[0], label="resident after step (params+optim state)")
ax.bar([i + 0.19 for i in x], peak, width=0.36, color=SERIES[2], alpha=0.75, label="peak during step (+activations, comm buffers)")
for i in x:
    ax.hlines(ledger[i], i - 0.38, i + 0.0, color=GLOW, ls="--", lw=1.6)
    ax.text(i - 0.19, resident[i] + 0.35, f"{resident[i]:.1f}", ha="center", fontsize=10.5, color=SERIES[0], fontweight="bold")
    ax.text(i + 0.19, peak[i] + 0.35, f"{peak[i]:.1f}", ha="center", fontsize=10.5, color=SERIES[2], fontweight="bold")
ax.plot([], [], color=GLOW, ls="--", label="ledger prediction (grads transient)")
ax.set_xticks(list(x))
ax.set_xticklabels(["stage 0\n(no ZeRO)", "stage 1\nshard optim", "stage 2\n+ shard grads", "stage 3\n+ shard params"], fontsize=9.5)
ax.set_ylabel("per-GPU memory (GiB)")
ax.set_ylim(0, 26)
ax.set_title("ZeRO's memory ladder, measured (GPT-2 Large, 8 GPUs, DeepSpeed)")
ax.annotate("stage-1 drop: 7.8 GiB\n= 12Ψ·7/8 = 7.6 GiB predicted (3%)",
            xy=(0.81, 2.68), xytext=(0.62, 12.5), fontsize=9.5, color=TEXT,
            arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.annotate("stage 1 = stage 2:\ngrads never persist anyway",
            xy=(2 - 0.19, 2.68), xytext=(1.75, 7.5), fontsize=9.5, color=TEXT,
            arrowprops=dict(arrowstyle="->", color=TEXT2, lw=1))
ax.legend(loc="upper right", fontsize=9)
save(fig, FIG / "fig-3-memory-ladder")

# ---- fig 4: step 时间与暴露的通信 -----------------------------------------
single_ms = 152.9
step_ms = [float(r["step_ms"]) for r in rows]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
colors = [SERIES[1], SERIES[4], SERIES[2], SERIES[3]]
ax.bar(range(4), step_ms, width=0.56, color=colors, alpha=0.92)
ax.axhline(single_ms, color=TEXT2, ls=":", lw=1.5)
ax.text(-0.42, single_ms - 26, f"single-GPU compute ({single_ms:.0f} ms)", fontsize=9.5, color=TEXT2)
for i, v in enumerate(step_ms):
    ax.text(i, v + 6, f"{v:.0f} ms", ha="center", fontsize=10.5, color=colors[i], fontweight="bold")
    ax.text(i, v - 24, f"+{v - single_ms:.0f} exposed", ha="center", fontsize=8.5, color="#0b0f19", fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(["stage 0", "stage 1", "stage 2", "stage 3"])
ax.set_ylabel("time per step (ms)")
ax.set_ylim(0, 430)
ax.set_title("Step time: sharding is not slower — stage 3's prefetch even overlaps best")
save(fig, FIG / "fig-4-step-time")

print("done")
