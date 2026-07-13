"""第 03 篇示意图:ZeRO 劈开 all-reduce + 三级显存账本。"""

import os

SURFACE = "#0b0f19"
SURFACE2 = "#131c2e"
TEXT = "#d0dce8"
TEXT2 = "#7a8899"
GLOW = "#00c8ff"
S = ["#0099c4", "#e66767", "#9085e9", "#199e70", "#c98500", "#e14d92"]
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
# 显存分段惯用色(style-guide):参数=s1 梯度=s3 优化器状态=s5
C_P, C_G, C_O = S[0], S[2], S[4]


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{MONO}">\n'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{GLOW}"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n')


def text(x, y, s, size=13, fill=TEXT, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>\n')


def arrow(x1, y1, x2, y2, color=GLOW, width=1.6):
    halo = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width * 3}" opacity="0.22" stroke-linecap="round"/>\n')
    return halo + (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                   f'stroke-width="{width}" marker-end="url(#arr)"/>\n')


# ---------------- fig 1: 劈开 all-reduce,中间插本地更新(ZeRO-1 心脏)
def fig1():
    W, H = 960, 470
    s = svg_open(W, H)
    s += text(W / 2, 32, "ZeRO's key move: split the all-reduce open, update in the middle", 17, TEXT, weight="bold")
    s += text(W / 2, 54, "post #1: all-reduce = reduce-scatter + all-gather &#183; after RS, rank k already owns the FULL sum of shard k", 11.5, TEXT2)

    # DDP path (top)
    y1 = 92
    s += text(80, y1 + 8, "DDP", 13, S[1], "start", weight="bold")
    boxes = [("backward", S[0], 130), ("all-reduce (grads)", S[2], 190), ("EVERY rank updates ALL of &#952;", S[1], 250), ("needs full optimizer state &#215;N", None, 0)]
    x = 200
    s += f'<rect x="{x}" y="{y1 - 18}" width="130" height="40" rx="6" fill="{S[0]}" opacity="0.9"/>\n' + text(x + 65, y1 + 6, "backward", 11, "#0b0f19", weight="bold")
    s += arrow(x + 134, y1 + 2, x + 164, y1 + 2)
    x = 368
    s += f'<rect x="{x}" y="{y1 - 18}" width="190" height="40" rx="6" fill="{S[2]}" opacity="0.9"/>\n' + text(x + 95, y1 + 6, "all-reduce(g)  2(N-1)/N&#183;S", 10, "#0b0f19", weight="bold")
    s += arrow(x + 194, y1 + 2, x + 224, y1 + 2)
    x = 596
    s += f'<rect x="{x}" y="{y1 - 18}" width="240" height="40" rx="6" fill="none" stroke="{S[1]}" stroke-width="1.6"/>\n'
    s += text(x + 120, y1 - 1, "every rank updates ALL &#952;", 10.5, S[1], weight="bold")
    s += text(x + 120, y1 + 14, "&#8594; full fp32 state on every rank", 9, TEXT2)

    # ZeRO path (bottom)
    y2 = 190
    s += text(80, y2 + 8, "ZeRO-1", 13, S[3], "start", weight="bold")
    x = 200
    s += f'<rect x="{x}" y="{y2 - 18}" width="130" height="40" rx="6" fill="{S[0]}" opacity="0.9"/>\n' + text(x + 65, y2 + 6, "backward", 11, "#0b0f19", weight="bold")
    s += arrow(x + 134, y2 + 2, x + 164, y2 + 2)
    x = 368
    s += f'<rect x="{x}" y="{y2 - 18}" width="150" height="40" rx="6" fill="{S[2]}" opacity="0.9"/>\n' + text(x + 75, y2 + 6, "reduce-scatter", 10.5, "#0b0f19", weight="bold")
    s += arrow(x + 154, y2 + 2, x + 184, y2 + 2)
    x = 556
    s += f'<rect x="{x}" y="{y2 - 18}" width="180" height="40" rx="6" fill="{S[3]}" opacity="0.95"/>\n'
    s += text(x + 90, y2 - 1, "LOCAL update of shard k", 10, "#0b0f19", weight="bold")
    s += text(x + 90, y2 + 13, "only 1/N of fp32 state", 9, "#0b0f19")
    s += arrow(x + 184, y2 + 2, x + 214, y2 + 2)
    x = 774
    s += f'<rect x="{x}" y="{y2 - 18}" width="150" height="40" rx="6" fill="{S[2]}" opacity="0.9"/>\n' + text(x + 75, y2 + 6, "all-gather(&#952;')", 10.5, "#0b0f19", weight="bold")

    # equality note
    s += (f'<rect x="120" y="264" width="720" height="66" rx="8" fill="{SURFACE2}" stroke="rgba(0,200,255,0.25)"/>\n')
    s += text(480, 288, "same collectives, rearranged:  RS + AG = all-reduce  &#8594;  communication volume unchanged (2(N-1)/N &#183; S)", 11.5, TEXT)
    s += text(480, 310, "but each rank now stores only 1/N of the optimizer state  &#8594;  12&#936; becomes 12&#936;/N", 12, GLOW, weight="bold")

    # shard ownership strip
    y3 = 370
    s += text(140, y3 + 24, "who updates what:", 11, TEXT2, "end")
    for k in range(4):
        x = 160 + k * 190
        s += f'<rect x="{x}" y="{y3}" width="180" height="38" rx="6" fill="{SURFACE2}" stroke="rgba(0,200,255,0.3)"/>\n'
        s += text(x + 12, y3 + 24, f"G{k}", 11, TEXT2, "start")
        for j in range(4):
            fill_op = "0.95" if j == k else "0.18"
            s += f'<rect x="{x + 44 + j * 32} " y="{y3 + 9}" width="26" height="20" rx="3" fill="{C_O}" opacity="{fill_op}"/>\n'
    s += text(480, y3 + 62, "bright = the optimizer-state shard this rank owns, stores, and updates &#183; dim = never materialized here", 10, TEXT2)
    s += "</svg>\n"
    return s


# ---------------- fig 2: 三级账本(理论条形,Ψ 记账)
def fig2():
    W, H = 960, 560
    s = svg_open(W, H)
    s += text(W / 2, 32, "The three-stage ledger (per GPU, N = 8, GPT-2 Large &#936; = 0.77B)", 17, TEXT, weight="bold")
    s += text(W / 2, 54, "params (bf16, 2&#936;) &#183; grads (bf16, 2&#936;) &#183; optimizer state (fp32 master + Adam m,v = 12&#936;) &#183; dim = sharded away /8", 11, TEXT2)

    # (label, p, g, o) in units of Ψ actually held per GPU
    N = 8
    rows = [
        ("stage 0 (= DDP)", 2, 2, 12, "16&#936; &#8776; 12.3 GB"),
        ("stage 1: shard optim", 2, 2, 12 / N, "5.5&#936; &#8776; 4.3 GB"),
        ("stage 2: + shard grads", 2, 2 / N, 12 / N, "3.75&#936; &#8776; 2.9 GB"),
        ("stage 3: + shard params", 2 / N, 2 / N, 12 / N, "2&#936; &#8776; 1.5 GB"),
    ]
    x0, bw_full = 235, 570
    unit = bw_full / 16.0
    y = 96
    for label, p, g, o, tot in rows:
        s += text(x0 - 14, y + 25, label, 12, TEXT, "end")
        x = x0
        for val, full_val, c in [(p, 2, C_P), (g, 2, C_G), (o, 12, C_O)]:
            # full extent (dashed ghost)
            s += (f'<rect x="{x}" y="{y}" width="{full_val * unit - 3}" height="38" rx="5" '
                  f'fill="{c}" opacity="0.13"/>\n')
            s += (f'<rect x="{x}" y="{y}" width="{full_val * unit - 3}" height="38" rx="5" '
                  f'fill="none" stroke="{c}" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>\n')
            # actually held
            s += f'<rect x="{x}" y="{y}" width="{max(val * unit - 3, 6)}" height="38" rx="5" fill="{c}" opacity="0.95"/>\n'
            x += full_val * unit
        s += text(x0 + 16 * unit + 14, y + 25, tot, 11.5, GLOW, "start", weight="bold")
        y += 74
    # legend
    ly = y + 6
    for c, lab, dx in [(C_P, "params", 0), (C_G, "grads", 130), (C_O, "optimizer state", 260)]:
        s += f'<rect x="{x0 + dx}" y="{ly}" width="16" height="16" rx="3" fill="{c}"/>\n'
        s += text(x0 + dx + 24, ly + 13, lab, 10.5, TEXT2, "start")
    s += text(W / 2, ly + 52, "solid = held on this GPU &#183; dashed ghost = what DDP would hold &#183; stage 3&#8217;s price: params must be re-all-gathered each fwd/bwd", 10.5, TEXT2)
    s += text(W / 2, ly + 76, "this is the paper&#8217;s ledger &#183; &#167;4 measures what DeepSpeed actually holds &#8212; and where they differ, the difference teaches", 11, GLOW)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-split-allreduce", fig1), ("fig-2-three-stage-ledger", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
