"""第 00 篇(序章)示意图:显存账本 + 5D 并行地图。"""

import os

SURFACE = "#0b0f19"
SURFACE2 = "#131c2e"
TEXT = "#d0dce8"
TEXT2 = "#7a8899"
GLOW = "#00c8ff"
S = ["#0099c4", "#e66767", "#9085e9", "#199e70", "#c98500", "#e14d92"]
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")


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


# ---------------------------------------------------------------- fig 1: 显存账本
def fig1():
    W, H = 960, 420
    s = svg_open(W, H)
    s += text(W / 2, 34, "The memory ledger: training &#8776; storing 8 copies of your model", 17, TEXT, weight="bold")
    s += text(W / 2, 56, "mixed-precision Adam &#183; &#936; = number of parameters &#183; bytes per item shown as &#215;&#936;", 11.5, TEXT2)

    # segments: (label, xPsi, color)
    segs = [
        ("bf16 params", 2, S[0]),
        ("bf16 grads", 2, S[2]),
        ("fp32 master params", 4, S[4]),
        ("Adam m (fp32)", 4, S[4]),
        ("Adam v (fp32)", 4, S[4]),
    ]
    total = sum(x for _, x, _ in segs)  # 16
    x0, y0, bw, bh = 70, 120, 820, 64
    x = x0
    for i, (label, psi, c) in enumerate(segs):
        w = bw * psi / total
        op = [1.0, 1.0, 1.0, 0.75, 0.5][i]
        s += (f'<rect x="{x}" y="{y0}" width="{w - 3}" height="{bh}" rx="6" '
              f'fill="{c}" opacity="{op}"/>\n')
        s += text(x + w / 2 - 1, y0 + bh / 2 - 2, f"{psi}&#936;", 15, "#0b0f19", weight="bold")
        s += text(x + w / 2 - 1, y0 + bh / 2 + 15, label.split(" (")[0], 9.5, "#0b0f19")
        x += w
    # brackets
    s += text(x0 + bw * 2 / 16, y0 - 14, "compute copies (bf16)", 10.5, TEXT2)
    s += text(x0 + bw * 10 / 16, y0 - 14, "optimizer state (fp32) &#8212; 3/4 of the bill", 10.5, TEXT2)
    s += (f'<line x1="{x0}" y1="{y0 + bh + 18}" x2="{x0 + bw}" y2="{y0 + bh + 18}" '
          f'stroke="{GLOW}" stroke-width="1.2" marker-end="url(#arr)" marker-start="url(#arr)"/>\n')
    s += text(x0 + bw / 2, y0 + bh + 36, "16&#936; bytes  (+ activations, batch/seq-dependent)", 12.5, GLOW, weight="bold")

    # example row
    ex_y = y0 + bh + 70
    s += (f'<rect x="{x0}" y="{ex_y}" width="{bw}" height="76" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.2)"/>\n')
    s += text(x0 + 20, ex_y + 26, "example &#8212; 7B model:", 12, TEXT, "start", weight="bold")
    s += text(x0 + 20, ex_y + 48, "checkpoint on disk (bf16):  2&#936; = 14 GB      training state:  16&#936; = 112 GB  &#8594;  does not fit in one 96 GB GPU", 11.5, TEXT2, "start")
    s += text(x0 + 20, ex_y + 66, "and that is before activations &#8212; this wall is why ZeRO/FSDP exist (posts #3-#4)", 11.5, TEXT2, "start")
    s += "</svg>\n"
    return s


# ---------------------------------------------------------------- fig 2: 5D 地图
def fig2():
    W, H = 960, 748
    s = svg_open(W, H)
    s += text(W / 2, 34, "The 5D parallelism map: what gets cut, and what it costs", 17, TEXT, weight="bold")
    s += text(W / 2, 56, "every scheme = cut something &#183; place shards on a GPU group &#183; glue back with a collective", 11.5, TEXT2)

    rows = [
        # (name, cuts, primitive, color, kind)
        ("DP", "Data Parallelism", "the batch &#8212; model is replicated", "all-reduce (grads)", S[0], "dp"),
        ("TP", "Tensor Parallelism", "each weight matrix, row/col-wise", "all-reduce (activations)", S[3], "tp"),
        ("PP", "Pipeline Parallelism", "the layer stack, into stages", "p2p send/recv", S[2], "pp"),
        ("CP", "Context Parallelism", "the sequence dimension", "ring KV exchange", S[4], "cp"),
        ("EP", "Expert Parallelism", "MoE experts across ranks", "all-to-all", S[5], "ep"),
    ]
    y = 84
    for name, full, cuts, prim, c, kind in rows:
        s += (f'<rect x="30" y="{y}" width="900" height="118" rx="10" fill="{SURFACE2}" '
              f'stroke="rgba(0,200,255,0.15)"/>\n')
        s += text(58, y + 44, name, 22, c, "start", weight="bold")
        s += text(58, y + 66, full, 9.5, TEXT2, "start")
        # mini diagram: model = 4x6 grid of cells at x=300..480
        gx, gy, cw, ch = 320, y + 18, 26, 16
        for r in range(4):
            for col in range(6):
                fill = c
                op = 0.85
                if kind == "tp" and col >= 3:
                    op = 0.3
                if kind == "pp" and r >= 2:
                    op = 0.3
                if kind == "ep" and col >= 3:
                    op = 0.3
                s += (f'<rect x="{gx + col * (cw + 2)}" y="{gy + r * (ch + 2)}" width="{cw}" '
                      f'height="{ch}" rx="2" fill="{fill}" opacity="{op if kind not in ("dp", "cp") else 0.85}"/>\n')
        # cut line
        if kind == "tp" or kind == "ep":
            cx = gx + 3 * (cw + 2) - 1
            s += f'<line x1="{cx}" y1="{gy - 4}" x2="{cx}" y2="{gy + 4 * (ch + 2) + 2}" stroke="{GLOW}" stroke-width="2" stroke-dasharray="5,4"/>\n'
        if kind == "pp":
            cy = gy + 2 * (ch + 2) - 1
            s += f'<line x1="{gx - 4}" y1="{cy}" x2="{gx + 6 * (cw + 2) + 2}" y2="{cy}" stroke="{GLOW}" stroke-width="2" stroke-dasharray="5,4"/>\n'
        if kind == "dp":
            # two model copies + different data chips
            s += f'<rect x="{gx - 10}" y="{gy - 6}" width="{6 * (cw + 2) + 16}" height="{4 * (ch + 2) + 10}" rx="6" fill="none" stroke="{TEXT2}" stroke-width="1"/>\n'
            s += text(gx - 4, gy + 4 * (ch + 2) + 20, "&#215;N full replicas, each fed different data", 10, TEXT2, "start")
        if kind == "cp":
            # sequence bar under model
            s += f'<rect x="{gx}" y="{gy + 4 * (ch + 2) + 4}" width="{6 * (cw + 2) - 2}" height="8" rx="2" fill="{GLOW}" opacity="0.5"/>\n'
            cx = gx + 3 * (cw + 2) - 1
            s += f'<line x1="{cx}" y1="{gy + 4 * (ch + 2)}" x2="{cx}" y2="{gy + 4 * (ch + 2) + 16}" stroke="{GLOW}" stroke-width="2" stroke-dasharray="4,3"/>\n'
            s += text(gx - 4, gy + 4 * (ch + 2) + 26, "&#8593; cuts the sequence, not the model", 10, TEXT2, "start")
        # texts
        s += text(575, y + 40, "cuts:", 10.5, TEXT2, "start")
        s += text(575, y + 58, cuts, 12, TEXT, "start")
        s += text(575, y + 82, f"glue: {prim}", 11, c, "start", weight="bold")
        y += 128
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-memory-ledger", fig1), ("fig-2-5d-map", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
