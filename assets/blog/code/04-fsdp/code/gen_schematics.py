"""第 04 篇示意图:FSDP1 vs FSDP2 切分几何 + FSDP2 prefetch 时间线。"""

import os

SURFACE = "#0b0f19"
SURFACE2 = "#131c2e"
TEXT = "#d0dce8"
TEXT2 = "#7a8899"
GLOW = "#00c8ff"
S = ["#0099c4", "#e66767", "#9085e9", "#199e70", "#c98500", "#e14d92"]
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
RANK_C = [S[0], S[3], S[4], S[5]]  # 4 个 rank 的归属色


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


# ------------- fig 1: 切分几何 --------------------------------------------
def fig1():
    W, H = 960, 560
    s = svg_open(W, H)
    s += text(W / 2, 32, "Same idea, different geometry: how FSDP1 and FSDP2 cut a weight matrix", 16.5, TEXT, weight="bold")
    s += text(W / 2, 54, "one [8&#215;12] weight W, sharded across 4 ranks &#183; color = which rank stores it", 11.5, TEXT2)

    cw, ch = 34, 22  # cell size

    # --- FSDP1: flatten + equal split by elements
    y0 = 96
    s += text(60, y0 + 8, "FSDP1", 14, S[2], "start", weight="bold")
    s += text(60, y0 + 26, "FlatParameter", 9.5, TEXT2, "start")
    # flattened bar: 8*12=96 elements -> bar of 96 units, split every 24
    bx, bw_, bh_ = 220, 660, 30
    n_el = 96
    for k in range(4):
        x = bx + k * (bw_ / 4)
        s += f'<rect x="{x}" y="{y0}" width="{bw_ / 4 - 2}" height="{bh_}" rx="4" fill="{RANK_C[k]}" opacity="0.9"/>\n'
        s += text(x + bw_ / 8, y0 + bh_ / 2 + 4, f"rank {k}: elements {k * 24}&#8211;{k * 24 + 23}", 9.5, "#0b0f19", weight="bold")
    s += text(bx + bw_ / 2, y0 - 10, "W.flatten()  (row-major: row 0, then row 1, ...)", 9.5, TEXT2)
    # matrix under it showing row ownership with mid-row cuts
    my = y0 + 56
    s += text(bx - 14, my + 50, "the same cut,\nseen on the matrix:", 9.5, TEXT2, "end")
    for r in range(8):
        for c in range(12):
            el = r * 12 + c
            k = el // 24
            s += (f'<rect x="{bx + c * (cw - 6)}" y="{my + r * (ch - 8)}" width="{cw - 8}" '
                  f'height="{ch - 10}" rx="2" fill="{RANK_C[k]}" opacity="0.85"/>\n')
    # highlight broken row: row 2 belongs to rank0 (el 24..35 -> row2 = el 24..35 all rank1? row2: 24-35 -> 24//24=1.. all rank1). Actually rows 0-1 rank0, row2-3 rank1... 24 el = 2 rows exactly!
    # 8 rows x 12 = 96, 96/4 = 24 = exactly 2 rows -> boundary aligns. Use 4 ranks over 96 with offset: make it misalign by using 5-rank? Instead use uneven: shard size = ceil for first ranks in FSDP1 is per flat buffer including OTHER params -> boundaries generally not aligned. Simulate by shifting cut by half a row: draw cut lines at elements 18, 42, 66 to represent multi-param flat buffer.
    s += text(bx + 330, my + 8 * (ch - 8) + 18, "in a real FlatParameter the buffer holds MANY params back-to-back,", 9.5, S[1])
    s += text(bx + 330, my + 8 * (ch - 8) + 34, "so cut points land mid-row / mid-tensor &#8212; a shard is just a byte range", 9.5, S[1])

    # --- FSDP2: per-parameter DTensor, dim-0
    y1 = 336
    s += text(60, y1 + 8, "FSDP2", 14, S[3], "start", weight="bold")
    s += text(60, y1 + 26, "per-param DTensor", 9.5, TEXT2, "start")
    s += text(60, y1 + 40, "Shard(dim=0)", 9.5, TEXT2, "start")
    for r in range(8):
        k = r // 2
        for c in range(12):
            s += (f'<rect x="{bx + c * (cw - 6)}" y="{y1 + r * (ch - 8)}" width="{cw - 8}" '
                  f'height="{ch - 10}" rx="2" fill="{RANK_C[k]}" opacity="0.85"/>\n')
    for k in range(4):
        s += text(bx + 12 * (cw - 6) + 16, y1 + (2 * k + 1) * (ch - 8), f"rank {k}: rows {2 * k}&#8211;{2 * k + 1}", 10, RANK_C[k], "start", weight="bold")
    s += text(bx + 170, y1 + 8 * (ch - 8) + 22, "every rank holds COMPLETE rows &#8212; a shard is itself a small [2&#215;12] matrix", 10.5, S[3])
    s += text(W / 2, H - 14, "keep &#8220;rows stay whole&#8221; in mind &#8212; it looks like a detail today and becomes the whole story in post #9", 10.5, GLOW)
    s += "</svg>\n"
    return s


# ------------- fig 2: FSDP2 一步的通信时间线 --------------------------------
def fig2():
    W, H = 960, 500
    s = svg_open(W, H)
    s += text(W / 2, 30, "One FSDP2 step: gather on demand, prefetch ahead, reshard behind", 16.5, TEXT, weight="bold")
    s += text(W / 2, 52, "unit = one transformer block &#183; AG = all-gather params (bf16) &#183; RS = reduce-scatter grads (fp32)", 11, TEXT2)

    x0, lane_w = 190, 720
    t_unit = lane_w / 12.0

    def lane(y, label, sub=""):
        out = text(x0 - 14, y + 20, label, 11, TEXT, "end")
        if sub:
            out += text(x0 - 14, y + 36, sub, 9, TEXT2, "end")
        out += f'<line x1="{x0}" y1="{y + 44}" x2="{x0 + lane_w}" y2="{y + 44}" stroke="rgba(0,200,255,0.15)"/>\n'
        return out

    def block(x, y, w, color, label="", h=32, op=0.9, outline=False):
        style = (f'fill="none" stroke="{color}" stroke-width="1.4"' if outline
                 else f'fill="{color}" opacity="{op}"')
        out = f'<rect x="{x0 + x * t_unit}" y="{y}" width="{w * t_unit - 3}" height="{h}" rx="4" {style}/>\n'
        if label:
            out += text(x0 + (x + w / 2) * t_unit, y + h / 2 + 3.5, label, 9.5,
                        color if outline else "#0b0f19", weight="bold")
        return out

    # forward
    yF = 86
    s += text(x0 + lane_w / 2, yF - 6, "forward", 11.5, S[0], weight="bold")
    s += lane(yF, "compute")
    for i in range(4):
        s += block(1.2 + i * 2.0, yF, 1.9, S[0], f"fwd blk{i + 1}")
    s += lane(yF + 56, "NCCL", "stream")
    s += block(0.0, yF + 56, 1.1, S[3], "AG1")
    for i in range(3):
        s += block(1.2 + i * 2.0, yF + 56, 1.1, S[3], f"AG{i + 2}")
    s += text(x0 + 2.0 * t_unit, yF + 118, "prefetch: AG of block k+1 runs UNDER fwd of block k &#8594; only AG1 is exposed", 9.5, TEXT2, "start")
    s += text(x0 + 2.0 * t_unit, yF + 134, "red tick = reshard: free the gathered full params (ZeRO-3 semantics)", 9.5, S[1], "start")
    # reshard markers
    for i in range(3):
        x = x0 + (3.1 + i * 2.0) * t_unit
        s += f'<line x1="{x}" y1="{yF + 34}" x2="{x}" y2="{yF + 50}" stroke="{S[1]}" stroke-width="2"/>\n'

    # backward
    yB = 250
    s += text(x0 + lane_w / 2, yB - 6, "backward (reverse order)", 11.5, S[2], weight="bold")
    s += lane(yB, "compute")
    for i in range(4):
        s += block(1.2 + i * 2.2, yB, 2.1, S[0], f"bwd blk{4 - i}")
    s += lane(yB + 56, "NCCL", "stream")
    s += block(0.0, yB + 56, 1.1, S[3], "AG4")
    for i in range(3):
        s += block(1.2 + i * 2.2, yB + 56, 1.1, S[3], f"AG{3 - i}")
    for i in range(4):
        s += block(2.4 + i * 2.2, yB + 56, 1.0, S[2], f"RS{4 - i}")
    s += text(x0 + 1.2 * t_unit, yB + 118, "backward needs the params AGAIN (they were resharded) &#8594; re-AG each block, then RS its grads", 9.5, TEXT2, "start")

    # note box
    s += (f'<rect x="{x0}" y="{yB + 148}" width="{lane_w}" height="66" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.2)"/>\n')
    s += text(x0 + lane_w / 2, yB + 172, "reshard_after_forward=False (ZeRO-2 semantics): skip the red ticks, keep full params until backward", 10.5, TEXT)
    s += text(x0 + lane_w / 2, yB + 192, "&#8594; backward re-AG disappears, memory floor rises by one full bf16 model &#8212; measured in &#167;4", 10.5, GLOW)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-cut-geometry", fig1), ("fig-2-prefetch-timeline", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
