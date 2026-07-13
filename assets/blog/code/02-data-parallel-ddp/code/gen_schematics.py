"""Schematics for post 02: DP structure plus the DDP bucketing/overlap timeline."""

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


def arrow(x1, y1, x2, y2, color=GLOW, width=1.6, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ""
    halo = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width * 3}" opacity="0.22" stroke-linecap="round"/>\n')
    return halo + (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                   f'stroke-width="{width}" marker-end="url(#arr)"{d}/>\n')


# ------------------------------------------------ fig 1: one DP step
def fig1():
    W, H = 960, 500
    s = svg_open(W, H)
    s += text(W / 2, 32, "One step of data parallelism (N = 4)", 17, TEXT, weight="bold")
    s += text(W / 2, 54, "replicas start identical &#183; see different data &#183; one all-reduce makes them identical again", 11.5, TEXT2)

    bw, bh = 190, 320
    for k in range(4):
        x = 40 + k * 230
        s += (f'<rect x="{x}" y="80" width="{bw}" height="{bh}" rx="10" fill="{SURFACE2}" '
              f'stroke="rgba(0,200,255,0.35)" stroke-width="1.3"/>\n')
        s += text(x + 12, 102, f"GPU {k}", 11, TEXT2, "start")
        # data shard
        s += f'<rect x="{x + 24}" y="118" width="{bw - 48}" height="26" rx="4" fill="{S[k]}" opacity="0.9"/>\n'
        s += text(x + bw / 2, 135, f"batch shard {k}", 10.5, "#0b0f19", weight="bold")
        s += arrow(x + bw / 2, 152, x + bw / 2, 172)
        # model replica
        s += f'<rect x="{x + 24}" y="176" width="{bw - 48}" height="46" rx="6" fill="none" stroke="{S[0]}" stroke-width="1.6"/>\n'
        s += text(x + bw / 2, 196, "model replica", 11, TEXT)
        s += text(x + bw / 2, 212, "(identical &#952;)", 9.5, TEXT2)
        s += arrow(x + bw / 2, 230, x + bw / 2, 250)
        # local grad
        s += f'<rect x="{x + 24}" y="254" width="{bw - 48}" height="30" rx="4" fill="{S[2]}" opacity="0.85"/>\n'
        s += text(x + bw / 2, 273, f"local grad g{k}", 10.5, "#0b0f19", weight="bold")
        s += text(x + bw / 2, 306, "forward + backward", 9, TEXT2)
        s += text(x + bw / 2, 320, "(no communication)", 9, TEXT2)
        s += arrow(x + bw / 2, 330, x + bw / 2, 352)
    # all-reduce bar
    s += f'<rect x="40" y="356" width="{3 * 230 + bw - 0}" height="40" rx="8" fill="none" stroke="{GLOW}" stroke-width="2"/>\n'
    s += text(W / 2 - 20, 381, "all-reduce:  every GPU gets  g = (g0+g1+g2+g3) / N", 13, GLOW, weight="bold")
    s += text(W / 2, 428, "then each replica applies the SAME update  &#952; &#8592; &#952; &#8722; &#951;&#183;g   &#8594;   replicas stay bit-identical, forever", 12, TEXT)
    s += text(W / 2, 470, "the invariant: DP never approximates anything &#8212; it computes exactly what one big-batch GPU would compute", 10.5, TEXT2)
    s += "</svg>\n"
    return s


# ------------------------------------------------ fig 2: bucketing + overlap timeline
def fig2():
    W, H = 960, 560
    s = svg_open(W, H)
    s += text(W / 2, 32, "DDP's two tricks: bucketing + overlapping comm with backward", 17, TEXT, weight="bold")
    s += text(W / 2, 54, "backward computes gradients last-layer-first &#183; a bucket fires its all-reduce the moment it fills", 11.5, TEXT2)

    x0, lane_w = 190, 700
    t_unit = lane_w / 10.0  # 10 time units

    def lane(y, label, sub=""):
        out = text(x0 - 14, y + 22, label, 11, TEXT, "end")
        if sub:
            out += text(x0 - 14, y + 38, sub, 9, TEXT2, "end")
        out += f'<line x1="{x0}" y1="{y + 46}" x2="{x0 + lane_w}" y2="{y + 46}" stroke="rgba(0,200,255,0.15)"/>\n'
        return out

    def block(x, y, w, color, label="", op=0.9, h=34):
        out = f'<rect x="{x0 + x * t_unit}" y="{y}" width="{w * t_unit - 3}" height="{h}" rx="4" fill="{color}" opacity="{op}"/>\n'
        if label:
            out += text(x0 + (x + w / 2) * t_unit, y + h / 2 + 4, label, 10, "#0b0f19", weight="bold")
        return out

    # ---- panel A: no overlap (whole model in one bucket)
    s += text(x0 + lane_w / 2, 92, "(a) one giant bucket = no overlap: comm waits for the whole backward", 12, S[1], weight="bold")
    yA = 108
    s += lane(yA, "compute", "stream")
    for i, lab in enumerate(["bwd L4", "bwd L3", "bwd L2", "bwd L1"]):
        s += block(i * 1.5, yA, 1.5, S[0], lab)
    s += lane(yA + 62, "NCCL", "stream")
    s += block(6.0, yA + 62, 3.2, S[2], "all-reduce (entire model)")
    s += text(x0 + 9.2 * t_unit + 8, yA + 62 + 22, "&#8592; fully exposed", 10, S[1], "start", weight="bold")
    # step time bracket
    s += f'<line x1="{x0}" y1="{yA + 118}" x2="{x0 + 9.2 * t_unit}" y2="{yA + 118}" stroke="{TEXT2}" stroke-width="1" marker-end="url(#arr)"/>\n'
    s += text(x0 + 4.6 * t_unit, yA + 134, "step time = backward + FULL all-reduce", 10, TEXT2)

    # ---- panel B: bucketed overlap
    s += text(x0 + lane_w / 2, 300, "(b) buckets: gradients for later layers fly while earlier layers still compute", 12, S[3], weight="bold")
    yB = 316
    s += lane(yB, "compute", "stream")
    for i, lab in enumerate(["bwd L4", "bwd L3", "bwd L2", "bwd L1"]):
        s += block(i * 1.5, yB, 1.5, S[0], lab)
    s += lane(yB + 62, "NCCL", "stream")
    for i, lab in enumerate(["AR bkt1", "AR bkt2", "AR bkt3", "AR bkt4"]):
        s += block(1.5 + i * 1.5, yB + 62, 1.4, S[2], lab)
    s += text(x0 + 7.6 * t_unit + 6, yB + 62 + 22, "&#8592; only the tail is exposed", 10, S[3], "start", weight="bold")
    s += f'<line x1="{x0}" y1="{yB + 118}" x2="{x0 + 7.5 * t_unit}" y2="{yB + 118}" stroke="{TEXT2}" stroke-width="1" marker-end="url(#arr)"/>\n'
    s += text(x0 + 3.75 * t_unit, yB + 134, "step time = backward + one bucket's tail", 10, TEXT2)

    # tradeoff note
    s += (f'<rect x="{x0}" y="{yB + 160}" width="{lane_w}" height="54" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.2)"/>\n')
    s += text(x0 + lane_w / 2, yB + 182, "bucket too small &#8594; pays the latency floor per bucket (post #1) &#183; too big &#8594; nothing left to overlap", 10.5, TEXT)
    s += text(x0 + lane_w / 2, yB + 200, "the sweet spot is measured, not guessed &#8594; see the bucket_cap_mb sweep below", 10.5, GLOW)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-dp-one-step", fig1), ("fig-2-bucket-overlap", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
