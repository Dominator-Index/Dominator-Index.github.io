"""
Schematic generator for post 01 (programmatic SVG, keeps the whole series visually consistent).

Generates:
  figures/fig-1-six-primitives.svg   overview of the six collective primitives
  figures/fig-2-ring-allreduce.svg   ring all-reduce step-by-step state (N=4)
  figures/fig-3-topology.svg         local PCIe topology (8 GPUs, PIX pairs + dual NUMA)

Colors follow _shared/style-guide.md. In-figure text is English (shared by the CN and EN versions).
"""

import math
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
            f'<defs>\n'
            f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{GLOW}"/></marker>\n'
            f'</defs>\n'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n')


def per_shard_markers():
    """Extra <marker> defs, one per shard color, so ring-diagram arrows can be color-coded."""
    out = "<defs>\n"
    for i, c in enumerate(S):
        out += (f'<marker id="arr{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>\n')
    out += "</defs>\n"
    return out


def text(x, y, s, size=13, fill=TEXT, anchor="middle", weight="normal", family=None):
    f = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{f}>{s}</text>\n')


def gpu_box(x, y, w=96, h=58, label="GPU 0"):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{SURFACE2}" stroke="{GLOW}" stroke-width="1.3"/>\n'
            + text(x + 8, y + 15, label, 10, TEXT2, "start"))


def chip(x, y, color, w=18, h=18, sigma=False, dim=False):
    op = ' opacity="0.28"' if dim else ""
    stroke = f' stroke="{TEXT}" stroke-width="1.4"' if sigma else ""
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{color}"{op}{stroke}/>\n'
    if sigma:
        s += text(x + w / 2, y + h / 2 + 4.5, "&#931;", 12, "#0b0f19", weight="bold")
    return s


def arrow(x1, y1, x2, y2, color=GLOW, marker="arr", dash=False, width=1.6):
    d = ' stroke-dasharray="5,4"' if dash else ""
    halo = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width * 3}" opacity="0.22" stroke-linecap="round"/>\n')
    return halo + (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                   f'stroke-width="{width}" marker-end="url(#{marker})"{d}/>\n')


def inline_row(cx, cy, items):
    """A horizontal sequence of chip/text tokens, each with an explicit width, centered
    as a group at (cx, cy). Used for the worked "own + arriving = summed" equation."""
    total = sum(it["w"] for it in items)
    x = cx - total / 2
    out = ""
    for it in items:
        w = it["w"]
        if it["kind"] == "chip":
            size = it.get("size", 11)
            cx0, cy0 = x + w / 2 - size / 2, cy - size / 2
            extra = f' stroke="{it["color"]}" stroke-width="{it.get("sw", 1.4)}"' if it.get("ring") else ""
            out += (f'<rect x="{cx0:.1f}" y="{cy0:.1f}" width="{size}" height="{size}" rx="2.2" '
                    f'fill="{it["color"]}" opacity="{it.get("op", 1.0)}"{extra}/>\n')
            if it.get("glyph"):
                out += text(cx0 + size / 2, cy0 + size / 2 + 3, it["glyph"], it.get("gsize", 7.5), "#0b0f19", weight="bold")
        else:  # text
            out += text(x + w / 2, cy + it.get("dy", 3.5), it["s"], it.get("size", 10.5),
                        it.get("fill", TEXT2), weight=it.get("weight", "normal"))
        x += w
    return out


def polar(ccx, ccy, r, deg):
    """Point at angle `deg` (0=+x axis, increasing = clockwise on screen) on a circle."""
    rad = math.radians(deg)
    return ccx + r * math.cos(rad), ccy + r * math.sin(rad)


def flying_chip(cx, cy, color, idx, size=15):
    """A small in-flight shard marker: color-matched square with its index, for arrow midpoints."""
    half = size / 2
    out = (f'<rect x="{cx - half - 2}" y="{cy - half - 2}" width="{size + 4}" height="{size + 4}" '
           f'rx="4" fill="{SURFACE}"/>\n')
    out += f'<rect x="{cx - half}" y="{cy - half}" width="{size}" height="{size}" rx="3" fill="{color}"/>\n'
    out += text(cx, cy + 3.8, str(idx), 10, "#0b0f19", weight="bold")
    return out


# ---------------------------------------------------------------- fig 1
def fig1():
    """Six-primitive overview: each panel shows before on top, after below, 4 ranks."""
    W, H = 960, 742
    s = svg_open(W, H)
    s += text(W / 2, 30, "The six collective primitives (N = 4 GPUs)", 17, TEXT, weight="bold")
    s += text(W / 2, 50, "colored chip = one shard (S/N bytes) &#183; &#931; = reduced (summed) shard &#183; wide bar = full tensor (S bytes)", 11.5, TEXT2)

    # panel geometry
    PW, PH = 452, 204
    cols_x = [18, 490]
    rows_y = [70, 292, 514]

    def rank_row(px, py, contents):
        """contents: list of 4 lists of (color, sigma, dim) or ('bar', color)"""
        out = ""
        for k in range(4):
            bx = px + k * (96 + 14)
            out += gpu_box(bx, py, label=f"G{k}")
            item = contents[k]
            if item and item[0] == "bar":
                out += (f'<rect x="{bx + 8}" y="{py + 26}" width="80" height="20" rx="3" '
                        f'fill="{item[1]}"/>\n')
                if len(item) > 2 and item[2]:
                    out += text(bx + 48, py + 40.5, "&#931;", 13, "#0b0f19", weight="bold")
            elif item:
                for j, (c, sig, dim) in enumerate(item):
                    if c is None:
                        continue
                    out += chip(bx + 8 + j * 21, py + 26, c, sigma=sig, dim=dim)
        return out

    def panel(px, py, title, before, after, note):
        out = f'<rect x="{px}" y="{py}" width="{PW}" height="{PH}" rx="10" fill="none" stroke="rgba(0,200,255,0.18)"/>\n'
        out += text(px + PW / 2, py + 22, title, 14, GLOW, weight="bold")
        out += rank_row(px + 12, py + 34, before)
        out += arrow(px + PW / 2, py + 100, px + PW / 2, py + 124)
        out += rank_row(px + 12, py + 128, after)
        out += text(px + PW / 2, py + PH - 8, note, 10.5, TEXT2)
        return out

    c = S
    full = [(c[j], False, False) for j in range(4)]
    empty = None

    # broadcast: root has full tensor -> all have it
    s += panel(cols_x[0], rows_y[0], "Broadcast",
               [("bar", c[0]), empty, empty, empty],
               [("bar", c[0])] * 4,
               "root sends the full tensor to everyone &#183; 1 &#8594; N")
    # scatter
    s += panel(cols_x[1], rows_y[0], "Scatter",
               [full, empty, empty, empty],
               [[(c[k] if j == k else None, False, False) for j in range(4)] for k in range(4)],
               "root deals one shard to each rank &#183; 1 &#8594; N")
    # gather
    s += panel(cols_x[0], rows_y[1], "Gather",
               [[(c[k] if j == k else None, False, False) for j in range(4)] for k in range(4)],
               [full, empty, empty, empty],
               "every rank sends its shard to root &#183; N &#8594; 1")
    # all-gather
    s += panel(cols_x[1], rows_y[1], "All-Gather",
               [[(c[k] if j == k else None, False, False) for j in range(4)] for k in range(4)],
               [full] * 4,
               "everyone ends up with all shards &#183; N &#8594; N")
    # reduce-scatter
    s += panel(cols_x[0], rows_y[2], "Reduce-Scatter",
               [full] * 4,
               [[(c[k] if j == k else None, j == k, False) for j in range(4)] for k in range(4)],
               "shard k gets summed across ranks, lands on rank k &#183; N &#8594; N")
    # all-reduce
    s += panel(cols_x[1], rows_y[2], "All-Reduce",
               [full] * 4,
               [[(c[j], True, False) for j in range(4)] for k in range(4)],
               "= reduce-scatter + all-gather &#183; everyone gets the full sum")

    s += "</svg>\n"
    return s


# ---------------------------------------------------------------- fig 2
# Ring geometry shared by every panel: G0 top, G1 right, G2 bottom, G3 left,
# increasing k = clockwise on screen (matches the ring's send direction).
NODE_ANGLE = [-90, 0, 90, 180]
RING_R = 72
NODE_W, NODE_H = 50, 28
RING_CHIP, RING_GAP = 10, 1.4
ARC_CLEARANCE = 25  # degrees kept clear of each node so arcs don't cross the boxes


def ring_node(ccx, ccy, k, cnt_row, just_updated):
    """One GPU's box (label + 4 shard chips) at its compass position on the ring."""
    nx, ny = polar(ccx, ccy, RING_R, NODE_ANGLE[k])
    bx, by = nx - NODE_W / 2, ny - NODE_H / 2
    out = (f'<rect x="{bx:.1f}" y="{by:.1f}" width="{NODE_W}" height="{NODE_H}" rx="6" '
           f'fill="{SURFACE2}" stroke="{GLOW}" stroke-width="1.1"/>\n')
    out += text(bx + 5, by + 9.5, f"G{k}", 7.5, TEXT2, "start", weight="bold")
    row_w = 4 * RING_CHIP + 3 * RING_GAP
    cx0, cy = nx - row_w / 2, by + NODE_H - RING_CHIP - 3
    for j in range(4):
        n = cnt_row[j]
        full = n == 4
        op = {1: 0.30, 2: 0.55, 3: 0.8, 4: 1.0}[n]
        cx = cx0 + j * (RING_CHIP + RING_GAP)
        if full:
            extra = f' stroke="{TEXT}" stroke-width="1.2"'
        elif j == just_updated:
            extra = f' stroke="{S[j]}" stroke-width="1.5"'
        else:
            extra = ""
        out += (f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{RING_CHIP}" height="{RING_CHIP}" rx="2.2" '
                f'fill="{S[j]}" opacity="{op}"{extra}/>\n')
        if full:
            out += text(cx + RING_CHIP / 2, cy + RING_CHIP / 2 + 3, "&#931;", 7.5, "#0b0f19", weight="bold")
        elif j == just_updated:
            out += text(cx + RING_CHIP / 2, cy + RING_CHIP / 2 + 3, "+", 8, "#0b0f19", weight="bold")
    return out


def ring_panel(ccx, ccy, cnt, rs_step, title):
    """One full step: faint ring guide, 4 colored/numbered arcs (or neutral for AG), 4 nodes."""
    out = text(ccx, ccy - RING_R - NODE_H / 2 - 12, title, 12.5, GLOW, weight="bold")
    out += (f'<circle cx="{ccx}" cy="{ccy}" r="{RING_R}" fill="none" '
            f'stroke="rgba(0,200,255,0.16)" stroke-width="1" stroke-dasharray="2,4"/>\n')
    if rs_step is not None:
        for k in range(4):
            b = (k + 1) % 4
            a0 = NODE_ANGLE[k] + ARC_CLEARANCE
            a1 = NODE_ANGLE[k] + 90 - ARC_CLEARANCE
            if isinstance(rs_step, int):
                idx = (b - rs_step) % 4
                color, marker = S[idx], f"arr{idx}"
            else:  # AG: neutral color, arrows just move already-finished shards
                idx, color, marker = None, GLOW, "arr"
            x0, y0 = polar(ccx, ccy, RING_R, a0)
            x1, y1 = polar(ccx, ccy, RING_R, a1)
            out += (f'<path d="M {x0:.1f} {y0:.1f} A {RING_R},{RING_R} 0 0,1 {x1:.1f} {y1:.1f}" '
                    f'fill="none" stroke="{color}" stroke-width="1.5" opacity="0.9" '
                    f'marker-end="url(#{marker})"/>\n')
            if idx is not None:
                mx, my = polar(ccx, ccy, RING_R, (a0 + a1) / 2)
                out += flying_chip(mx, my, color, idx, size=13)
    for k in range(4):
        just_updated = (k - rs_step) % 4 if isinstance(rs_step, int) else None
        out += ring_node(ccx, ccy, k, cnt[k], just_updated)
    return out


def fig2():
    """Ring all-reduce state evolution: 5 small-multiple rings (t=0, RS1-3, AG).
    Each panel is the literal ring topology: G0/G1/G2/G3 at 12/3/6/9 o'clock, arcs flow
    clockwise. Chip brightness = terms accumulated; Sigma = all 4 terms present. During
    reduce-scatter every arc carries a DIFFERENT shard, colored/numbered to match the
    column it lands in, and that slot gets a matching ring + "+" on arrival."""
    W, H = 1060, 440
    s = svg_open(W, H)
    s += per_shard_markers()
    s += text(W / 2, 28, "Ring All-Reduce, step by step (N = 4)", 17, TEXT, weight="bold")
    s += text(W / 2, 47, "G0/G1/G2/G3 sit clockwise at 12/3/6/9 o&#8217;clock &#183; brightness = how many of the 4 terms are accumulated", 11, TEXT2)

    # legend: chip position -> shard color/index (same order inside every node, every panel)
    s += text(W / 2, 63, "chip position &#8594; shard index (same order in every box):", 10.5, TEXT2)
    leg_x0 = W / 2 - (4 * 40) / 2 + 10
    for j in range(4):
        s += flying_chip(leg_x0 + j * 40, 74, S[j], j)

    # worked example: RS is a real addition, not just a move. Spells out the very first
    # hop (RS step 1, G0 -> G1) with the actual +/= signs and the actual chip visuals.
    # Text widths are estimated generously (monospace advance ~0.65em + padding) so tokens
    # never collide regardless of exact string length.
    def mono_w(txt, size, pad=16):
        return len(txt) * size * 0.65 + pad

    s += text(W / 2, 96, "every hop is a real addition &#8212; worked example, RS step 1, G0&#8594;G1:", 10.5, TEXT2)
    eq_own, eq_arr, eq_res = "own share (1 term)", "shard 0 arrives from G0", "now 2 of 4 terms summed"
    s += inline_row(W / 2, 116, [
        {"kind": "chip", "w": 18, "size": 11, "color": S[0], "op": 0.30},
        {"kind": "text", "w": mono_w(eq_own, 9), "s": eq_own, "size": 9},
        {"kind": "text", "w": 22, "s": "+", "size": 13, "fill": TEXT, "weight": "bold"},
        {"kind": "chip", "w": 28, "size": 15, "color": S[0], "op": 1.0, "glyph": "0", "gsize": 9},
        {"kind": "text", "w": mono_w(eq_arr, 9), "s": eq_arr, "size": 9},
        {"kind": "text", "w": 22, "s": "=", "size": 13, "fill": TEXT, "weight": "bold"},
        {"kind": "chip", "w": 18, "size": 11, "color": S[0], "op": 0.55, "ring": True, "glyph": "+", "gsize": 7.5},
        {"kind": "text", "w": mono_w(eq_res, 9), "s": eq_res, "size": 9},
    ])

    # accumulated counts per (step, rank, chunk)
    # phase 1: reduce-scatter around the ring; phase 2: all-gather
    def rs_counts(step):  # step = 0..3 (0 = initial)
        # RS step t (1-indexed): rank k receives the partial sum of chunk (k-t) mod 4
        # (already holding t terms), adds its own share -> t+1 terms.
        cnt = [[1] * 4 for _ in range(4)]
        for t in range(1, step + 1):
            for k in range(4):
                j = (k - t) % 4
                cnt[k][j] = t + 1
        return cnt

    def ag_counts(step):  # after RS done; step = 0..3
        cnt = rs_counts(3)
        # after RS: rank k owns full sum of chunk (k+1)%4? derive: at t=3, j=(k-3)%4=(k+1)%4 -> 4 terms
        for k in range(4):
            cnt[k][(k + 1) % 4] = 4
        for t in range(1, step + 1):
            for k in range(4):
                # AG step t: rank k receives full chunk ((k - t) + 1) % 4
                cnt[k][((k - t) + 1) % 4] = 4
        return cnt

    panels = [
        ("t = 0", rs_counts(0), None,
         "holds its own gradient share"),
        ("RS step 1", rs_counts(1), 1,
         "colored arc = shard carried"),
        ("RS step 2", rs_counts(2), 2,
         "one hop further, clockwise"),
        ("RS step 3", rs_counts(3), 3,
         "each rank owns one full shard"),
        ("AG steps 1&#8211;3", ag_counts(3), "ag",
         "pure data move, no more +"),
    ]

    n = len(panels)
    margin = 20
    pitch = (W - 2 * margin) / n
    ccy = 245
    for i, (title, cnt, rs_step, note) in enumerate(panels):
        ccx = margin + pitch / 2 + i * pitch
        s += ring_panel(ccx, ccy, cnt, rs_step, title)
        s += text(ccx, ccy + RING_R + NODE_H / 2 + 24, note, 9, TEXT2)
        if i > 0:
            s += (f'<line x1="{margin + i * pitch:.1f}" y1="{ccy - RING_R - 6}" '
                  f'x2="{margin + i * pitch:.1f}" y2="{ccy + RING_R + 30}" '
                  f'stroke="rgba(122,136,153,0.18)" stroke-width="1"/>\n')

    # bottom ledger
    yb = ccy + RING_R + NODE_H / 2 + 44
    lw = W - 2 * 120
    s += (f'<rect x="120" y="{yb}" width="{lw}" height="52" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.25)"/>\n')
    s += text(W / 2, yb + 21, "bytes sent per GPU = (N&#8722;1)&#183;S/N (reduce-scatter) + (N&#8722;1)&#183;S/N (all-gather)", 12.5, TEXT)
    s += text(W / 2, yb + 40, "= 2(N&#8722;1)/N &#183; S &#8776; 2S", 13.5, GLOW, weight="bold")
    s += "</svg>\n"
    return s


# ---------------------------------------------------------------- fig 3
def fig3():
    """Local topology: 2 NUMA nodes x 2 PCIe switches x 2 GPUs."""
    W, H = 960, 400
    s = svg_open(W, H)
    s += text(W / 2, 30, "Our testbed: 8&#215; RTX PRO 6000, PCIe only &#8212; no NVLink", 17, TEXT, weight="bold")
    s += text(W / 2, 50, 'nvidia-smi topo -m: PIX = same PCIe switch &#183; NODE = same NUMA &#183; SYS = across the UPI socket link', 11.5, TEXT2)

    def numa(x, label, gpus):
        out = (f'<rect x="{x}" y="80" width="430" height="250" rx="12" fill="none" '
               f'stroke="{TEXT2}" stroke-dasharray="6,5"/>\n')
        out += text(x + 215, 102, label, 12, TEXT2, weight="bold")
        for i, gx in enumerate([x + 30, x + 230]):
            out += (f'<rect x="{gx}" y="120" width="170" height="60" rx="8" fill="{SURFACE2}" '
                    f'stroke="rgba(0,200,255,0.35)"/>\n')
            out += text(gx + 85, 145, "PCIe switch", 11, TEXT2)
            out += text(gx + 85, 162, "(PIX pair)", 9.5, TEXT2)
            for j, gpx in enumerate([gx + 8, gx + 90]):
                g = gpus[i * 2 + j]
                out += gpu_box(gpx, 210, w=72, h=48, label=f"GPU {g}")
                out += (f'<line x1="{gpx + 36}" y1="210" x2="{gx + 85}" y2="180" '
                        f'stroke="rgba(0,200,255,0.5)" stroke-width="1.2"/>\n')
        return out

    s += numa(30, "NUMA node 0 (CPU 0-95)", [0, 1, 2, 3])
    s += numa(500, "NUMA node 1 (CPU 96-191)", [4, 5, 6, 7])
    # UPI link
    s += f'<line x1="458" y1="205" x2="502" y2="205" stroke="{S[1]}" stroke-width="10" opacity="0.25" stroke-linecap="round"/>\n'
    s += f'<line x1="458" y1="205" x2="502" y2="205" stroke="{S[1]}" stroke-width="4"/>\n' 
    s += f'<rect x="443" y="182" width="74" height="17" rx="4" fill="{SURFACE}"/>\n'
    s += text(480, 195, "UPI (SYS)", 11, S[1], weight="bold")
    s += text(480, 360, "every ring crossing the socket boundary is throttled by this single red link &#8212; the bandwidth ceiling of all 8-GPU collectives", 11.5, TEXT2)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-six-primitives", fig1),
                     ("fig-2-ring-allreduce", fig2),
                     ("fig-3-topology", fig3)]:
        path = os.path.join(OUT, f"{name}.svg")
        with open(path, "w") as f:
            f.write(fn())
        print("wrote", path)
