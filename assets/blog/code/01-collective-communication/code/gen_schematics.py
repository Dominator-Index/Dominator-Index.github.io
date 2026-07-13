"""
第 01 篇示意图生成器(程序化 SVG,保证全系列视觉一致)。

生成:
  figures/fig-1-six-primitives.svg   六种集合通信原语总览
  figures/fig-2-ring-allreduce.svg   ring all-reduce 逐步状态(N=4)
  figures/fig-3-topology.svg         本机 PCIe 拓扑(8 卡, PIX 对 + 双 NUMA)

配色遵循 _shared/style-guide.md。图内文字英文(中英文版共用)。
"""

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


def arrow(x1, y1, x2, y2, dash=False, width=1.6):
    d = ' stroke-dasharray="5,4"' if dash else ""
    halo = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GLOW}" '
            f'stroke-width="{width * 3}" opacity="0.22" stroke-linecap="round"/>\n')
    return halo + (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GLOW}" '
                   f'stroke-width="{width}" marker-end="url(#arr)"{d}/>\n')


# ---------------------------------------------------------------- fig 1
def fig1():
    """六原语总览:每个 panel 上排 before、下排 after,4 ranks。"""
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
def fig2():
    """ring all-reduce 状态演化:行 = 步骤,列 = 4 GPU。
    chip 亮度 = 该分片已累加的项数;Σ = 4 项全齐。"""
    W, H = 960, 700
    s = svg_open(W, H)
    s += text(W / 2, 30, "Ring All-Reduce, step by step (N = 4)", 17, TEXT, weight="bold")
    s += text(W / 2, 50, "each cell: the 4 shard-slots on one GPU &#183; brightness = how many of the 4 terms are accumulated", 11.5, TEXT2)

    # accumulated counts per (step, rank, chunk)
    # phase 1: reduce-scatter around the ring; phase 2: all-gather
    def rs_counts(step):  # step = 0..3 (0 = initial)
        # RS 第 t 步(1-indexed):rank k 收到 chunk (k-t) mod 4 的部分和(已含 t 项),
        # 加上自己的一份 -> t+1 项。
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

    rows = [
        ("t = 0  (initial)", rs_counts(0), "every rank holds its own full gradient, 4 shards &#215; S/4"),
        ("RS step 1", rs_counts(1), "send shard, add into what arrives &#183; S/4 bytes on each link"),
        ("RS step 2", rs_counts(2), "partial sums keep travelling clockwise"),
        ("RS step 3", rs_counts(3), "each rank now owns ONE fully-summed shard"),
        ("AG step 1-3", ag_counts(3), "the finished shards travel once around &#183; 3 more sends of S/4"),
    ]

    x0, y0 = 150, 78
    CW, RH = 180, 108
    for k in range(4):
        s += text(x0 + k * CW + 48, y0 - 6, f"GPU {k}", 12, TEXT2)
    for r, (label, cnt, note) in enumerate(rows):
        y = y0 + r * RH + 12
        s += text(18, y + 32, label, 12, GLOW, "start", weight="bold")
        s += text(18, y + 48, "", 10, TEXT2, "start")
        for k in range(4):
            bx = x0 + k * CW
            s += gpu_box(bx, y, w=96, h=58, label=f"G{k}")
            for j in range(4):
                n = cnt[k][j]
                full = n == 4
                op = {1: 0.30, 2: 0.55, 3: 0.8, 4: 1.0}[n]
                cx = bx + 8 + j * 21
                extra = f' stroke="{TEXT}" stroke-width="1.4"' if full else ""
                s += (f'<rect x="{cx}" y="{y + 26}" width="18" height="18" rx="3" '
                      f'fill="{S[j]}" opacity="{op}"{extra}/>\n')
                if full:
                    s += text(cx + 9, y + 26 + 13.5, "&#931;", 11, "#0b0f19", weight="bold")
        # ring arrows between columns (except last annotation row uses same)
        if r in (1, 2, 3, 4):
            for k in range(4):
                x_from = x0 + k * CW + 96
                x_to = x0 + ((k + 1) % 4) * CW
                if k < 3:
                    s += arrow(x_from + 4, y + 29, x_to - 6, y + 29, width=1.3)
            # wrap-around arrow drawn as curve above
            s += (f'<path d="M {x0 + 3 * CW + 96 + 4} {y + 8} C {x0 + 3 * CW + 150} {y - 16}, '
                  f'{x0 - 40} {y - 16}, {x0 - 4} {y + 20}" fill="none" stroke="{GLOW}" '
                  f'stroke-width="1.3" marker-end="url(#arr)" opacity="0.8"/>\n')
        s += text(x0 + 2 * CW - 45, y + 84, note, 10.5, TEXT2)

    # bottom ledger
    yb = y0 + 5 * RH + 26
    s += (f'<rect x="120" y="{yb}" width="720" height="52" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.25)"/>\n')
    s += text(480, yb + 21, "bytes sent per GPU = (N&#8722;1)&#183;S/N (reduce-scatter) + (N&#8722;1)&#183;S/N (all-gather)", 12.5, TEXT)
    s += text(480, yb + 40, "= 2(N&#8722;1)/N &#183; S &#8776; 2S", 13.5, GLOW, weight="bold")
    s += "</svg>\n"
    return s


# ---------------------------------------------------------------- fig 3
def fig3():
    """本机拓扑:2 NUMA × 2 PCIe switch × 2 GPU。"""
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
