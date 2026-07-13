"""RMNP vs Muon standalone post schematics: fig-1 operator locality x shard geometry, fig-2 the communication ledger."""

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
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n'
            f'<defs>'
            f'<marker id="ag" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{GLOW}"/></marker>'
            f'<marker id="ar" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{S[1]}"/></marker>'
            f'</defs>\n')


def text(x, y, s, size=13, fill=TEXT, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>\n')


def box(x, y, w, h, fill=SURFACE2, stroke=GLOW, sw=1.5, rx=8, opacity=1.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}/>\n')


def arrow(x1, y1, x2, y2, color=GLOW, sw=2, marker="ag"):
    out = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
           f'stroke-width="{sw * 3}" opacity="0.18"/>\n')
    out += (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"/>\n')
    return out


def sharded_matrix(x0, y0, w, h, n_shards=4, label=True):
    """FSDP2 row-sharded matrix: n_shards row blocks, each labeled GPU k. Returns (svg, block height)."""
    s = ""
    bh = h / n_shards
    for k in range(n_shards):
        y = y0 + k * bh
        s += box(x0, y, w, bh - 4, fill=SURFACE2, stroke=GLOW, sw=1.2, rx=5)
        if label:
            s += text(x0 - 10, y + bh / 2 + 2, f"GPU {k}", 10, TEXT2, "end")
    return s, bh


def fig1():
    W, H = 960, 560
    s = svg_open(W, H)
    s += text(W / 2, 30, "The same FSDP2 dim-0 sharding, two optimizers, two very different bills", 15, TEXT, weight="bold")
    s += text(W / 2, 50, "each GPU holds whole rows of every [m&#215;n] weight (and its momentum shard)", 10.5, TEXT2)

    mw, mh = 280, 240

    # ---- left: RMNP ----
    lx, ly = 110, 110
    s += text(lx + mw / 2, 96, "RMNP: update row i = M&#7522; / &#8214;M&#7522;&#8214;", 12.5, S[3], weight="bold")
    m_svg, bh = sharded_matrix(lx, ly, mw, mh)
    s += m_svg
    # highlight one row inside GPU1 (visual only, no inline text)
    ry = ly + bh + 14
    s += f'<rect x="{lx + 6}" y="{ry}" width="{mw - 12}" height="10" rx="3" fill="{S[3]}" opacity="0.9"/>\n'
    s += arrow(lx + 20, ry + 5, lx + mw - 20, ry + 5, S[3], 1.6, "ag")
    s += text(lx + mw / 2, ly + mh + 22, "&#8214;M&#7522;&#8214; sweeps only its own row &#8212; which lives whole on one GPU &#10003;", 9.5, S[3])
    s += box(lx - 20, ly + mh + 38, mw + 40, 84, fill="rgba(25,158,112,0.08)", stroke=S[3], sw=1.2, rx=8)
    s += text(lx + mw / 2, ly + mh + 62, "every rank preconditions its own shard", 10.5, TEXT)
    s += text(lx + mw / 2, ly + mh + 84, "communication: 0 bytes", 12.5, S[3], weight="bold")
    s += text(lx + mw / 2, ly + mh + 104, "compute O(mn) elementwise, load-balanced by construction", 9, TEXT2)

    # ---- right: Muon ----
    rx_, ry_ = 570, 110
    s += text(rx_ + mw / 2, 96, "Muon: update = NS(M) &#8776; UV&#7488;", 12.5, S[1], weight="bold")
    m_svg, bh = sharded_matrix(rx_, ry_, mw, mh)
    s += m_svg
    # arrows showing row-row coupling across GPUs (A = XX^T)
    y_a = ry_ + bh * 0.5
    y_b = ry_ + bh * 2.5
    y_c = ry_ + bh * 3.5
    s += arrow(rx_ + mw * 0.25, y_a, rx_ + mw * 0.25, y_c - 4, S[1], 1.6, "ar")
    s += arrow(rx_ + mw * 0.5, y_b, rx_ + mw * 0.5, y_a + 4, S[1], 1.6, "ar")
    s += arrow(rx_ + mw * 0.75, y_c, rx_ + mw * 0.75, y_b + 4, S[1], 1.6, "ar")
    s += text(rx_ + mw / 2, ly + mh + 22, "A = XX&#7488;: A&#7522;&#11388; = &#9001;row i, row j&#9002; &#8212; rows on DIFFERENT GPUs", 9.5, S[1])
    s += box(rx_ - 20, ry_ + mh + 38, mw + 40, 84, fill="rgba(230,103,103,0.08)", stroke=S[1], sw=1.2, rx=8)
    s += text(rx_ + mw / 2, ry_ + mh + 62, "no rank can start NS without the others' rows", 10.5, TEXT)
    s += text(rx_ + mw / 2, ry_ + mh + 84, "all-gather full matrix: O(mn) / step", 12.5, S[1], weight="bold")
    s += text(rx_ + mw / 2, ry_ + mh + 104, "+ NS matmul chain O(mn&#183;min(m,n)), needs scheduling", 9, TEXT2)

    s += text(W / 2, 532, "locality of the precondition operator &#215; geometry of the shard = the communication bill", 11, GLOW)
    s += "</svg>\n"
    return s


def fig2():
    W, H = 960, 470
    s = svg_open(W, H)
    s += text(W / 2, 32, "The ledger: optimizer-added communication per [m&#215;n] matrix, per step, per GPU", 14.5, TEXT, weight="bold")
    s += text(W / 2, 52, "gradient sync (DP all-reduce / ZeRO reduce-scatter) is identical for both and excluded", 10, TEXT2)

    rows = [
        ("DDP (replicated)", "full matrix", "0&#8202;*", S[3], "0", S[3]),
        ("FSDP2 / ZeRO-3 (dim-0)", "whole-row block", "all-gather O(mn)", S[1], "0", S[3]),
        ("TP row-block split", "whole-row block", "all-gather O(mn)", S[1], "0", S[3]),
        ("TP column split", "1/tp of every row", "all-gather O(mn)", S[1], "all-reduce O(m) vec", S[4]),
    ]
    x0, y0, rh = 60, 96, 64
    col_x = [x0, 300, 520, 740]
    col_w = [230, 210, 210, 160]
    s += text(col_x[0] + 8, y0 - 10, "layout", 10.5, TEXT2, "start")
    s += text(col_x[1] + 8, y0 - 10, "each GPU holds", 10.5, TEXT2, "start")
    s += text(col_x[2] + col_w[2] / 2 - 55, y0 - 10, "Muon", 12, TEXT, "middle", "bold")
    s += text(col_x[3] + col_w[3] / 2, y0 - 10, "RMNP", 12, TEXT, "middle", "bold")
    for r, (layout, holds, muon, cm, rmnp, cr) in enumerate(rows):
        y = y0 + r * rh
        s += box(x0 - 8, y, 880, rh - 8, fill=SURFACE2, stroke=TEXT2, sw=0.8, rx=8, opacity=0.55)
        s += text(col_x[0] + 8, y + rh / 2 + 1, layout, 11, TEXT, "start")
        s += text(col_x[1] + 8, y + rh / 2 + 1, holds, 10, TEXT2, "start")
        s += box(col_x[2] - 4, y + 9, col_w[2] - 30, rh - 26, fill="none", stroke=cm, sw=1.4, rx=7)
        s += text(col_x[2] + (col_w[2] - 30) / 2 - 4, y + rh / 2 + 1, muon, 10.5, cm, "middle", "bold")
        s += box(col_x[3] - 4, y + 9, col_w[3], rh - 26, fill="none", stroke=cr, sw=1.4, rx=7)
        s += text(col_x[3] + col_w[3] / 2 - 4, y + rh / 2 + 1, rmnp, 10.5, cr, "middle", "bold")

    yb = y0 + 4 * rh + 12
    s += text(x0, yb + 6, "* replicated: NS re-computed on every rank (or amortized Moonlight-style at the price of an O(&#936;) all-reduce)", 9, TEXT2, "start")

    s += box(60, yb + 22, 880, 88, fill="rgba(0,200,255,0.05)", stroke=GLOW, sw=1.2, rx=10)
    s += text(80, yb + 46, "GPT-2 Large (708M matrix params), 8 GPUs, bf16 momentum, per optimizer step:", 10.5, TEXT, "start")
    s += text(80, yb + 68, "Muon must move ~1.15 GiB/GPU on the un-overlappable critical path (measured: +57 ms comm, +303 ms redundant NS);", 11, S[1], "start", "bold")
    s += text(80, yb + 88, "RMNP moves 0 (FSDP2/row-TP) or a ~3 MiB vector (column-TP): 5.3 ms total — a 67&#215; gap that grows with scale.", 11, S[3], "start", "bold")

    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-locality-geometry", fig1), ("fig-2-comm-ledger", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print(f"wrote {name}.svg")
