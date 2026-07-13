"""Part 05 schematics: the Column×Row golden pair + why the order cannot flip."""

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


def arrow(x1, y1, x2, y2, color=GLOW, width=1.6):
    halo = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width * 3}" opacity="0.22" stroke-linecap="round"/>\n')
    return halo + (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                   f'stroke-width="{width}" marker-end="url(#arr)"/>\n')


def mat(x, y, w, h, color, label="", sub="", split=None, op=0.9):
    """split: ('h', frac) horizontal cut or ('v', frac) vertical cut, two-tone marks the two ranks."""
    out = ""
    if split is None:
        out += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{color}" opacity="{op}"/>\n'
    elif split[0] == "h":
        out += f'<rect x="{x}" y="{y}" width="{w}" height="{h / 2 - 1}" rx="4" fill="{S[0]}" opacity="{op}"/>\n'
        out += f'<rect x="{x}" y="{y + h / 2 + 1}" width="{w}" height="{h / 2 - 1}" rx="4" fill="{S[3]}" opacity="{op}"/>\n'
    else:
        out += f'<rect x="{x}" y="{y}" width="{w / 2 - 1}" height="{h}" rx="4" fill="{S[0]}" opacity="{op}"/>\n'
        out += f'<rect x="{x + w / 2 + 1}" y="{y}" width="{w / 2 - 1}" height="{h}" rx="4" fill="{S[3]}" opacity="{op}"/>\n'
    if label:
        out += text(x + w / 2, y + h / 2 + 4, label, 11, "#0b0f19" if split is None else TEXT, weight="bold")
    if sub:
        out += text(x + w / 2, y + h + 16, sub, 9.5, TEXT2)
    return out


# ---- fig 1: the golden pair ------------------------------------------------
def fig1():
    W, H = 960, 460
    s = svg_open(W, H)
    s += text(W / 2, 30, "Megatron's golden pair: Column-cut fc1 + Row-cut fc2 (TP = 2)", 16.5, TEXT, weight="bold")
    s += text(W / 2, 52, "cyan = rank 0&#8217;s piece &#183; green = rank 1&#8217;s &#183; the GeLU in the middle needs NO communication", 11.5, TEXT2)

    yC = 120
    # X replicated
    s += mat(50, yC, 70, 90, S[2], "X", "[B, H]  replicated")
    s += arrow(128, yC + 45, 168, yC + 45)
    s += text(148, yC + 30, "f", 13, GLOW, weight="bold")
    # W1 column-cut (out dim = rows of [out,in] -> horizontal split)
    s += mat(172, yC - 10, 100, 110, None, "", split=("h", 0.5))
    s += text(222, yC + 118, "W1 [4H,H] col-parallel", 9.2, TEXT2)
    s += text(222, yC + 132, "split OUT dim: whole rows", 9.2, TEXT2)
    s += arrow(280, yC + 45, 320, yC + 45)
    # Y halves + GeLU
    s += mat(324, yC - 10, 70, 50, S[0], "Y&#8320;", "", op=0.95)
    s += mat(324, yC + 50, 70, 50, S[3], "Y&#8321;", "")
    s += text(359, yC + 118, "Y = GeLU(XW1&#7488;)", 9.2, TEXT2)
    s += text(359, yC + 132, "own half &#8212; LOCAL", 9.2, S[4])
    s += arrow(402, yC + 45, 442, yC + 45)
    # W2 row-cut (in dim = cols of [out,in] -> vertical split)
    s += mat(446, yC - 10, 100, 110, None, "", "", split=("v", 0.5))
    s += text(496, yC + 118, "W2 [H,4H] row-parallel", 9.2, TEXT2)
    s += text(496, yC + 132, "split IN dim = Y&#8217;s split", 9.2, TEXT2)
    s += arrow(554, yC + 45, 594, yC + 45)
    # partial sums
    s += mat(598, yC - 10, 70, 50, S[0], "Z&#8320;", "", op=0.6)
    s += mat(598, yC + 50, 70, 50, S[3], "Z&#8321;", "", op=0.6)
    s += text(633, yC + 118, "partial sums [B,H]", 9.2, TEXT2)
    s += text(633, yC + 132, "Z = Z&#8320;+Z&#8321;", 9.2, TEXT2)
    s += arrow(676, yC + 45, 716, yC + 45)
    s += text(696, yC + 30, "g", 13, GLOW, weight="bold")
    # all-reduce
    s += (f'<rect x="720" y="{yC - 10}" width="120" height="110" rx="8" fill="none" '
          f'stroke="{GLOW}" stroke-width="2"/>\n')
    s += text(780, yC + 38, "all-reduce", 11.5, GLOW, weight="bold")
    s += text(780, yC + 56, "Z on every rank", 9.5, TEXT)
    s += text(780, yC + 72, "(the ONLY fwd comm)", 9, TEXT2)

    # f/g conjugate note
    yN = 300
    s += (f'<rect x="120" y="{yN}" width="720" height="118" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.2)"/>\n')
    s += text(480, yN + 26, "f and g are conjugate operators &#8212; each is the other&#8217;s transpose in backward:", 11.5, TEXT)
    s += text(480, yN + 52, "f :  forward = identity (copy X)          backward = all-reduce (sum the partial dX from both ranks)", 10.5, TEXT2)
    s += text(480, yN + 72, "g :  forward = all-reduce (sum partial Z)  backward = identity (copy dZ)", 10.5, TEXT2)
    s += text(480, yN + 98, "per layer per step: 1 all-reduce fwd + 1 bwd (&#215;2 with attention) &#183; message = activation [B&#183;s, H], not weights", 10.5, GLOW)
    s += "</svg>\n"
    return s


# ---- fig 2: why the order cannot flip ---------------------------------------
def fig2():
    W, H = 960, 330
    s = svg_open(W, H)
    s += text(W / 2, 30, "Why the order cannot flip: GeLU(a + b) &#8800; GeLU(a) + GeLU(b)", 16.5, TEXT, weight="bold")
    s += text(W / 2, 52, "put the Row-cut first and the nonlinearity forces an all-reduce IN THE MIDDLE of the layer", 11.5, TEXT2)

    # good path
    y1 = 100
    s += text(70, y1 + 24, "&#10003; Col&#8594;Row", 12, S[3], "start", weight="bold")
    steps = [("XW1&#7488; (col-cut)", S[0]), ("GeLU &#8212; local &#10003;", S[3]), ("&#183;W2&#7488; (row-cut)", S[0]), ("all-reduce", None)]
    x = 220
    for lab, c in steps:
        if c is None:
            s += f'<rect x="{x}" y="{y1}" width="130" height="44" rx="6" fill="none" stroke="{GLOW}" stroke-width="2"/>\n'
            s += text(x + 65, y1 + 27, lab, 10.5, GLOW, weight="bold")
        else:
            s += f'<rect x="{x}" y="{y1}" width="130" height="44" rx="6" fill="{c}" opacity="0.9"/>\n'
            s += text(x + 65, y1 + 27, lab, 10.5, "#0b0f19", weight="bold")
        if x < 700:
            s += arrow(x + 134, y1 + 22, x + 162, y1 + 22)
        x += 166
    s += text(886, y1 + 27, "1 comm", 11, S[3], "start", weight="bold")

    # bad path
    y2 = 200
    s += text(70, y2 + 24, "&#10007; Row&#8594;Col", 12, S[1], "start", weight="bold")
    steps = [("XW1&#7488; (row-cut)", S[0]), ("all-reduce!", None), ("GeLU", S[0]), ("&#183;W2&#7488; + all-reduce", None)]
    x = 220
    for lab, c in steps:
        if c is None:
            s += f'<rect x="{x}" y="{y2}" width="130" height="44" rx="6" fill="none" stroke="{S[1]}" stroke-width="2"/>\n'
            s += text(x + 65, y2 + 27, lab, 10.5, S[1], weight="bold")
        else:
            s += f'<rect x="{x}" y="{y2}" width="130" height="44" rx="6" fill="{c}" opacity="0.9"/>\n'
            s += text(x + 65, y2 + 27, lab, 10.5, "#0b0f19", weight="bold")
        if x < 700:
            s += arrow(x + 134, y2 + 22, x + 162, y2 + 22)
        x += 166
    s += text(886, y2 + 27, "2 comms", 11, S[1], "start", weight="bold")
    s += text(345, y2 + 62, "partial sums must be completed BEFORE the nonlinearity &#8212; comm lands mid-layer", 9.5, S[1])

    s += text(W / 2, 306, "the nonlinearity dictates the cut order &#8212; that is the whole design of Megatron TP", 11, GLOW)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-golden-pair", fig1), ("fig-2-why-order", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
