"""Part 06 schematics: Megatron-SP (splitting AR=AG+RS in space) + Ring Attention."""

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


# ---- fig 1: Megatron-SP ----------------------------------------------------
def fig1():
    W, H = 960, 470
    s = svg_open(W, H)
    s += text(W / 2, 30, "Megatron-SP: split the all-reduce in SPACE (the identity's third appearance)", 16, TEXT, weight="bold")
    s += text(W / 2, 52, "plain TP: activations enter/leave each block via all-reduce &#183; SP: = all-gather on entry + reduce-scatter on exit", 11, TEXT2)

    # top: plain TP
    y0 = 100
    s += text(70, y0 + 26, "TP only", 12, S[1], "start", weight="bold")
    regions = [("LN / dropout", S[2], "", 150),
               ("g: all-reduce", None, "", 112),
               ("attention / MLP (TP)", S[0], "", 190),
               ("g: all-reduce", None, "", 112),
               ("LN / dropout", S[2], "", 150)]
    x = 150
    for lab, c, sub, w in regions:
        if c is None:
            s += f'<rect x="{x}" y="{y0}" width="{w}" height="52" rx="6" fill="none" stroke="{GLOW}" stroke-width="1.8"/>\n'
            s += text(x + w / 2, y0 + 30, lab, 9.5, GLOW, weight="bold")
        else:
            s += f'<rect x="{x}" y="{y0}" width="{w}" height="52" rx="6" fill="{c}" opacity="0.85"/>\n'
            s += text(x + w / 2, y0 + 30, lab, 10, "#0b0f19", weight="bold")
        x += w + 8
    s += text(150 + 75, y0 + 70, "activations REPLICATED &#215;N", 9, S[1])
    s += text(150 + 150 + 8 + 112 + 8 + 95, y0 + 70, "TP protects only this part", 9, TEXT2)

    # bottom: TP + SP
    y1 = 230
    s += text(70, y1 + 26, "TP + SP", 12, S[3], "start", weight="bold")
    regions = [("LN / dropout", S[3], 150, "SHARDED [b,s/N,h]"),
               ("&#7511;: all-gather", None, 112, "gather s"),
               ("attention / MLP (TP)", S[0], 190, "unchanged"),
               ("&#7511;&#773;: reduce-scatter", None, 112, "scatter+sum"),
               ("LN / dropout", S[3], 150, "SHARDED [b,s/N,h]")]
    x = 150
    for lab, c, w, sub in regions:
        if c is None:
            s += f'<rect x="{x}" y="{y1}" width="{w}" height="52" rx="6" fill="none" stroke="{S[4]}" stroke-width="1.8"/>\n'
            s += text(x + w / 2, y1 + 30, lab, 9.5, S[4], weight="bold")
        else:
            s += f'<rect x="{x}" y="{y1}" width="{w}" height="52" rx="6" fill="{c}" opacity="0.85"/>\n'
            s += text(x + w / 2, y1 + 30, lab, 10, "#0b0f19", weight="bold")
        s += text(x + w / 2, y1 + 70, sub, 9, S[3] if c else TEXT2)
        x += w + 8

    # ledger
    s += (f'<rect x="130" y="{y1 + 100}" width="700" height="92" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.22)"/>\n')
    s += text(480, y1 + 126, "communication: all-reduce = all-gather + reduce-scatter (post #1) &#8594; SAME bytes, just placed on both sides", 10.5, TEXT)
    s += text(480, y1 + 148, "memory: LN/dropout activations shrink from [b,s,h] to [b,s/N,h] &#8212; the last replicated activations disappear", 10.5, GLOW)
    s += text(480, y1 + 172, "ZeRO split the identity in TIME (update between the halves) &#183; SP splits it in SPACE (TP region between the halves)", 10, TEXT2)
    s += "</svg>\n"
    return s


# ---- fig 2: Ring Attention -------------------------------------------------
def fig2():
    W, H = 960, 520
    s = svg_open(W, H)
    s += text(W / 2, 30, "Ring Attention (context parallelism): Q stays home, KV makes the round trip", 16, TEXT, weight="bold")
    s += text(W / 2, 52, "each rank owns sequence chunk k (its Q_k, K_k, V_k) &#183; KV blocks travel the ring &#183; online softmax keeps exactness", 10.5, TEXT2)

    # 4 GPU boxes in a row with ring arrows
    y0 = 100
    for k in range(4):
        x = 90 + k * 210
        s += (f'<rect x="{x}" y="{y0}" width="170" height="150" rx="10" fill="{SURFACE2}" '
              f'stroke="rgba(0,200,255,0.35)" stroke-width="1.3"/>\n')
        s += text(x + 12, y0 + 20, f"GPU {k}", 10.5, TEXT2, "start")
        # Q resident
        s += f'<rect x="{x + 14}" y="{y0 + 32}" width="60" height="34" rx="4" fill="{S[0]}"/>\n'
        s += text(x + 44, y0 + 53, f"Q{k}", 11, "#0b0f19", weight="bold")
        s += text(x + 44, y0 + 80, "stays", 8.5, TEXT2)
        # KV traveling
        s += f'<rect x="{x + 92}" y="{y0 + 32}" width="60" height="34" rx="4" fill="{S[4]}"/>\n'
        s += text(x + 122, y0 + 53, f"K,V", 11, "#0b0f19", weight="bold")
        s += text(x + 122, y0 + 80, "travels &#8594;", 8.5, S[4])
        # accumulator
        s += f'<rect x="{x + 14}" y="{y0 + 96}" width="138" height="38" rx="4" fill="none" stroke="{S[3]}" stroke-width="1.4"/>\n'
        s += text(x + 83, y0 + 113, "O, m, l accumulate", 9, S[3], weight="bold")
        s += text(x + 83, y0 + 126, "(online softmax)", 8, TEXT2)
        if k < 3:
            s += arrow(x + 174, y0 + 49, x + 206, y0 + 49, color=S[4], width=1.8)
    # wrap-around
    s += (f'<path d="M {90 + 3 * 210 + 172} {y0 + 20} C {90 + 3 * 210 + 230} {y0 - 30}, 30 {y0 - 30}, 86 {y0 + 40}" '
          f'fill="none" stroke="{S[4]}" stroke-width="1.8" marker-end="url(#arr)" opacity="0.85"/>\n')
    s += text(W / 2, y0 - 24, "N&#8722;1 hops &#183; each hop sends [2 &#183; s/N &#183; h] bytes &#183; can overlap with the block compute", 9.5, TEXT2)

    # online softmax box
    yS = 300
    s += (f'<rect x="120" y="{yS}" width="720" height="170" rx="8" fill="{SURFACE2}" '
          f'stroke="rgba(0,200,255,0.22)"/>\n')
    s += text(480, yS + 26, "why chunked softmax is EXACT (not an approximation): carry running (m, l) and rescale", 11, GLOW, weight="bold")
    s += text(480, yS + 56, "m&#8342; = max(m&#8342;&#8331;&#8321;, rowmax(S&#8342;))        S&#8342; = Q K&#8342;&#7488;/&#8730;d", 10.5, TEXT)
    s += text(480, yS + 80, "l&#8342; = l&#8342;&#8331;&#8321;&#183;e^(m&#8342;&#8331;&#8321;&#8722;m&#8342;) + &#931; e^(S&#8342;&#8722;m&#8342;)", 10.5, TEXT)
    s += text(480, yS + 104, "O&#8342; = O&#8342;&#8331;&#8321;&#183;e^(m&#8342;&#8331;&#8321;&#8722;m&#8342;) + e^(S&#8342;&#8722;m&#8342;) V&#8342;        final: O/l", 10.5, TEXT)
    s += text(480, yS + 134, "the same trick that powers FlashAttention&#8217;s tiling &#8212; CP just runs the tiles on different GPUs", 10, TEXT2)
    s += text(480, yS + 154, "attention is the ONLY place tokens interact &#8212; everything else (LN, MLP) is token-local and needs no comm under CP", 10, S[3])
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-megatron-sp", fig1), ("fig-2-ring-attention", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
