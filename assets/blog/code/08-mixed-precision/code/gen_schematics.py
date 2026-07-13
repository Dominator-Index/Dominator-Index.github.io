"""Part 08 schematics: fig-1 float format bit layouts (range vs precision), fig-2 the mixed precision training loop (flow vs state)."""

import os

SURFACE = "#0b0f19"
SURFACE2 = "#131c2e"
TEXT = "#d0dce8"
TEXT2 = "#7a8899"
GLOW = "#00c8ff"
S = ["#0099c4", "#e66767", "#9085e9", "#199e70", "#c98500", "#e14d92"]
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
C_SIGN, C_EXP, C_MAN = S[5], S[4], S[0]  # sign magenta / exponent amber / mantissa cyan


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{MONO}">\n'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n'
            f'<defs>'
            f'<marker id="ag" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{GLOW}"/></marker>'
            f'<marker id="at" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{TEXT2}"/></marker>'
            f'</defs>\n')


def text(x, y, s, size=13, fill=TEXT, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>\n')


def box(x, y, w, h, fill=SURFACE2, stroke=GLOW, sw=1.5, rx=8, opacity=1.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}/>\n')


def arrow(x1, y1, x2, y2, color=GLOW, sw=2, marker="ag"):
    # glow = thick translucent underline (no filter, safe for inkscape)
    out = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
           f'stroke-width="{sw * 3}" opacity="0.18"/>\n')
    out += (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"/>\n')
    return out


def bits_row(x0, y0, name, n_exp, n_man, cell=23):
    """One row of bit layout: 1 sign + n_exp exponent + n_man mantissa. Returns (svg, total width)."""
    s = text(x0 - 14, y0 + 17, name, 14, TEXT, "end", "bold")
    x = x0
    groups = [(1, C_SIGN), (n_exp, C_EXP), (n_man, C_MAN)]
    for n, color in groups:
        for _ in range(n):
            s += (f'<rect x="{x}" y="{y0}" width="{cell - 2}" height="24" rx="3" '
                  f'fill="{color}" opacity="0.85"/>\n')
            x += cell
        x += 5  # gap between groups
    w = x - x0 - 5
    # group labels (sign is in the top legend, not labeled per row)
    xe = x0 + cell + 5
    xm = xe + n_exp * cell + 5
    s += text(xe + (n_exp * cell - 5) / 2, y0 + 40, f"{n_exp} exponent bits &#8594; RANGE", 9.5, C_EXP)
    s += text(xm + (n_man * cell - 5) / 2, y0 + 40, f"{n_man} mantissa bits &#8594; PRECISION", 9.5, C_MAN)
    return s, w


def fig1():
    W, H = 960, 470
    s = svg_open(W, H)
    s += text(W / 2, 30, "The same bits buy range OR precision — not both", 15, TEXT, weight="bold")
    s += text(W / 2, 50, "exponent bits decide how large/small a number can be; mantissa bits decide how many digits survive", 10.5, TEXT2)

    # legend: sign bit
    s += f'<rect x="822" y="40" width="12" height="12" rx="3" fill="{C_SIGN}" opacity="0.85"/>\n'
    s += text(840, 50, "sign bit", 9.5, TEXT2, "start")

    x0, cell = 130, 24
    # fp32
    r, w32 = bits_row(x0, 86, "fp32", 8, 23, cell)
    s += r
    s += text(x0 + w32, 144, "max ~3.4e38 &#183; ~7.2 decimal digits", 10.5, TEXT2, "end")

    # fp16
    r, w16 = bits_row(x0, 176, "fp16", 5, 10, cell)
    s += r
    s += text(x0 + w16 + 16, 193, "max 65504 (!) &#183; min normal 6.1e-5 (!) &#183; ~3.3 digits", 10.5, S[1], "start")

    # bf16
    r, wb = bits_row(x0, 266, "bf16", 8, 7, cell)
    s += r
    s += text(x0 + wb + 16, 283, "max ~3.4e38 (same as fp32) &#183; ~2.4 digits", 10.5, S[3], "start")

    # bf16 = truncated fp32: dashed box around the first 16 bits of fp32
    keep = 1 + 8 + 7
    s += box(x0 - 4, 80, keep * cell + 5 + 5 + 2, 36, fill="none", stroke=GLOW, sw=1.2, rx=6, dash="5 4")
    s += text(x0 + (keep * cell) / 2, 74, "bf16 = fp32 with the last 16 mantissa bits cut off (cast up = pad zeros, lossless)", 9.5, GLOW)

    # bottom: how the two 16-bit formats end up differently
    s += box(60, 330, 840, 108, fill=SURFACE2, stroke=TEXT2, sw=1, rx=8, opacity=0.7)
    s += text(80, 356, "fp16:", 12, S[1], "start", "bold")
    s += text(140, 356, "narrow range — gradients below 6e-5 underflow toward 0, logits above 65504 overflow to inf", 11, TEXT, "start")
    s += text(140, 374, "&#8594; needs dynamic loss scaling (multiply loss by S, divide grads by S, retune S on every inf/nan)", 10, TEXT2, "start")
    s += text(80, 404, "bf16:", 12, S[3], "start", "bold")
    s += text(140, 404, "fp32's range, 2.4 digits of precision — nothing under/overflows that fp32 wouldn't", 11, TEXT, "start")
    s += text(140, 422, "&#8594; no loss scaling; the lost precision is repaid elsewhere (fp32 accumulate + fp32 state, fig. 2)", 10, TEXT2, "start")

    s += "</svg>\n"
    return s


def fig2():
    W, H = 960, 560
    s = svg_open(W, H)
    s += text(W / 2, 30, "One training step in bf16 mixed precision: FLOW runs bf16, STATE stays fp32", 15, TEXT, weight="bold")

    # the two partitions
    s += box(40, 66, 450, 400, fill="rgba(0,153,196,0.05)", stroke=S[0], sw=1.2, rx=10, dash="6 5")
    s += text(70, 92, "FLOW — bf16", 13, S[0], "start", "bold")
    s += text(70, 108, "per-token traffic: cheap &#215; huge volume", 9.5, TEXT2, "start")
    s += box(510, 66, 410, 400, fill="rgba(201,133,0,0.05)", stroke=S[4], sw=1.2, rx=10, dash="6 5")
    s += text(540, 92, "STATE — fp32", 13, S[4], "start", "bold")
    s += text(540, 108, "per-param ledger: tiny additions must survive", 9.5, TEXT2, "start")

    # FLOW column
    s += box(70, 130, 200, 46)
    s += text(170, 149, "bf16 params  2Ψ", 11.5, TEXT, weight="bold")
    s += text(170, 165, "the compute projection", 9, TEXT2)

    s += box(70, 226, 390, 74)
    s += text(265, 246, "forward + backward — every matmul in bf16", 11.5, TEXT, weight="bold")
    s += text(265, 264, "Tensor Core: bf16 &#215; bf16 &#8594; fp32 accumulator Σ &#8594; bf16 out", 10, GLOW)
    s += text(265, 280, "activations stored bf16 &#183; LayerNorm/softmax/loss internally fp32", 9, TEXT2)

    s += box(70, 350, 200, 46)
    s += text(170, 369, "bf16 grads", 11.5, TEXT, weight="bold")
    s += text(170, 385, "what backward writes out", 9, TEXT2)

    s += arrow(170, 176, 170, 224, TEXT2, 1.5, "at")
    s += arrow(170, 300, 170, 348, TEXT2, 1.5, "at")

    # STATE column
    s += box(560, 350, 310, 46)
    s += text(715, 369, "fp32 grad buffer  2Ψ|4Ψ", 11.5, TEXT, weight="bold")
    s += text(715, 385, "micro-batch accumulation adds in fp32", 9, TEXT2)

    s += box(560, 226, 310, 74)
    s += text(715, 246, "optimizer step — all fp32", 11.5, TEXT, weight="bold")
    s += text(715, 264, "Adam m 4Ψ + v 4Ψ; update Δ hits the master", 10, TEXT2)
    s += text(715, 280, "|Δ| ~ 1e-3..1e-5 of |w| — needs 23 mantissa bits", 9, S[1])

    s += box(560, 130, 310, 46)
    s += text(715, 149, "fp32 master params  4Ψ", 11.5, TEXT, weight="bold")
    s += text(715, 165, "the real weights; never do matmuls", 9, TEXT2)

    s += arrow(715, 348, 715, 302, TEXT2, 1.5, "at")
    s += arrow(715, 224, 715, 178, TEXT2, 1.5, "at")

    # the two cast points (crossing the partitions, glow)
    s += arrow(272, 373, 558, 373, GLOW, 2)
    s += text(415, 362, "cast UP — lossless", 10.5, GLOW, weight="bold")
    s += text(415, 390, "pad 16 zero bits", 9, TEXT2)

    s += arrow(558, 153, 272, 153, GLOW, 2)
    s += text(415, 142, "cast DOWN — round to 7 mantissa bits", 10.5, GLOW, weight="bold")
    s += text(415, 170, "regenerated every step; the only lossy hop", 9, TEXT2)

    # bottom ledger
    s += box(60, 486, 840, 52, fill=SURFACE2, stroke=TEXT2, sw=1, rx=8, opacity=0.7)
    s += text(480, 508, "the ledger, finally itemized:  2Ψ bf16 params + 4Ψ master + 4Ψ m + 4Ψ v + grads (bf16 2Ψ | fp32 4Ψ)", 11, TEXT)
    s += text(480, 526, "= 16Ψ or 18Ψ per parameter — post #0's number; which one you pay = which precision holds the gradient accumulation", 10, TEXT2)

    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig-1-float-formats", fig1), ("fig-2-mixed-precision-loop", fig2)]:
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(fn())
        print(f"wrote {name}.svg")
