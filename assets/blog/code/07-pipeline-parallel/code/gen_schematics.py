"""第 07 篇示意图:三种流水线调度的时间线(p=4)。1F1B 槽位按依赖约束精确推导。"""

import os

SURFACE = "#0b0f19"
SURFACE2 = "#131c2e"
TEXT = "#d0dce8"
TEXT2 = "#7a8899"
GLOW = "#00c8ff"
S = ["#0099c4", "#e66767", "#9085e9", "#199e70", "#c98500", "#e14d92"]
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
CF, CB = S[0], S[2]  # forward 青 / backward 紫


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{MONO}">\n'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n')


def text(x, y, s, size=13, fill=TEXT, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>\n')


def slot(x0, y0, t, s_row, kind, idx, sw=42, sh=30):
    x, y = x0 + t * sw, y0 + s_row * (sh + 4)
    c = CF if kind == "F" else CB
    out = f'<rect x="{x}" y="{y}" width="{sw - 3}" height="{sh}" rx="3" fill="{c}" opacity="0.92"/>\n'
    out += text(x + sw / 2 - 1.5, y + sh / 2 + 4, f"{kind}{idx}", 10, "#0b0f19", weight="bold")
    return out


def panel(s, x0, y0, title, sched, span, note, p=4, sw=42):
    s_ = text(x0 + span * sw / 2, y0 - 12, title, 12, TEXT, weight="bold")
    for r in range(p):
        s_ += text(x0 - 12, y0 + r * 34 + 20, f"stage {r}", 9.5, TEXT2, "end")
        s_ += f'<rect x="{x0}" y="{y0 + r * 34}" width="{span * sw - 3}" height="30" rx="3" fill="{SURFACE2}" opacity="0.6"/>\n'
    for (t, r, kind, idx) in sched:
        s_ += slot(x0, y0, t, r, kind, idx, sw=sw)
    s_ += text(x0 + span * sw / 2, y0 + 4 * 34 + 16, note, 9.5, TEXT2)
    return s + s_


def fig1():
    W, H = 960, 700
    s = svg_open(W, H)
    s += text(W / 2, 28, "Three pipeline schedules, same 4 stages (F = forward, B = backward of one microbatch)", 14.5, TEXT, weight="bold")
    s += text(W / 2, 48, "grey = idle = the bubble &#183; equal slot widths for clarity (real backward &#8776; 2&#215; forward)", 10.5, TEXT2)

    x0 = 120
    # (a) naive m=1
    sched = [(t, t, "F", 1) for t in range(4)] + [(4 + t, 3 - t, "B", 1) for t in range(4)]
    s = panel(s, x0, 92, "(a) naive model parallel (m = 1): one GPU works, three watch", sched, 8,
              "bubble = 6/8 = (p&#8722;1)/(m+p&#8722;1) with m=1 &#8594; 75% idle", sw=52)

    # (b) GPipe m=4
    sched = []
    m, p = 4, 4
    for i in range(m):
        for r in range(p):
            sched.append((i + r, r, "F", i + 1))
    for i in range(m):
        for r in range(p):
            sched.append((m + p - 1 + (m - 1 - i) + (p - 1 - r), r, "B", i + 1))
    s = panel(s, x0, 280, "(b) GPipe (m = 4): all forwards, then all backwards", sched, 14,
              "bubble = 6/14 = (p&#8722;1)/(m+p&#8722;1) &#8776; 43% &#183; must hold ALL m activations &#8594; memory O(m)")

    # (c) 1F1B m=4(槽位按依赖精确推导)
    sched_1f1b = {
        0: [("F", 1, 0), ("F", 2, 1), ("F", 3, 2), ("F", 4, 3), ("B", 1, 7), ("B", 2, 9), ("B", 3, 11), ("B", 4, 13)],
        1: [("F", 1, 1), ("F", 2, 2), ("F", 3, 3), ("F", 4, 4), ("B", 1, 6), ("B", 2, 8), ("B", 3, 10), ("B", 4, 12)],
        2: [("F", 1, 2), ("F", 2, 3), ("B", 1, 5), ("F", 3, 6), ("B", 2, 7), ("F", 4, 8), ("B", 3, 9), ("B", 4, 11)],
        3: [("F", 1, 3), ("B", 1, 4), ("F", 2, 5), ("B", 2, 6), ("F", 3, 7), ("B", 3, 8), ("F", 4, 9), ("B", 4, 10)],
    }
    sched = [(t, r, k, i) for r, ops in sched_1f1b.items() for (k, i, t) in ops]
    s = panel(s, x0, 468, "(c) 1F1B (m = 4): interleave one-forward-one-backward after warm-up", sched, 14,
              "SAME bubble as GPipe &#183; but at most p microbatches in flight &#8594; memory O(p), independent of m", )

    s += text(W / 2, 672, "the bubble is geometry, not implementation: it shrinks only by raising m (or interleaved/zero-bubble schedules, see text)", 10.5, GLOW)
    s += "</svg>\n"
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fig-1-schedules.svg"), "w") as f:
        f.write(fig1())
    print("wrote fig-1-schedules")
