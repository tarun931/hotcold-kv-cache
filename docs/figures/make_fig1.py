"""Figure 1: architecture of the hot/cold KV cache.

Places every box, arrow and label with python-pptx and writes them as one
editable slide at print size (7.2 x 4.0 in). Open the slide in Keynote or
PowerPoint to adjust it, then export the PDF from there.

Run:  python docs/figures/make_fig1.py
Writes docs/figures/fig1_architecture.pptx
"""

import os

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

W, H = 7.2, 4.0          # slide size, inches
FONT = "Helvetica"

rgb = RGBColor.from_string
INK, GREY, WHITE = rgb("262626"), rgb("6B6B6B"), rgb("FFFFFF")
GPU_FILL, GPU_LINE = rgb("EDF3FA"), rgb("8EAED3")
CPU_FILL, CPU_LINE = rgb("FBF3E6"), rgb("D9B072")
TOPK, RECENT = rgb("AFC8E6"), rgb("5E8FC8")
EVICT, ASYNC = rgb("C8553D"), rgb("23877D")
STEP, CPU_BAR, COPY_BAR = rgb("DCE6F2"), rgb("E9C98F"), rgb("9EBDE2")
DASH = MSO_LINE_DASH_STYLE.DASH
L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT
TOP, MID = MSO_ANCHOR.TOP, MSO_ANCHOR.MIDDLE


# ---------------------------------------------------------------- primitives

def _style_line(line, color, width, dash):
    if color is None:
        line.fill.background()
        return
    line.color.rgb = color
    line.width = Pt(width)
    if dash is not None:
        line.dash_style = dash


def _write(tf, lines, size, color, align, anchor, margin):
    """lines: str or (str, {size, bold, italic, color}) per paragraph."""
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(min(margin, 0.02))
    for i, item in enumerate(lines):
        text, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = text
        f = run.font
        f.name = FONT
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", False)
        f.italic = opt.get("italic", False)
        f.color.rgb = opt.get("color", color)


def box(sl, x, y, w, h, lines=(), fill=WHITE, line=INK, lw=0.75, dash=None,
        rounded=True, size=6, color=INK, align=C, anchor=MID, margin=0.03):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = sl.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = min(0.5, 0.05 / min(w, h))   # ~0.05 in corner radius
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    _style_line(shp.line, line, lw, dash)
    _write(shp.text_frame, lines, size, color, align, anchor, margin)
    return shp


def text(sl, x, y, w, h, lines, size=6, color=INK, align=C, anchor=MID, margin=0.0):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _write(tb.text_frame, lines, size, color, align, anchor, margin)
    return tb


def arrow(sl, x1, y1, x2, y2, color=INK, lw=0.9, dash=None, both=False):
    con = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                  Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _style_line(con.line, color, lw, dash)
    ln = con.line._get_or_add_ln()
    for tag in (["a:headEnd"] if both else []) + ["a:tailEnd"]:
        end = etree.SubElement(ln, qn(tag))
        end.set("type", "triangle")
        end.set("w", "med")
        end.set("len", "med")
    return con


def b(s):  # bold first line of a box
    return (s, {"bold": True, "size": 7})


# -------------------------------------------------------------------- figure

def draw(sl):
    # regions
    gx0, gx1 = 0.08, 3.60            # GPU
    cx0, cx1 = 4.50, 7.12            # CPU
    ry0, ry1 = 0.08, 2.90
    box(sl, gx0, ry0, gx1 - gx0, ry1 - ry0, fill=GPU_FILL, line=GPU_LINE, rounded=False)
    box(sl, cx0, ry0, cx1 - cx0, ry1 - ry0, fill=CPU_FILL, line=CPU_LINE, rounded=False)
    text(sl, gx0 + 0.08, 0.10, 1.0, 0.17, [("GPU", {"bold": True, "size": 8})], align=L)
    text(sl, cx0 + 0.08, 0.10, 1.4, 0.17, [("CPU (host memory)", {"bold": True, "size": 8})], align=L)
    text(sl, gx1, 0.10, cx0 - gx1, 0.17, ["PCIe"], size=6.5, color=GREY)

    # columns and rows
    px0, px1 = 0.20, 1.38            # attention pipeline
    mx0, mx1 = 1.80, 3.46            # GPU per-layer state
    kx0, kx1 = 4.62, 7.00            # CPU per-layer state
    r1 = (0.48, 0.96)                # query row
    r2 = (1.10, 1.74)                # hot/cold tier row
    r3 = (1.86, 2.30)                # retrieved-KV row
    mid = lambda r: (r[0] + r[1]) / 2
    ital = lambda s: (s, {"italic": True, "color": GREY, "size": 6})

    text(sl, px0, 0.28, px1 - px0, 0.15, [ital("Patched attention (×32 layers)")])
    text(sl, mx0, 0.28, mx1 - mx0, 0.15, [ital("Per-layer GPU state")])
    text(sl, kx0, 0.28, kx1 - kx0, 0.15, [ital("Per-layer CPU state")])

    # attention pipeline
    pw = px1 - px0
    box(sl, px0, r1[0], pw, r1[1] - r1[0],
        [b("Q, K, V projections"), "+ RoPE (pre-RoPE q, k kept)"])
    box(sl, px0, 1.14, pw, 0.56,
        [b("Hot-tier update"), "K, V + pre-RoPE K, pos", "compress when full"])
    box(sl, px0, r3[0], pw, r3[1] - r3[0],
        [b("Inject retrieved KV"), "RoPE re-applied at", "original positions"])
    box(sl, px0, 2.42, pw, 0.42, [b("Attention"), "Triton kernel (decode)", "SDPA (prefill)"])
    pc = (px0 + px1) / 2
    arrow(sl, pc, r1[1], pc, 1.14)
    arrow(sl, pc, 1.70, pc, r3[0])
    arrow(sl, pc, r3[1], pc, 2.42)

    # GPU state
    mw = mx1 - mx0
    box(sl, mx0, r1[0], mw, r1[1] - r1[0],
        [b("Query pool"), "last 4 decode queries (pre-RoPE)"])
    box(sl, mx0, r2[0], mw, r2[1] - r2[0], [b("Hot tier: KV cache ≤ budget B")],
        anchor=TOP)
    bx, by, bh = mx0 + 0.10, 1.37, 0.17          # hot-tier occupancy bar
    seg = [(0.80, TOPK, "top-k by score", INK, None),
           (0.30, RECENT, "recent", WHITE, None),
           (0.36, None, "free", GREY, DASH)]
    x = bx
    for w, fill, lab, col, dash in seg:
        box(sl, x, by, w, bh, fill=fill, line=GREY if dash else None, lw=0.5, dash=dash,
            rounded=False)
        text(sl, x, by, w, bh, [lab], size=6, color=col)
        x += w
    text(sl, mx0, 1.55, mw, 0.15, ["compressed to 0.75 B when full"], size=6, color=GREY)
    box(sl, mx0, r3[0], mw, r3[1] - r3[0],
        [b("Staged retrieved KV"), "side CUDA stream + ready event"])

    arrow(sl, px1, mid(r1), mx0, mid(r1))
    text(sl, px1, mid(r1) - 0.15, mx0 - px1, 0.13, ["query"], size=6)
    arrow(sl, px1, mid(r2), mx0, mid(r2), both=True)
    text(sl, px1, mid(r2) - 0.15, mx0 - px1, 0.13, ["K, V"], size=6)
    arrow(sl, mx0, mid(r3), px1, mid(r3))
    text(sl, px1, mid(r3) - 0.15, mx0 - px1, 0.13, ["if ready"], size=6)

    # CPU state
    kw = kx1 - kx0
    box(sl, kx0, r1[0], kw, r1[1] - r1[0],
        [b("Retrieval thread"), "launched every 16 decode steps"])
    box(sl, kx0, r2[0], kw, r2[1] - r2[0], [b("Cold tier")], anchor=TOP)
    ring = (kx0 + 0.08, 5.70)
    faiss = (5.90, kx1 - 0.08)
    sy0, sy1 = 1.29, 1.68
    box(sl, ring[0], sy0, ring[1] - ring[0], sy1 - sy0,
        [("Ring buffer", {"bold": True}), "32,768 tokens", "pre-RoPE K, V, pos"], lw=0.5)
    box(sl, faiss[0], sy0, faiss[1] - faiss[0], sy1 - sy0,
        [("FAISS IndexFlatIP", {"bold": True}), "unit-norm", "pre-RoPE keys"], lw=0.5)
    box(sl, kx0, r3[0], kw, r3[1] - r3[0],
        [b("Pinned staging buffer"), "pre-RoPE K, V, pos, valid mask"])

    fc = (faiss[0] + faiss[1]) / 2
    rc = (ring[0] + ring[1]) / 2
    arrow(sl, fc, r1[1], fc, sy0, color=ASYNC, dash=DASH)
    text(sl, fc + 0.05, r1[1] + 0.02, 0.8, 0.13, ["search top-64"], size=6, align=L)
    arrow(sl, faiss[0], (sy0 + sy1) / 2, ring[1], (sy0 + sy1) / 2, color=ASYNC, dash=DASH)
    text(sl, ring[1], sy0 - 0.01, faiss[0] - ring[1], 0.12, ["ids"], size=5.5)
    arrow(sl, rc, sy1, rc, r3[0], color=ASYNC, dash=DASH)
    text(sl, rc + 0.05, sy1 + 0.03, 0.6, 0.13, ["gather"], size=6, align=L)

    # across PCIe
    gap0, gap1 = mx1, kx0
    lab_x, lab_w = gx1 - 0.08, (cx0 - gx1) + 0.16

    def across(y, top, bottom, color, dash, leftward=False):
        if leftward:
            arrow(sl, gap1, y, gap0, y, color=color, dash=dash, lw=1.1)
        else:
            arrow(sl, gap0, y, gap1, y, color=color, dash=dash, lw=1.1)
        text(sl, lab_x, y - 0.16, lab_w, 0.14, [top], size=6)
        text(sl, lab_x, y + 0.02, lab_w, 0.14, [bottom], size=6, color=GREY)

    across(mid(r1), "pooled query", "every 16 steps", ASYNC, DASH)
    across(mid(r2), "evicted tokens", "pre-RoPE K, V, pos", EVICT, None)
    across(mid(r3), "top-64 tokens", "async copy", ASYNC, DASH, leftward=True)

    # decode timeline (schematic)
    text(sl, 0.10, 2.98, 4.0, 0.15,
         [("Decode timeline for one layer (schematic)", {"italic": True, "color": GREY})],
         align=L)
    lanes = [("GPU main stream", 3.30), ("CPU retrieval thread", 3.52), ("GPU side stream", 3.74)]
    lane_h = 0.14
    for name, y in lanes:
        text(sl, 0.10, y, 1.22, lane_h, [name], size=6, align=R)
    t0, pitch, sq = 1.45, 0.31, 0.27
    xs = lambda k: t0 + k * pitch
    y_main, y_cpu, y_copy = (y for _, y in lanes)
    for k in range(18):
        launch, inject = k in (0, 16), k == 2
        box(sl, xs(k), y_main, sq, lane_h, fill=ASYNC if inject else STEP,
            line=INK if launch else None, lw=0.75, rounded=False)
    for k, lab in ((0, "t: launch"), (2, "inject when ready"), (16, "t+16: launch")):
        text(sl, xs(k) + sq / 2 - 0.40, 3.15, 0.80, 0.13, [lab], size=6)
    for k in (0, 16):
        box(sl, xs(k), y_cpu, pitch + sq, lane_h, fill=CPU_BAR, line=None, rounded=False)
    text(sl, xs(2), y_cpu, 2.6, lane_h, ["FAISS search + gather into pinned buffer"],
         size=6, align=L)
    box(sl, xs(1), y_copy, sq, lane_h, fill=COPY_BAR, line=None, rounded=False)
    text(sl, xs(2), y_copy, 2.6, lane_h, ["host-to-device copy, overlapped with decode"],
         size=6, align=L)
    text(sl, xs(10), y_copy, xs(17) + sq - xs(10), lane_h,
         [("decode never waits for the search or the copy",
           {"italic": True, "color": GREY})], size=6, align=R)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    draw(slide)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_architecture.pptx")
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
