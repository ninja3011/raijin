import sys, os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FIG, LAY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NAVY, DEEP, MID, SKY, LIGHT, PALE, INK, GREY = "0B2545", "13315C", "1F6FB2", "4EA8DE", "A9D6F5", "E7F1FB", "1B2533", "6B7686"
rgb = lambda h: RGBColor.from_string(h)
FONT = "Calibri"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
TEXT_W = Cm(17)

st = doc.styles["Normal"]
st.font.name = FONT; st.font.size = Pt(10.5); st.font.color.rgb = rgb(INK)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.12

def set_font(style, size, color, bold=True, before=0, after=4):
    s = doc.styles[style]
    s.font.name = FONT; s.font.size = Pt(size); s.font.bold = bold; s.font.color.rgb = rgb(color); s.font.italic = False
    rpr = s.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(a), FONT)
    s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True
set_font("Heading 1", 20, NAVY, before=0, after=8)
set_font("Heading 2", 13.5, MID, before=12, after=4)
set_font("Heading 3", 11.5, DEEP, before=8, after=2)

def shade(el, fill):
    pr = el.get_or_add_tcPr() if el.tag.endswith("tc") else el
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    pr.append(shd)

def cell_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None: b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, (sz, color) in kw.items():
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single" if sz else "nil")
        if sz: e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), color)
        b.append(e)

def table_borders(t, none=True):
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); b.append(e)
    tblPr.append(b)

def cell_margins(t, top=70, bottom=70, left=110, right=110):
    tblPr = t._tbl.tblPr; m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)

def fix_grid(t, widths):
    tbl = t._tbl; tblPr = tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    tw = tblPr.find(qn("w:tblW"))
    if tw is None: tw = OxmlElement("w:tblW"); tblPr.append(tw)
    tw.set(qn("w:w"), str(int(sum(widths) * 567))); tw.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for i, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths[i] * 567)))

def keep_row(row):
    trPr = row._tr.get_or_add_trPr(); e = OxmlElement("w:cantSplit"); trPr.append(e)

def header_repeat(row):
    trPr = row._tr.get_or_add_trPr(); e = OxmlElement("w:tblHeader"); trPr.append(e)

def runs(p, text, size=None, color=None, bold=None, italic=None):
    """text with **bold** segments"""
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if not seg: continue
        r = p.add_run(seg)
        if i % 2 == 1: r.bold = True
        elif bold is not None: r.bold = bold
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = rgb(color)
        if italic: r.italic = True
    return p

def para(text, after=6, size=None, color=None, align=None, italic=None):
    p = doc.add_paragraph(); runs(p, text, size=size, color=color, italic=italic)
    p.paragraph_format.space_after = Pt(after)
    if align: p.alignment = align
    return p

def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet"); runs(p, it)
        p.paragraph_format.space_after = Pt(3); p.paragraph_format.left_indent = Cm(0.6)

def numbered(items):
    for it in items:
        p = doc.add_paragraph(style="List Number"); runs(p, it)
        p.paragraph_format.space_after = Pt(3)

def h1(num, text, newpage=True):
    if newpage: doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE); doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    # kicker
    k = doc.add_paragraph(); r = k.add_run(f"SECTION {num}" if num else ""); r.font.size = Pt(9); r.bold = True; r.font.color.rgb = rgb(SKY)
    rpr = r._r.get_or_add_rPr(); sp = OxmlElement("w:spacing"); sp.set(qn("w:val"), "40"); rpr.append(sp)
    k.paragraph_format.space_after = Pt(0); k.paragraph_format.keep_with_next = True
    h = doc.add_heading(text, level=1)
    pPr = h._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "18"); bt.set(qn("w:space"), "6"); bt.set(qn("w:color"), MID)
    bd.append(bt); pPr.append(bd)
    return h

def h2(t): return doc.add_heading(t, level=2)
def h3(t): return doc.add_heading(t, level=3)

FIGN = [0]
def figure(name, caption, width=16.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True; p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(name if os.path.isabs(name) else os.path.join(FIG, name), width=Cm(width))
    FIGN[0] += 1
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {FIGN[0]}. "); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = rgb(MID)
    r = c.add_run(caption); r.font.size = Pt(9); r.font.color.rgb = rgb(GREY); r.italic = True
    c.paragraph_format.space_after = Pt(10)

TABN = [0]
def table(header, rows, widths, caption=None, highlight=(), align_right_from=None, font=9.5, bold_first=False):
    if caption:
        TABN[0] += 1
        c = doc.add_paragraph(); c.paragraph_format.keep_with_next = True; c.paragraph_format.space_before = Pt(6); c.paragraph_format.space_after = Pt(3)
        r = c.add_run(f"Table {TABN[0]}. "); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = rgb(MID)
        r = c.add_run(caption); r.font.size = Pt(9); r.font.color.rgb = rgb(GREY)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    table_borders(t); cell_margins(t)
    for ri, row in enumerate([header] + rows):
        tr = t.rows[ri]; keep_row(tr)
        if ri == 0: header_repeat(tr)
        for ci, val in enumerate(row):
            cell = tr.cells[ci]; cell.width = Cm(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
            if ri == 0:
                shade(cell._tc, NAVY); runs(p, str(val), size=font, color="FFFFFF", bold=True)
            else:
                hl = (ri - 1) in highlight
                shade(cell._tc, LIGHT if hl else (PALE if ri % 2 == 0 else "FFFFFF"))
                runs(p, str(val), size=font, color=NAVY if hl else INK, bold=True if (hl or (bold_first and ci == 0)) else None)
                cell_borders(cell, bottom=(4, "C9DDF2"))
            if align_right_from is not None and ci >= align_right_from: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ci, w in enumerate(widths):
        for row in t.rows: row.cells[ci].width = Cm(w)
    fix_grid(t, widths)
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4); sp.paragraph_format.line_spacing = 0.6
    return t

def callout(title, text, fill=PALE, bar=MID):
    t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER; table_borders(t); cell_margins(t, 120, 120, 200, 160)
    c = t.rows[0].cells[0]; c.width = TEXT_W; fix_grid(t, [17]); shade(c._tc, fill); cell_borders(c, left=(36, bar))
    p = c.paragraphs[0]; runs(p, title, size=10.5, color=bar, bold=True); p.paragraph_format.space_after = Pt(2)
    p2 = c.add_paragraph(); runs(p2, text, size=10.5, color=INK); p2.paragraph_format.space_after = Pt(0)
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4)

def tiles(items, dark=False):
    t = doc.add_table(rows=1, cols=len(items)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; table_borders(t); cell_margins(t, 140, 140, 120, 120)
    w = 17.0 / len(items); fix_grid(t, [w] * len(items))
    for i, (big, small) in enumerate(items):
        c = t.rows[0].cells[i]; c.width = Cm(w)
        shade(c._tc, NAVY if (dark or i % 2 == 0) else MID)
        cell_borders(c, left=(24, "FFFFFF"), right=(24, "FFFFFF"))
        p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
        r = p.add_run(big); r.bold = True; r.font.size = Pt(20); r.font.color.rgb = rgb("FFFFFF")
        p2 = c.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER; p2.paragraph_format.space_after = Pt(0)
        r = p2.add_run(small); r.font.size = Pt(9); r.font.color.rgb = rgb(LIGHT)
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(6)

# ---------- footer ----------
def page_field(p):
    r = p.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r._r.append(f1)
    r2 = p.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"; r2._r.append(it)
    r3 = p.add_run(); f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end"); r3._r.append(f2)
    for x in (r, r2, r3): x.font.size = Pt(9); x.font.color.rgb = rgb(MID); x.bold = True
sec.different_first_page_header_footer = True
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = fp.add_run("RAIJIN  ·  MXINT8 attention accelerator  ·  sky130  ·  page "); r.font.size = Pt(9); r.font.color.rgb = rgb(GREY)
page_field(fp)
pPr = fp._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bt = OxmlElement("w:top")
bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "6"); bt.set(qn("w:space"), "4"); bt.set(qn("w:color"), LIGHT); bd.append(bt); pPr.append(bd)

# ---------- cover ----------
t = doc.add_table(rows=1, cols=1); table_borders(t); cell_margins(t, 500, 420, 480, 400)
c = t.rows[0].cells[0]; c.width = TEXT_W; fix_grid(t, [17]); shade(c._tc, NAVY)
p = c.paragraphs[0]; r = p.add_run("AI HARDWARE ARCHITECTURES  ·  IIT BOMBAY  ·  FINAL PROJECT"); r.font.size = Pt(9); r.bold = True; r.font.color.rgb = rgb(SKY)
p = c.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
r = p.add_run("Raijin"); r.font.size = Pt(46); r.bold = True; r.font.color.rgb = rgb("FFFFFF")
p = c.add_paragraph(); p.paragraph_format.space_after = Pt(16)
r = p.add_run("An MX-format attention accelerator with a decode-oriented KV cache for edge LLM inference"); r.font.size = Pt(16); r.font.color.rgb = rgb(LIGHT)
p = c.add_paragraph(); p.paragraph_format.space_after = Pt(0)
r = p.add_run("Ninad Jangle   ·   November 2026   ·   RTL to GDS on SkyWater 130 nm with open tools"); r.font.size = Pt(10.5); r.font.color.rgb = rgb("FFFFFF")
doc.add_paragraph().paragraph_format.space_after = Pt(2)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
p.add_run().add_picture(os.path.join(LAY, "raijin_chip_final6_full.png"), width=Cm(11.2))
c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = c.add_run("The sign-off layout, chip_p2_final6: four 1 KB SRAM macros (left) and the attention core (right) on a 2.2 × 2.2 mm die.")
r.font.size = Pt(9); r.italic = True; r.font.color.rgb = rgb(GREY)
tiles([("50 MHz", "met at the typical corner"), ("0", "bit mismatches in 29 tests"), ("+0.07%", "perplexity for the arithmetic"), ("67 GOPS/W", "gate-level, post-layout")])

# ---------- abstract ----------
h1("", "Abstract")
para("Raijin is an attention accelerator for running small language models on edge devices. It was taken from RTL to a post-layout, LVS-clean layout in SkyWater 130 nm, using only open-source tools.")
para("One datapath of MXINT8 dot-product units and base-2 softmax lanes runs three dataflows under one controller: unfused attention, fused streaming attention (the FlashAttention recurrence in hardware) and KV-cache decode.")
para("Around that datapath sit the features that text generation needs and earlier accelerators lack: causal prefill, grouped-query and speculative decode, a paged KV cache with per-page 8- or 4-bit precision, attention sinks, RoPE on write, exact key skipping and a heavy-hitter eviction tracker.")
tiles([("+0.07%", "perplexity, chip arithmetic"), ("4.1×", "faster prefill"), ("4.0×", "faster decode"), ("5.3×", "fewer value reads")], dark=False)
para("Every block is bit-exact against a Python golden model. On a pretrained model (TinyStories-8M) the cost is +4.3% as is, and −0.01% once a calibrated key mean is folded into the K bias, with no hardware change. The complete P = 2 chip meets 50 MHz at the typical corner with DRC and LVS clean.")

# ---------- 1 intro ----------
h1("1", "Introduction")
para("On an edge device, generating text is limited by attention over the KV cache, not by matrix multiplies. Each new token reads every cached key and value for every head, so attention cost grows with context length while the weights stay fixed.")
para("The best-known attention accelerators (A3, SpAtten, ELSA) target encoder models such as BERT, where a whole sentence arrives at once. None of them has a KV cache, a decode mode or a positional-encoding path, and their RTL and flows are not public.")
callout("The question this project asks", "What should an attention engine designed from the start for edge decoding contain, and can it be built and verified all the way to layout with open tools?")
h2("Contributions")
table(["#", "Contribution", "What it gives"], [
    ["1", "**One datapath, three dataflows**", "Unfused, fused streaming and decode share the same units; MAC arrays 99–100% busy at 64+ tokens"],
    ["2", "**MX block arithmetic end to end**", "+0.04% to +0.15% perplexity; exponents become shift amounts, no calibration"],
    ["3", "**A KV cache built for decode**", "Causal prefill, GQA and speculative decode, sinks, paging, 8/4-bit per page, eviction tracker"],
    ["4", "**Exact key skipping**", "Max-first decode skips 64–69% of value reads for +0.05% perplexity"],
    ["5", "**RoPE on write**", "Positions applied as keys enter the cache, with no extra pass"],
    ["6", "**A complete open chip**", "QSPI host port, clock gating, sky130 layout DRC and LVS clean"],
], [0.8, 5.2, 11.0], caption="What Raijin contributes.")

# ---------- 2 background ----------
h1("2", "Background and related work")
h2("2.1 What attention computes")
para("Each token produces a **query** (what it is looking for), a **key** (what it contains) and a **value** (what it passes on). A query is scored against every key, softmax turns the scores into weights that sum to 1, and the output is the weighted average of the values:")
p = para("O = softmax(Q·Kᵀ / √d) · V", size=15, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER); p.runs[0].bold = True
para("Done naively, this stores the whole N × N score matrix and makes extra passes for the row maximum and the sum. That memory traffic, not the multiplies, dominates the cost.")
h2("2.2 Online softmax and FlashAttention")
para("Online softmax keeps a running maximum m and sum l per query row. When a new key raises the maximum, the partial sum and output are rescaled by exp(m_old − m_new). FlashAttention uses this to keep the score matrix on chip.")
para("Raijin runs the same recurrence as a hardware stream. Keys flow past a group of query rows, the exponent is base 2 (log₂e/√d is folded into Q on the host), and the division by l is one reciprocal per row at the end.")
h2("2.3 Microscaling (MX) formats")
para("In the OCP MX formats, a block of elements shares one power-of-two scale. Raijin uses MXINT8 with 16-element blocks: each row is one shared 8-bit exponent plus 16 signed 8-bit integers. A dot product is then an integer dot product plus an exponent add.")
h2("2.4 KV-cache techniques Raijin builds in")
table(["Technique", "Idea", "Raijin feature"], [
    ["Multi-/grouped-query attention", "Several query heads share one KV head", "GQA decode (4)"],
    ["StreamingLLM", "A few sink tokens plus a sliding window", "Attention sinks (7)"],
    ["PagedAttention (vLLM)", "Cache managed in pages through a page table", "Paged KV (10)"],
    ["KV quantisation (KIVI)", "Keys and values stored in 2–4 bits", "MXINT4 / MXFP4 KV (6, 14, 16)"],
    ["RoPE", "Positions as rotations of element pairs", "RoPE on write (9)"],
    ["Speculative decoding", "Verify several draft tokens in one pass", "Speculative verify (4)"],
    ["H2O", "Evict the keys with the least total attention", "Heavy-hitter tracker (15)"],
], [4.6, 7.4, 5.0], caption="LLM serving techniques and where they live in Raijin (feature numbers from section 4).")
h2("2.5 Prior attention accelerators")
table(["Design", "Idea", "Node, status", "KV / decode"], [
    ["A3 (HPCA 2020)", "Approximate search for keys likely to score high", "40 nm, synthesis", "No"],
    ["SpAtten (HPCA 2021)", "Cascade token and head pruning, top-k engine", "40 nm, synthesis", "No"],
    ["ELSA (ISCA 2021)", "Hash-based filter before the dot product", "40 nm, post-layout", "No"],
    ["Raijin (this work)", "Exact MX arithmetic, fused online softmax, decode KV cache", "sky130, post-layout, open", "Yes"],
], [3.6, 7.0, 3.9, 2.5], caption="Prior work. Raijin computes every score exactly and only skips keys whose weight is already zero.", highlight=(3,))

# ---------- 3 architecture ----------
h1("3", "Architecture")
para("Raijin is three reusable blocks (a dot-product array, an accumulate array and a softmax unit) around four SRAM banks, with one controller. It is parameterised by N = 16 keys per pass, d = 16 and P query rows in parallel (2 or 4).")
figure("fig_arch.png", "The datapath. K streams from SRAM port 1 into the DPU array and V from port 0 into the accumulators; the dashed box is the streaming compute path.")
h2("3.1 Number format")
para("Every row of Q, K, V and O is MXINT8. Scores leave the dot-product unit as integers whose exponent is e_q + e_k, and a requantiser aligns them for the softmax lanes. Exponents are only ever added or used as shift amounts.")
h2("3.2 The key idea: base-2 online softmax with shifts")
figure("fig_softmax.png", "One key step in a softmax lane. Because the running maximum is an integer, every rescale is a bit shift.")
bullets([
    "**Integer maximum.** m is rounded up to an integer, so rescaling by 2^(m_old − m_new) is a right shift, not a multiply.",
    "**Weights from a table.** 2^(s − m) is a 16-entry lookup, or one or two powers of two in the multiplier-free MF1 and MF2 modes.",
    "**Shift-add P·V.** In MF2, p·v becomes shift-and-add, for +0.07% perplexity. The one division per row happens at the end.",
])
h2("3.3 Blocks")
table(["Block", "What it does"], [
    ["Dot-product array", "P units of 16 INT8 MACs; each cycle one key row meets P query rows held in flops"],
    ["Softmax lanes", "One per query row: running max m, sum l (14-bit Q6.8), weight and rescale shift"],
    ["Accumulate array", "P × 16 lanes adding p·V into the partial output and shifting it when m moves"],
    ["Output stage", "One reciprocal 1/l per row, a 24 × 13 multiply and requantisation back to MXINT8"],
    ["Memory", "Four OpenRAM 1 KB macros; together one 128-bit row per port per cycle"],
], [4.0, 13.0], caption="The datapath blocks. Peak rate is 32·P MACs per cycle.", bold_first=True)
h2("3.4 Three dataflows")
figure("fig_cycles.png", "Cycles per pass. Fusing the softmax into the key stream gives 3.7× fewer cycles than the textbook baseline.", width=14.5)
bullets([
    "**Unfused:** scores to a buffer, batch softmax, then P·V. The baseline.",
    "**Fused:** keys stream past P query rows with online softmax; the score matrix is never stored. Group pipelining overlaps one group's output with the next group's keys.",
    "**Decode:** one new query against the cached keys and values, the step a language model repeats for every token.",
])
para("A single 19-state FSM drives all three. The host reaches every buffer and register through one byte-wide bus, which the chip top bridges to SPI or QSPI.")

# ---------- 4 features ----------
h1("4", "Features for LLM inference")
para("Each feature was added with its RTL, golden model and a bit-exact test in one commit, with a measured result. Features 6 to 10 and 14 add no cycles, because they sit on the addresses or the write path.")
table(["#", "Feature", "Measured result"], [
    ["1", "Timing closure (register duplication)", "fmax 42.1 → 44.8 MHz, wirelength −20%"],
    ["2", "Group pipelining", "Fused pass 357 → 173 cycles at P = 2"],
    ["3", "Causal prefill", "97 cycles at P = 4, against about 680 from decode steps"],
    ["4", "GQA / MQA and speculative decode", "4 query rows in 53 cycles, against 200"],
    ["6", "MXINT4 KV cache", "32 tokens in the SRAM that held 16"],
    ["7", "Attention sinks", "48 tokens through a 16-row cache, cosine 0.99979"],
    ["8", "Key skip and V-read gating", "23–39% of V reads saved"],
    ["9", "RoPE on write", "Cosine 0.9989–0.9999 against FP32 RoPE"],
    ["10", "Paged KV cache", "Two sequences share a prompt page, bit-exact"],
    ["11", "Chip top: SPI and clock gating", "Core clock off 90.8% of a decode loop"],
    ["12", "QSPI", "Q/K/V load 1,069 → 286 µs"],
    ["13", "Token window", "36.2 µs per token, about 27,700 tokens/s"],
    ["14", "Per-page mixed precision", "Cosine 0.966 → 0.991"],
    ["15", "Heavy-hitter tracker", "Within 0.25% of exact H2O; decode 50 → 64 cycles when on"],
    ["16", "MXFP4 KV elements", "All-4-bit KV cost +3.33% → +2.61%"],
    ["17", "Max-first decode", "V reads 86% → 36%, for +0.06% perplexity"],
], [1.0, 6.4, 9.6], caption="The LLM features. Feature 5 (a P = 4 layout on a smaller die) is future work.", font=9.2)
figure("fig_ablation.png", "What the features buy on one GQA layer (8 query heads, 2 KV heads, d = 64, 512-token prompt, P = 4).", width=14.5)
h2("4.1 Why each feature matters")
bullets([
    "**Causal prefill and GQA decode** give the speed: prefill becomes one masked pass, and decode serves a whole KV group in the time of one head.",
    "**4-bit values, mixed pages and key skip** shrink the KV cache and its memory traffic, which is what grows with context.",
    "**Sinks and paging** let a small on-chip cache serve long or shared contexts without copying data.",
    "**QSPI and the token window** make the host link fast enough that the chip, not the bus, sets the token rate.",
])
h2("4.2 Max-first decode")
para("Normal decode judges each key against the running maximum, so a key that arrives before the strongest one is read and then rescaled away. Max-first decode scores every key first (pass A), then visits only keys whose weight is nonzero (pass B).")
figure("fig_vreads.png", "Keys that need a value read, in two trained language models. No score is approximated.", width=12.5)
h2("4.3 Heavy-hitter tracker and MXFP4")
para("The tracker adds each key's weight relative to the row maximum into a 16-bit accumulator per slot, with no exponential and no divider. The same 3-stage walk finds the slot with the smallest total, skipping sinks and recent tokens.")
table(["Budget", "Model", "No eviction", "Sliding window", "Chip H2O", "Exact H2O"], [
    ["16", "Tiny Shakespeare", "4.6146", "4.6331", "**4.6081**", "4.6058"],
    ["16", "WikiText-2", "3.2942", "3.3105", "**3.3001**", "3.3008"],
    ["12", "Tiny Shakespeare", "4.6044", "4.6178", "4.6174", "4.6178"],
    ["8", "Tiny Shakespeare", "4.5851", "**4.6234**", "4.7928", "4.7808"],
    ["8", "WikiText-2", "3.3120", "**3.3669**", "3.4308", "3.4322"],
], [1.6, 3.6, 2.7, 3.0, 3.0, 3.1], caption="Perplexity at a fixed cache budget. The chip's divider-free statistic tracks exact H2O within 0.25%.", align_right_from=2)
para("MXFP4 stores E2M1 nibbles. Twice an E2M1 value is an integer, so a nibble widens to the INT4 scale and the rest of the datapath is shared. For K and V both 4-bit, the perplexity cost falls from +3.33% to +2.61%.")

# ---------- 5 verification ----------
h1("5", "Verification")
para("Every block and every pass is checked bit for bit against a Python golden model, not against a floating-point tolerance. Any change that flips a single output bit fails a test.")
tiles([("29", "cocotb tests"), ("0", "bit mismatches"), ("0.9998", "cosine vs FP32"), ("3", "levels: block, core, chip")])
table(["Layer", "What is checked"], [
    ["Golden model", "Integer-exact MX quantiser, dot product, three exponent modes, online softmax, reciprocal, requantiser, INT4/MXFP4 packing, RoPE, page table, tracker, max-first decode"],
    ["Vectorised model", "A NumPy decode path checked against the golden model, fast enough to run inside a transformer"],
    ["Block tests (6)", "Each arithmetic block, for example RoPE swept over 4,192 input pairs"],
    ["Core tests (19)", "Every dataflow and feature over random, sharp and outlier tensors, P = 2 and 4, edge cases"],
    ["Chip tests (4)", "The same passes over SPI and QSPI through the chip top, including the token window"],
    ["Macro model", "The PDK's own OpenRAM Verilog model in place of the behavioural memory"],
], [3.6, 13.4], caption="The verification stack. Tests run in cocotb on Icarus Verilog, and also on the gate-level netlist.", bold_first=True)
callout("Lesson learned", "Icarus does not re-evaluate Verilog functions that read module state when that state changes. This caused failures that only appeared in the full suite, so all such logic was rewritten as plain wires.")

# ---------- 6 physical ----------
h1("6", "Physical implementation")
para("The complete chip, with all 17 features, the SPI/QSPI port, clock gating and reset synchronisers, is laid out in sky130 with OpenLane 2 and four OpenRAM macros on a 4.84 mm² die. Yosys synthesises, OpenROAD places, routes and times it, and Magic and Netgen run DRC and LVS.")
tiles([("+0.14 ns", "setup slack, tt, 50 MHz"), ("0", "routing DRC errors"), ("Clean", "LVS"), ("1.419 mm²", "standard cells")])
figure(os.path.join(LAY, "raijin_chip_final6_zoom.png"), "Close-up of 80 × 60 µm of core logic in the sign-off layout (met1 to met5).", width=13.5)
h2("6.1 Reaching 50 MHz")
para("Timing closed over seven layout runs. Register duplication and carry-free rounding fixed the early runs. The features added after RTL freeze then cost 0.46 ns, which three RTL fixes won back without changing any cycle count.")
figure("fig_progress.png", "Typical-corner setup slack at 50 MHz across the layout runs.", width=15)
bullets([
    "The heavy-hitter walk was split into a 3-stage pipeline (+2 cycles per tracked decode step).",
    "The token-window row feeding the RoPE phase multiplier became a registered copy.",
    "The max-first drop count into the skip counters was registered and added one cycle later.",
])
h2("6.2 Sign-off layout: chip_p2_final6")
figure("fig_corners.png", "Setup slack by corner. 50 MHz is met at the nominal and min-RC typical corners and at every fast corner.", width=15)
table(["Check", "Result"], [
    ["Hold", "Met at tt and ss; 2 misses of at most 0.05 ns at the max-RC fast corner (input ports)"],
    ["Routing DRC / LVS", "0 errors / clean, circuits match uniquely"],
    ["Antenna", "1 net, 1.06× the met1 ratio"],
    ["Magic DRC", "All inside the OpenRAM macros (bitcell rules), as in every sky130 OpenRAM design"],
    ["Die / standard cells", "2.2 × 2.2 mm (4.84 mm²) / 1.419 mm², 45.8% utilisation"],
    ["Cells / flip-flops", "303,496 / 8,768"],
    ["Macros", "4 × 1 KB OpenRAM, 0.763 mm²"],
], [4.2, 12.8], caption="Sign-off checks for chip_p2_final6.", bold_first=True)
callout("Known limits", "The max-RC typical corner misses by 0.67 ns on two path groups, both fixed in RTL afterwards. The slow corner runs at about 26 MHz, and is only indicative because the OpenRAM macro has a typical-corner timing model only.", fill="FFF6E5", bar="B7791F")
h2("6.3 Power")
para("Power comes from a gate-level simulation of the routed netlist: it runs one fused pass and one decode step, matches the golden model bit for bit and annotates 100% of the 440,708 pins.")
figure("fig_power.png", "Power during a fused pass (N = d = 16, P = 2): 70.3 mW for 3.46 µs, or 243 nJ per pass and 67 GOPS/W.", width=12.5)
bullets([
    "**Clock tree:** 15.9 mW busy or idle, since the core itself is not gated. The chip-top gate turns it off for 90.8% of a decode loop.",
    "**SRAM:** 7.7 mW is a constant from the OpenRAM library, so the SRAM share is uncertain.",
    "**Vectorless estimates** ranged from 32 to 127 mW and were wrong both ways, which is why the gate-level run was needed.",
])

# ---------- 7 evaluation ----------
h1("7", "Evaluation")
h2("7.1 Accuracy inside trained transformers")
para("Every attention head of a real language model was replaced by the bit-exact chip model, and perplexity was measured. Perplexity is how surprised the model is by real text, so lower is better.")
figure("fig_accuracy.png", "Perplexity cost on a 4-layer character model trained on Tiny Shakespeare (FP32 perplexity 4.852, 8,192 predictions).", width=15.5)
bullets([
    "The chip's arithmetic costs +0.07% in MF2 mode; the full edge operating point (INT8 K, INT4 V, key skip) costs +0.72%.",
    "Values tolerate 4 bits far better than keys: +0.81% for INT4 V against +3.33% for both.",
    "WikiText-2 gives the same ranking: +0.04% for MF2 and +0.51% at the operating point.",
])
h3("A pretrained model: TinyStories-8M")
para("This GPT-Neo model has 16 heads of d = 16 per layer, which match the chip exactly. As is, the chip costs +4.3% perplexity. A study split the loss and found its cause in the integer score path, not in the MX format.")
table(["Variant", "Perplexity vs FP32", "Top-1 agreement"], [
    ["MXINT8 Q, K, V with an FP32 softmax", "+0.08%", "98.38%"],
    ["Chip, exact exponent", "+3.55%", "93.58%"],
    ["Chip, after key-mean removal", "−0.01%", "98.47%"],
    ["Chip, MF2, all rewrites", "+0.22%", "97.18%"],
    ["Chip, MF2, all rewrites, V in INT4", "+2.74%", "90.78%"],
], [8.2, 4.4, 4.4], caption="Where the TinyStories-8M loss comes from (300 windows).", highlight=(2,), align_right_from=1)
callout("The fix costs nothing", "GPT-Neo's keys share a large mean, so every score carries a big common offset that saturates the 12-bit score. Softmax ignores a shift common to a row, so subtracting the key mean, folded once into the K bias, removes the loss with no hardware change.")
para("**Worked example.** One head of the prompt \"Once upon a time, there was a little girl named Lily. She\" was replayed on the RTL: 97 cycles, bit-exact. The FP32 model ends \"…in the sunshine\" and the Raijin model \"…in the park\".")
h2("7.2 Long generation: a 1,000-word story")
para("With a 32-token MXFP4 cache and every head on the chip model, TinyStories-8M wrote a 1,005-word story (1,460 tokens). The host keeps 4 sink tokens plus either the newest tokens or the heaviest hitters from the on-chip tracker.")
callout("Opening of the story", "\"Once upon a time, in a small village by the sea, there lived a girl named Lily. Lily loved to dance, but she was always very foolish and never listened to her mom and dad…\"")
figure("fig_story.png", "Perplexity over 6,682 predictions on the 8 longest validation stories.", width=15.5)
bullets([
    "The 32-token window alone costs +26% even in FP32; 4-bit KV is most of the rest. A larger INT8 cache is the clear next step.",
    "The demo found a real bug: 32 flat keys could overflow the softmax sum. l was widened to 14 bits, with a new test that fails on the old RTL.",
    "H2O loses to the plain window on this model, which leans on recent context, so the window is the recommended policy.",
])
h2("7.3 Scaling model")
para("A cycle model written from the controller's schedule matches every RTL cycle count exactly, except causal prefill at P = 4 (2% high). Decode is bound by KV bandwidth, one K and one V row per cycle, so it does not depend on P.")
table(["N", "d", "P", "Fused cycles", "MAC use", "GOPS", "Decode tokens/s"], [
    ["16", "16", "4", "111", "58%", "6.6", "900,000"],
    ["128", "64", "4", "16,431", "100%", "11.5", "82,418"],
    ["512", "64", "4", "262,191", "100%", "11.5", "21,614"],
    ["512", "64", "8", "131,123", "100%", "23.0", "21,614"],
    ["2048", "128", "4", "8,388,655", "100%", "11.5", "2,741"],
], [1.6, 1.6, 1.4, 3.2, 2.4, 2.2, 4.6], caption="Extending the design to LLM-sized heads at 45 MHz.", align_right_from=0)
h2("7.4 Comparison with prior accelerators")
figure("fig_compare.png", "Energy efficiency. Prior numbers are 40 nm results scaled to 130 nm (clock ÷ 3.25, area × 10.6, energy × 8.7).", width=15)
table(["Design (at 130 nm)", "Clock", "Area", "GOP/s", "GOP/s/mm²", "GOP/J"], [
    ["A3", "308 MHz", "22.0 mm²", "68", "3.1", "31"],
    ["SpAtten 1/8", "308 MHz", "16.4 mm²", "111", "6.8", "44"],
    ["ELSA (with memories)", "308 MHz", "22.7 mm²", "335", "14.8", "84"],
    ["Raijin P = 2, measured", "49.3 MHz", "4.84 mm²", "4.67", "0.96", "67"],
    ["Raijin P = 2, at 308 MHz", "308 MHz", "4.84 mm²", "29", "6.0", "–"],
], [4.6, 2.4, 2.6, 2.2, 2.6, 2.6], caption="Throughput and efficiency. The prior designs have 2–16× more multipliers and a much higher clock.", highlight=(3,), align_right_from=1)
para("At the same clock, Raijin's area efficiency is in the range of A3 and SpAtten. ELSA leads because it approximates which keys to compute; Raijin computes every score exactly, and none of the three supports decode.")

# ---------- 8 applications ----------
h1("8", "Applications and comparison with CPUs and GPUs")
para("Raijin is a milliwatt-scale attention co-processor for always-on, battery-powered devices. A host microcontroller runs the projections and feed-forward layers; Raijin runs attention over the KV cache.")
table(["Application", "Why Raijin fits"], [
    ["Voice assistant (wake word, then commands)", "Always on at tens of mW; the clock gate idles the core between frames"],
    ["Predictive text on wearables and e-readers", "Short contexts fit the on-chip cache; one token per burst"],
    ["Hearing aids and earbuds", "Well under 100 mW and no DRAM"],
    ["Industrial and medical sensors", "Sliding-window decode with sinks over a stream that never ends"],
    ["Small robots and drones", "Offloads attention from the MCU over four SPI pins"],
    ["Offline tiny-LLM devices and education kits", "Open design, cheap 130 nm process, no cloud"],
], [7.0, 10.0], caption="Where a chip like Raijin fits.", bold_first=True)
table(["Device", "Process", "Throughput", "Power", "Efficiency"], [
    ["Raijin, P = 2 (this chip)", "130 nm", "4.7 GOPS", "70 mW", "67 GOPS/W"],
    ["Raijin, P = 8 (model)", "130 nm", "≈ 23 GOPS", "≈ 0.2 W", "≈ 100 GOPS/W"],
    ["Microcontroller (M4/M7)", "40–90 nm", "0.2–1 GOPS", "20–100 mW", "5–15 GOPS/W"],
    ["Laptop CPU, 8 cores", "4–7 nm", "0.5–2 TOPS", "15–45 W", "20–50 GOPS/W"],
    ["Edge GPU (Orin Nano)", "8 nm", "20–40 TOPS", "7–15 W", "2–3 TOPS/W"],
    ["Desktop GPU (4090)", "4–5 nm", "≈ 650 TOPS", "≈ 450 W", "≈ 1.5 TOPS/W"],
], [5.0, 2.4, 3.2, 2.9, 3.5], caption="Against standard platforms. Rows other than Raijin are typical published ranges (about ±2×).", highlight=(0,), align_right_from=1)
bullets([
    "**Against a microcontroller**, its intended host, Raijin is about 5–20× faster and 5–10× more efficient on attention.",
    "**Against GPUs**, most of the per-watt gap is the process: 130 nm against 4–8 nm costs roughly 30–50× in energy per operation.",
    "**End-to-end limits:** the host runs the rest of the model, the QSPI link caps TinyStories-8M at about 200 tokens/s, and the on-chip cache holds 16 to 32 tokens per head.",
])

# ---------- 9 conclusion ----------
h1("9", "Conclusion and future work")
para("A small, exact attention engine can carry the whole decode-side feature set of modern LLM serving for under 1% perplexity, and was built and laid out with open tools. The speed comes from the dataflow features, and the energy savings from the KV-cache features.")
tiles([("3.7×", "fused vs unfused"), ("4.1× / 4.0×", "prefill / decode"), ("5.3×", "fewer V reads"), ("< 1%", "perplexity cost")])
h2("Limitations")
bullets([
    "**Clock and size:** 50 MHz and P = 2 are set by sky130 cells and OpenRAM macros.",
    "**Power:** the SRAM share of the 67 GOPS/W figure is a library constant.",
    "**Scale:** accuracy was measured on small models with d = 16.",
    "**Cache size:** 16 to 32 tokens on chip; longer contexts need a page streamer that is not built yet.",
])
h2("Future work, in order of value")
numbered([
    "A P = 4 layout on a smaller die, then P = 8, for 11.5 and 23 GOPS at the same clock.",
    "An external-memory page streamer behind the page table.",
    "A larger INT8 KV cache, which the story benchmark shows is worth more than better arithmetic.",
    "MXFP8 elements, and an evaluation on a 1B-class model at d = 64.",
    "A full tape-out: ChipFoundry sky130 first, then GF180 through wafer.space.",
])

# ---------- references ----------
h1("", "References")
refs = [
    "T. J. Ham et al., \"A3: Accelerating Attention Mechanisms in Neural Networks with Approximation,\" HPCA 2020. arXiv 2002.10941.",
    "H. Wang, Z. Zhang, S. Han, \"SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning,\" HPCA 2021. arXiv 2012.09852.",
    "T. J. Ham et al., \"ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks,\" ISCA 2021.",
    "M. Milakov, N. Gimelshein, \"Online normalizer calculation for softmax,\" 2018.",
    "T. Dao et al., \"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,\" NeurIPS 2022.",
    "Open Compute Project, \"OCP Microscaling Formats (MX) Specification v1.0,\" 2023.",
    "N. Shazeer, \"Fast Transformer Decoding: One Write-Head is All You Need,\" 2019; J. Ainslie et al., \"GQA,\" EMNLP 2023.",
    "G. Xiao et al., \"Efficient Streaming Language Models with Attention Sinks,\" ICLR 2024.",
    "W. Kwon et al., \"Efficient Memory Management for Large Language Model Serving with PagedAttention,\" SOSP 2023.",
    "Z. Liu et al., \"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,\" ICML 2024.",
    "J. Su et al., \"RoFormer: Enhanced Transformer with Rotary Position Embedding,\" 2021.",
    "Y. Leviathan et al., \"Fast Inference from Transformers via Speculative Decoding,\" ICML 2023.",
    "Z. Zhang et al., \"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models,\" NeurIPS 2023.",
    "Raijin repository (github.com/ninja3011/raijin), branch claude/project-thread-g22jnw: RTL, golden model, tests, OpenLane configs and evaluation scripts.",
]
for i, rf in enumerate(refs, 1):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8); p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"[{i}]  "); r.bold = True; r.font.color.rgb = rgb(MID); r.font.size = Pt(9.5)
    r = p.add_run(rf); r.font.size = Pt(9.5)

# drop the empty first paragraph Document() starts with
first = doc.paragraphs[0]
if not first.text.strip() and not first.runs: first._p.getparent().remove(first._p)
doc.core_properties.title = "Raijin: an MX-format attention accelerator"; doc.core_properties.author = "Ninad Jangle"
doc.save(OUT); print("saved", OUT)
