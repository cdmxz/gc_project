"""python-docx 排版辅助函数：A4 页面、中文字体、图表题注、页眉页脚、代码块。"""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_FONT = "宋体"
HEAD_FONT = "微软雅黑"
MONO_FONT = "Consolas"
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0x1F, 0x7A, 0x4D)


def set_run_font(run, name, size=None, bold=None, color=None, ascii_name=None):
    run.font.name = ascii_name or name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def init_document():
    doc = Document()
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.4)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.line_spacing = 1.26
    normal.paragraph_format.space_after = Pt(3)
    return doc


def add_page_number_footer(section, text_prefix="第 "):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text_prefix)
    set_run_font(run, BODY_FONT, 9, color=GRAY)

    run2 = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run2._r.append(fld_begin)
    run2._r.append(instr)
    run2._r.append(fld_end)
    set_run_font(run2, BODY_FONT, 9, color=GRAY)

    run3 = p.add_run(" 页")
    set_run_font(run3, BODY_FONT, 9, color=GRAY)


def add_header_text(section, text):
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(text)
    set_run_font(run, HEAD_FONT, 8.5, color=GRAY)
    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:color"), "BFCFC5")
    borders.append(bottom)
    pPr.append(borders)


def heading(doc, text, level=1, page_break=False):
    p = doc.add_paragraph()
    if page_break:
        p.paragraph_format.page_break_before = True
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 9)
    pf.space_after = Pt(6 if level == 1 else 3)
    pf.keep_with_next = True
    run = p.add_run(text)
    size = {1: 15, 2: 12.5, 3: 11}.get(level, 11)
    set_run_font(run, HEAD_FONT, size, bold=True, color=INK)
    if level == 1:
        pPr = p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "10")
        bottom.set(qn("w:color"), "1F7A4D")
        bottom.set(qn("w:space"), "3")
        borders.append(bottom)
        pPr.append(borders)
    # 大纲级别，便于 Word 导航
    pPr = p._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    pPr.append(outline)
    return p


def para(doc, text, size=10.5, bold=False, color=INK, align=None, indent=True, space_after=4, italic=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_run_font(run, BODY_FONT, size, bold=bold, color=color)
    run.italic = italic
    return p


def bullet(doc, text, size=10.5, char="•", level=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(18 + level * 18)
    pf.first_line_indent = Pt(-11)
    pf.space_after = Pt(2)
    run = p.add_run(f"{char} ")
    set_run_font(run, BODY_FONT, size, color=ACCENT, bold=True)
    run2 = p.add_run(text)
    set_run_font(run2, BODY_FONT, size)
    return p


def caption(doc, text, before=2, after=8, size=9):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    run = p.add_run(text)
    set_run_font(run, BODY_FONT, size, bold=True, color=GRAY)
    return p


def _shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _cell_borders(cell, color="D5E2DA"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    tcPr.append(borders)


def table(doc, headers, rows, widths=None, caption_text=None, font_size=8.4,
          header_fill="E8F3EC", align_center_cols=None, first_col_center=False):
    if caption_text:
        caption(doc, caption_text, before=8, after=3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        cell = hdr[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        run = p.add_run(h)
        set_run_font(run, HEAD_FONT, font_size, bold=True, color=RGBColor(0x14, 0x33, 0x28))
        _shade(cell, header_fill)
        _cell_borders(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cell = cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            center = (align_center_cols is None and i > 0) or (align_center_cols and i in align_center_cols)
            if first_col_center and i == 0:
                center = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(str(val))
            set_run_font(run, BODY_FONT, font_size)
            _cell_borders(cell)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    # 表后留白
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.line_spacing = Pt(6)
    return t


def figure(doc, path, caption_text=None, width_cm=15.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    if caption_text:
        caption(doc, caption_text, before=0, after=8)


def code_block(doc, text, caption_text=None, size=7.9):
    if caption_text:
        caption(doc, caption_text, before=6, after=2)
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    _shade(cell, "F5F8F6")
    _cell_borders(cell, "DDE7E1")
    cell.text = ""
    lines = text.strip("\n").split("\n")
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    for i, line in enumerate(lines):
        run = p.add_run(line)
        set_run_font(run, MONO_FONT, size, color=RGBColor(0x1B, 0x2B, 0x22))
        if i < len(lines) - 1:
            run.add_break()
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.line_spacing = Pt(6)


def note_box(doc, text, fill="FFF9EC", border="EBD9AE", size=9.5):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    _shade(cell, fill)
    _cell_borders(cell, border)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run_font(run, BODY_FONT, size, color=RGBColor(0x6D, 0x55, 0x22))
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.line_spacing = Pt(6)
