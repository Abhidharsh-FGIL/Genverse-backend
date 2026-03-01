"""
ebook_export_service.py

Server-side PDF (ReportLab Platypus) and DOCX (python-docx) generation
for professional book-format export of eBooks.

Layout per user spec:
  Page 1  — Cover (title, author, image, description)
  Page 2  — About This Book
  Page 3  — Table of Contents  (with dotted leaders + estimated page nums)
  Page 4+ — Chapters  (each starts on a fresh page)
             eyebrow | 22pt title | rule | image | 12pt justified body |
             image | key-points box | page number
  Final   — Assessment Questions  (if present)
           — Thank You  (if present)

Typography:
  Font      : Times New Roman / Times-Roman  (12pt body, 22pt chapter titles)
  Line space: 1.5 × font size
  Margins   : 1 inch on all sides  (handled by ReportLab + python-docx)
  Images    : centred, aspect-ratio preserved, never stretched
"""

from __future__ import annotations

import base64
import io
import re
from xml.sax.saxutils import escape as xml_escape

from PIL import Image as PILImage

# ── ReportLab ─────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── python-docx ───────────────────────────────────────────────────────────────
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# ── Geometry constants ────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4               # 595.28 pt × 841.89 pt
MARGIN = 1.0 * inch               # 72 pt — 1-inch margins everywhere
CONTENT_W = PAGE_W - 2 * MARGIN   # ~451 pt  (usable width in PDF)

A4_W_IN   = 8.27                  # A4 width in inches (DOCX)
DOC_MARGIN = 1.0                  # inches

FONT_RL = "Times-Roman"           # ReportLab built-in
FONT_DOCX = "Times New Roman"     # python-docx


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(data_url: str) -> bytes | None:
    """Return raw bytes from a base64 data URL, or None on failure."""
    if not data_url:
        return None
    try:
        m = re.match(r"data:[^;]+;base64,(.+)", data_url, re.DOTALL)
        return base64.b64decode(m.group(1)) if m else None
    except Exception:
        return None


def _pdf_text(value: object) -> str:
    """Escape dynamic text for ReportLab Paragraph XML parser."""
    return xml_escape("" if value is None else str(value))


# ─────────────────────────────────────────────────────────────────────────────
# PDF  (ReportLab Platypus)
# ─────────────────────────────────────────────────────────────────────────────

def _rl_image(data_url: str, max_w: float, max_h: float) -> RLImage | None:
    """Build a centred ReportLab Image flowable, scaled to fit max_w × max_h (pts)."""
    raw = _decode_image(data_url)
    if not raw:
        return None
    try:
        buf = io.BytesIO(raw)
        pil = PILImage.open(buf)
        w, h = pil.size
        if not w or not h:
            return None
        scale = min(max_w / w, max_h / h, 1.0)
        buf.seek(0)
        img = RLImage(buf, width=w * scale, height=h * scale)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return None


def _page_num_cb(canvas, doc) -> None:
    """Footer callback: draws '— N —' centred at the bottom of every page."""
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(PAGE_W / 2, 0.42 * inch, f"\u2014 {doc.page} \u2014")
    canvas.restoreState()


def generate_pdf(ebook_json: dict, book_title: str) -> bytes:
    """Return raw PDF bytes for the given ebook_json."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 0.3 * inch,  # extra room for page-number footer
        title=book_title,
        author=ebook_json.get("author", ""),
    )

    # ── Style factory ─────────────────────────────────────────────────────────
    def sty(
        name: str, *,
        bold: bool = False, italic: bool = False, size: int = 12,
        align: int = TA_LEFT, color: str = "#000000",
        before: int = 0, after: int = 8, left_indent: int = 0,
        leading: float | None = None,
    ) -> ParagraphStyle:
        if bold:
            fn = "Times-Bold"
        elif italic:
            fn = "Times-Italic"
        else:
            fn = "Times-Roman"
        return ParagraphStyle(
            name,
            fontName=fn,
            fontSize=size,
            leading=leading if leading is not None else size * 1.5,
            alignment=align,
            spaceBefore=before,
            spaceAfter=after,
            textColor=colors.HexColor(color),
            leftIndent=left_indent,
        )

    S = {
        "cov_title":  sty("ct",  bold=True,   size=30, align=TA_CENTER),
        "cov_sub":    sty("cs",  italic=True, size=14, align=TA_CENTER, color="#444444", after=6),
        "cov_author": sty("ca",                size=14, align=TA_CENTER, color="#222222", after=16),
        "cov_desc":   sty("cd",  italic=True, size=11, align=TA_CENTER, color="#555555"),
        "sec_title":  sty("st",  bold=True,   size=20, align=TA_CENTER, after=18),
        "body":       sty("bd",                size=12, align=TA_JUSTIFY),
        "eyebrow":    sty("ey",                size=9,  color="#777777", after=8),
        "ch_title":   sty("cht", bold=True,   size=22, after=8),
        "toc_entry":  sty("te",                size=12, leading=20),
        "toc_pg":     sty("tp",                size=12, align=TA_RIGHT, color="#555555"),
        "dots":       ParagraphStyle(
                          "dots", fontName="Times-Roman", fontSize=10, leading=20,
                          alignment=TA_CENTER, textColor=colors.HexColor("#cccccc")),
        "kp_label":   sty("kpl", bold=True,   size=9,  color="#333333", after=6),
        "kp_item":    sty("kpi",               size=11, color="#111111", left_indent=14),
        "aq_grp":     sty("aqg", bold=True,   size=14, before=16, after=8),
        "aq_q":       sty("aqq", bold=True,   size=12, after=4),
        "aq_opt":     sty("aqo",               size=11, left_indent=20, after=2),
        "aq_ans":     sty("aqa", italic=True, size=10, color="#555555", left_indent=14, after=8),
        "ty_title":   sty("tyt", bold=True,   size=20, align=TA_CENTER, color="#222222", after=16),
        "ty_text":    sty("tyt2", italic=True, size=12, align=TA_CENTER, color="#444444"),
    }

    # ── Unpack ebook data ─────────────────────────────────────────────────────
    ej = ebook_json
    tp        = ej.get("title_page", {})
    toc       = ej.get("table_of_contents", [])
    chapters  = ej.get("chapters", [])
    ch_imgs   = ej.get("images", {}).get("chapter_images", {})
    cov_img   = ej.get("images", {}).get("cover_image")
    author    = tp.get("author") or ej.get("author", "")
    subtitle  = tp.get("subtitle", "")
    descr     = tp.get("description", "")
    summary   = ej.get("book_summary", "")
    thanks    = ej.get("thank_you_message", "")
    asmnt     = ej.get("final_assessment")

    story: list = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph(_pdf_text(book_title), S["cov_title"]))
    if subtitle:
        story.append(Paragraph(_pdf_text(subtitle), S["cov_sub"]))
    story.append(HRFlowable(
        width=60, thickness=2, color=colors.HexColor("#333333"),
        spaceBefore=10, spaceAfter=12,
    ))
    if author:
        story.append(Paragraph(f"by {_pdf_text(author)}", S["cov_author"]))
    if cov_img:
        img = _rl_image(cov_img, CONTENT_W * 0.75, 2.8 * inch)
        if img:
            story += [Spacer(1, 0.1 * inch), img]
    if descr:
        story += [Spacer(1, 0.25 * inch), Paragraph(_pdf_text(descr), S["cov_desc"])]
    story.append(PageBreak())

    # ── About This Book ───────────────────────────────────────────────────────
    if summary:
        story.append(Paragraph("About This Book", S["sec_title"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=colors.black, spaceAfter=20))
        for para in _split_paras(summary):
            story.append(Paragraph(_pdf_text(para), S["body"]))
        story.append(PageBreak())

    # ── Table of Contents ─────────────────────────────────────────────────────
    if toc:
        story.append(Paragraph("Table of Contents", S["sec_title"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=colors.black, spaceAfter=20))

        pg_count  = ej.get("page_count", 15)
        front     = 1 + (1 if summary else 0) + 1
        first_pg  = front + 1
        avg       = max(1, round((pg_count - front) / max(len(chapters), 1)))

        toc_rows = []
        for i, item in enumerate(toc):
            num  = str(item.get("chapter_number", i + 1))
            ttl  = item.get("title", "")
            pg   = str(first_pg + i * avg)
            toc_rows.append([
                Paragraph(f"<b>{_pdf_text(num)}.</b>\u2002{_pdf_text(ttl)}", S["toc_entry"]),
                Paragraph("." * 50, S["dots"]),
                Paragraph(_pdf_text(pg), S["toc_pg"]),
            ])

        tbl = Table(
            toc_rows,
            colWidths=[CONTENT_W * 0.56, CONTENT_W * 0.30, CONTENT_W * 0.14],
        )
        tbl.setStyle(TableStyle([
            ("VALIGN",         (0, 0), (-1, -1), "BOTTOM"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(PageBreak())

    # ── Chapters ──────────────────────────────────────────────────────────────
    for i, ch in enumerate(chapters):
        ch_num = ch.get("chapter_number", i + 1)
        ch_ttl = ch.get("title", "")
        body   = ch.get("content") or ch.get("description", "")
        kps    = ch.get("key_points") or []
        imgs   = ch_imgs.get(str(i), [])

        ch_story: list = [
            Paragraph(f"CHAPTER {ch_num}", S["eyebrow"]),
            Paragraph(_pdf_text(ch_ttl), S["ch_title"]),
            HRFlowable(width=52, thickness=2, color=colors.black, spaceAfter=18),
        ]

        # Image 1  (before body text)
        if imgs and imgs[0]:
            img = _rl_image(imgs[0], CONTENT_W * 0.8, 2.6 * inch)
            if img:
                ch_story += [Spacer(1, 0.1 * inch), img, Spacer(1, 0.2 * inch)]

        # Body paragraphs
        for para in _split_paras(body):
            ch_story.append(Paragraph(_pdf_text(para), S["body"]))

        # Image 2  (after body text)
        if len(imgs) > 1 and imgs[1]:
            img = _rl_image(imgs[1], CONTENT_W * 0.8, 2.6 * inch)
            if img:
                ch_story += [Spacer(1, 0.2 * inch), img, Spacer(1, 0.2 * inch)]

        # Key-points box
        if kps:
            kp_inner: list = [Paragraph("KEY POINTS", S["kp_label"])]
            for kp in kps:
                kp_inner.append(Paragraph(f"\u25b8\u2002{_pdf_text(kp)}", S["kp_item"]))
            kp_tbl = Table([[kp_inner]], colWidths=[CONTENT_W])
            kp_tbl.setStyle(TableStyle([
                ("LEFTPADDING",   (0, 0), (-1, -1), 14),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
                ("TOPPADDING",    (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f6f6f6")),
                ("LINEBEFORE",    (0, 0), (0, -1),  4, colors.HexColor("#333333")),
            ]))
            ch_story += [Spacer(1, 0.15 * inch), kp_tbl]

        story.append(PageBreak())
        story.extend(ch_story)

    # ── Assessment ────────────────────────────────────────────────────────────
    if asmnt:
        story.append(PageBreak())
        story.append(Paragraph("Assessment Questions", S["sec_title"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=colors.black, spaceAfter=20))

        def _qgroup(qs: list, label: str, qtype: str) -> None:
            if not qs:
                return
            story.append(Paragraph(label, S["aq_grp"]))
            for j, q in enumerate(qs):
                ch_ref = q.get("chapter_number", "")
                blk: list = [
                    Paragraph(
                        f"{j + 1}. {_pdf_text(q.get('question', ''))} "
                        f"<font color='#aaaaaa' size='9'>(Ch.\u202f{_pdf_text(ch_ref)})</font>",
                        S["aq_q"],
                    )
                ]
                if qtype == "mcq":
                    for k, opt in enumerate(q.get("options") or []):
                        blk.append(Paragraph(f"{chr(65 + k)})\u2002{_pdf_text(opt)}", S["aq_opt"]))
                if q.get("answer"):
                    blk.append(Paragraph(f"Answer:\u2002{_pdf_text(q['answer'])}", S["aq_ans"]))
                story.append(KeepTogether(blk))
                story.append(Spacer(1, 0.08 * inch))

        _qgroup(asmnt.get("mcq_questions"),          "Multiple Choice Questions", "mcq")
        _qgroup(asmnt.get("fill_in_blank_questions"), "Fill in the Blanks",       "fib")
        _qgroup(asmnt.get("short_answer_questions"),  "Short Answer Questions",   "sa")
        _qgroup(asmnt.get("long_answer_questions"),   "Long Answer Questions",    "la")

    # ── Thank You ─────────────────────────────────────────────────────────────
    if thanks:
        story.append(PageBreak())
        story += [
            Spacer(1, 1.5 * inch),
            Paragraph("Thank You", S["ty_title"]),
            Spacer(1, 0.2 * inch),
            Paragraph(_pdf_text(thanks), S["ty_text"]),
        ]

    doc.build(story, onFirstPage=_page_num_cb, onLaterPages=_page_num_cb)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# DOCX  (python-docx)
# ─────────────────────────────────────────────────────────────────────────────

def _split_paras(text: str) -> list[str]:
    """Split raw text into non-empty paragraphs (separated by blank lines)."""
    return [p.strip().replace("\n", " ") for p in text.split("\n\n") if p.strip()]


def _page_break(doc: Document) -> None:
    """Insert a hard page break into the document."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    run = p.add_run()
    br  = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def _sp(
    doc: Document, text: str, *,
    bold: bool = False, italic: bool = False, size: int = 12,
    align: int = WD_ALIGN_PARAGRAPH.LEFT,
    before: int = 0, after: int = 8,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Add a styled Times New Roman paragraph to doc."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after  = Pt(after)
    p.paragraph_format.line_spacing = Pt(size * 1.5)
    run = p.add_run(text)
    run.font.name   = FONT_DOCX
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def _add_rule(doc: Document, *, bold: bool = False) -> None:
    """Add a paragraph with a bottom border acting as a horizontal rule."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)
    pPr    = p._p.get_or_add_pPr()
    pBdr   = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "12" if bold else "6")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")
    pBdr.append(bottom)
    pPr.append(pBdr)


def _docx_image(doc: Document, data_url: str, max_w_in: float, max_h_in: float) -> None:
    """Insert a centred image from a base64 data URL (scales to fit constraints)."""
    raw = _decode_image(data_url)
    if not raw:
        return
    try:
        buf   = io.BytesIO(raw)
        pil   = PILImage.open(buf)
        w_px, h_px = pil.size
        if not w_px or not h_px:
            return
        dpi   = 96.0
        scale = min(max_w_in / (w_px / dpi), max_h_in / (h_px / dpi), 1.0)
        buf.seek(0)
        doc.add_picture(buf, width=Inches((w_px / dpi) * scale))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception:
        pass


def _no_border_cell(cell) -> None:
    """Remove all borders from a table cell."""
    tcPr     = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("left", "right", "top", "bottom", "insideH", "insideV"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "none")
        tcBorders.append(tag)
    tcPr.append(tcBorders)


def _kp_box(doc: Document, key_points: list[str]) -> None:
    """Render a key-points box: gray background + 4pt left border."""
    tbl  = doc.add_table(rows=1, cols=1)
    cell = tbl.rows[0].cells[0]

    # Gray shading
    tcPr = cell._tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  "F6F6F6")
    tcPr.append(shd)

    # Left border only
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom", "right"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "none")
        tcBorders.append(tag)
    left = OxmlElement("w:left")
    left.set(qn("w:val"),   "single")
    left.set(qn("w:sz"),    "24")
    left.set(qn("w:space"), "0")
    left.set(qn("w:color"), "333333")
    tcBorders.append(left)
    tcPr.append(tcBorders)

    # "KEY POINTS" label
    lp  = cell.paragraphs[0]
    lp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    lr  = lp.add_run("KEY POINTS")
    lr.font.name  = FONT_DOCX
    lr.font.bold  = True
    lr.font.size  = Pt(9)
    lr.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Bullet items
    for kp in key_points:
        ip = cell.add_paragraph(f"\u25b8\u2002{kp}")
        ip.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in ip.runs:
            run.font.name = FONT_DOCX
            run.font.size = Pt(11)


def _docx_footer_page_num(doc: Document) -> None:
    """Add centred '— N —' page numbers to all section footers."""
    for section in doc.sections:
        footer = section.footer
        p      = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.clear()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        def _run(txt: str) -> None:
            r = p.add_run(txt)
            r.font.name  = FONT_DOCX
            r.font.size  = Pt(10)
            r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        _run("\u2014\u2009")  # em-dash + thin space

        # PAGE field code
        r2 = p.add_run()
        r2.font.name = FONT_DOCX
        r2.font.size = Pt(10)
        r2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        for tag, ftype in [("begin", None), (None, " PAGE "), ("end", None)]:
            if tag is not None:
                fld = OxmlElement("w:fldChar")
                fld.set(qn("w:fldCharType"), tag)
                r2._r.append(fld)
            else:
                instr = OxmlElement("w:instrText")
                instr.set(qn("xml:space"), "preserve")
                instr.text = ftype
                r2._r.append(instr)

        _run("\u2009\u2014")  # thin space + em-dash


def generate_docx(ebook_json: dict, book_title: str) -> bytes:
    """Return raw DOCX bytes for the given ebook_json."""
    doc = Document()

    # A4 page, 1-inch margins on all sides
    for section in doc.sections:
        section.page_width    = Inches(A4_W_IN)
        section.page_height   = Inches(11.69)
        section.top_margin    = Inches(DOC_MARGIN)
        section.bottom_margin = Inches(DOC_MARGIN)
        section.left_margin   = Inches(DOC_MARGIN)
        section.right_margin  = Inches(DOC_MARGIN)

    # Effective image widths (within margins)
    MAX_IMG_W = A4_W_IN - 2 * DOC_MARGIN   # 6.27 in
    MAX_IMG_H = 3.0                          # in

    # ── Unpack ebook data ─────────────────────────────────────────────────────
    ej       = ebook_json
    tp       = ej.get("title_page", {})
    toc      = ej.get("table_of_contents", [])
    chapters = ej.get("chapters", [])
    ch_imgs  = ej.get("images", {}).get("chapter_images", {})
    cov_img  = ej.get("images", {}).get("cover_image")
    author   = tp.get("author") or ej.get("author", "")
    subtitle = tp.get("subtitle", "")
    descr    = tp.get("description", "")
    summary  = ej.get("book_summary", "")
    thanks   = ej.get("thank_you_message", "")
    asmnt    = ej.get("final_assessment")

    # ── Cover ─────────────────────────────────────────────────────────────────
    _sp(doc, book_title,
        bold=True, size=30, align=WD_ALIGN_PARAGRAPH.CENTER, before=48, after=8)
    if subtitle:
        _sp(doc, subtitle,
            italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    _add_rule(doc, bold=True)
    if author:
        _sp(doc, f"by {author}", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=20)
    if cov_img:
        _docx_image(doc, cov_img, MAX_IMG_W * 0.75, 2.8)
    if descr:
        _sp(doc, descr, italic=True, size=11,
            align=WD_ALIGN_PARAGRAPH.CENTER, before=12, color=(85, 85, 85))
    _page_break(doc)

    # ── About This Book ───────────────────────────────────────────────────────
    if summary:
        _sp(doc, "About This Book",
            bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
        _add_rule(doc, bold=True)
        for para in _split_paras(summary):
            _sp(doc, para, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        _page_break(doc)

    # ── Table of Contents ─────────────────────────────────────────────────────
    if toc:
        _sp(doc, "Table of Contents",
            bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
        _add_rule(doc, bold=True)

        pg_count = ej.get("page_count", 15)
        front    = 1 + (1 if summary else 0) + 1
        first_pg = front + 1
        avg      = max(1, round((pg_count - front) / max(len(chapters), 1)))

        tbl = doc.add_table(rows=0, cols=3)
        for i, item in enumerate(toc):
            row  = tbl.add_row().cells
            for cell in row:
                _no_border_cell(cell)

            num = str(item.get("chapter_number", i + 1))
            ttl = item.get("title", "")
            pg  = str(first_pg + i * avg)

            p0 = row[0].paragraphs[0]
            p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r0 = p0.add_run(f"{num}. {ttl}")
            r0.font.name = FONT_DOCX
            r0.font.size = Pt(12)

            p1 = row[1].paragraphs[0]
            p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r1 = p1.add_run("." * 25)
            r1.font.name = FONT_DOCX
            r1.font.size = Pt(10)
            r1.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

            p2 = row[2].paragraphs[0]
            p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r2 = p2.add_run(pg)
            r2.font.name = FONT_DOCX
            r2.font.size = Pt(12)
            r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

        _page_break(doc)

    # ── Chapters ──────────────────────────────────────────────────────────────
    for i, ch in enumerate(chapters):
        if i > 0:
            _page_break(doc)

        ch_num = ch.get("chapter_number", i + 1)
        ch_ttl = ch.get("title", "")
        body   = ch.get("content") or ch.get("description", "")
        kps    = ch.get("key_points") or []
        imgs   = ch_imgs.get(str(i), [])

        _sp(doc, f"CHAPTER {ch_num}", size=9, color=(119, 119, 119), after=4)
        _sp(doc, ch_ttl, bold=True, size=22, after=4)
        _add_rule(doc)

        if imgs and imgs[0]:
            _docx_image(doc, imgs[0], MAX_IMG_W * 0.8, MAX_IMG_H - 0.4)

        for para in _split_paras(body):
            _sp(doc, para, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

        if len(imgs) > 1 and imgs[1]:
            _docx_image(doc, imgs[1], MAX_IMG_W * 0.8, MAX_IMG_H - 0.4)

        if kps:
            _kp_box(doc, kps)

    # ── Assessment ────────────────────────────────────────────────────────────
    if asmnt:
        _page_break(doc)
        _sp(doc, "Assessment Questions",
            bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
        _add_rule(doc, bold=True)

        def _qgroup_docx(qs: list, label: str, qtype: str) -> None:
            if not qs:
                return
            _sp(doc, label, bold=True, size=14, before=12, after=6)
            for j, q in enumerate(qs):
                _sp(doc, f"{j + 1}. {q.get('question', '')}", bold=True, size=12, after=4)
                if qtype == "mcq":
                    for k, opt in enumerate(q.get("options") or []):
                        _sp(doc, f"   {chr(65 + k)}) {opt}", size=11, after=2)
                if q.get("answer"):
                    _sp(doc, f"Answer: {q['answer']}", italic=True, size=10,
                        color=(85, 85, 85), after=8)

        _qgroup_docx(asmnt.get("mcq_questions"),          "Multiple Choice Questions", "mcq")
        _qgroup_docx(asmnt.get("fill_in_blank_questions"), "Fill in the Blanks",       "fib")
        _qgroup_docx(asmnt.get("short_answer_questions"),  "Short Answer Questions",   "sa")
        _qgroup_docx(asmnt.get("long_answer_questions"),   "Long Answer Questions",    "la")

    # ── Thank You ─────────────────────────────────────────────────────────────
    if thanks:
        _page_break(doc)
        _sp(doc, "Thank You",
            bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, before=72, after=16)
        _sp(doc, thanks, italic=True, size=12,
            align=WD_ALIGN_PARAGRAPH.CENTER, color=(68, 68, 68))

    _docx_footer_page_num(doc)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
