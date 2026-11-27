"""PDF export (FR-14). Same CaseBrief object as the UI. White page, near-black text, blue headings."""
from __future__ import annotations
import io
import os
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
from .config import DISCLAIMER, STATUS_NONE
from .models import CaseBrief

INK, MUTED, BLUE, BORDER, SURFACE = (colors.HexColor(h) for h in ("#111318", "#5B6472", "#2563EB", "#E5E7EB", "#F7F8FA"))
_FONTS = [("DejaVuSans", ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                          "/Library/Fonts/Arial Unicode.ttf"]),
          ("DejaVuSans-Bold", ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                               "/Library/Fonts/Arial Unicode.ttf"])]


def _fonts() -> tuple[str, str]:
    """Prefer a Unicode TTF (so symbols like the rupee sign survive); fall back to built-in Helvetica."""
    try:
        names = []
        for name, paths in _FONTS:
            p = next((p for p in paths if os.path.exists(p)), None)
            if not p:
                return "Helvetica", "Helvetica-Bold"
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, p))
            names.append(name)
        return names[0], names[1]
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _label(brief: CaseBrief, ev_id: str) -> str:
    ev = brief.evidence[ev_id]
    return "Pasted text" if brief.document.source_type == "pasted" else f"Page {ev.page_number}"


def build_pdf(brief: CaseBrief) -> bytes:
    base, bold = _fonts()
    st = {
        "title": ParagraphStyle("t", fontName=bold, fontSize=18, leading=22, textColor=INK, spaceAfter=4),
        "meta": ParagraphStyle("m", fontName=base, fontSize=9, leading=12, textColor=MUTED),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=13, leading=16, textColor=BLUE, spaceBefore=14, spaceAfter=4),
        "body": ParagraphStyle("b", fontName=base, fontSize=10.5, leading=15, textColor=INK, spaceAfter=3),
        "status": ParagraphStyle("s", fontName=bold, fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=2),
        "ev": ParagraphStyle("e", fontName=base, fontSize=9, leading=12.5, textColor=INK, leftIndent=10,
                             backColor=SURFACE, borderPadding=(3, 4, 3, 4), spaceAfter=5),
        "notice": ParagraphStyle("n", fontName=bold, fontSize=9.5, leading=13, textColor=INK, backColor=SURFACE,
                                 borderPadding=(6, 6, 6, 6), spaceAfter=10),
    }
    s = []
    md = brief.metadata
    s.append(Paragraph(escape(md.get("title") or "Untitled judgment"), st["title"]))
    bits = [f"Document: {escape(brief.document.filename)}", f"Pages: {brief.document.page_count}"]
    for lab, k in (("Court", "court"), ("Case no.", "case_number"), ("Year", "year")):
        bits.append(f"{lab}: {escape(md[k]) if md.get(k) else 'Not detected'}")
    s.append(Paragraph(" | ".join(bits), st["meta"]))
    s.append(Spacer(1, 8))
    s.append(Paragraph(escape(DISCLAIMER), st["notice"]))
    for w in brief.warnings:
        s.append(Paragraph("Note: " + escape(w), st["meta"]))
    for sec in brief.sections:
        block = [Paragraph(escape(sec.title), st["h1"]), HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4)]
        if not sec.statements:
            block.append(Paragraph(escape(sec.section_status + ". " + sec.note), st["meta"]))
            s.append(KeepTogether(block))
            continue
        s.extend(block)
        for stt in sec.statements:
            s.append(Paragraph(escape(stt.text), st["body"]))
            s.append(Paragraph(escape(stt.evidence_status.upper()) + ("" if stt.evidence_status != STATUS_NONE else ""), st["status"]))
            for f in stt.flags:
                s.append(Paragraph("Review note: " + escape(f), st["meta"]))
            for eid in stt.evidence_ids:
                ev = brief.evidence[eid]
                s.append(Paragraph(f"<b>Source ({escape(_label(brief, eid))}, ref {escape(ev.source_id)}):</b> {escape(ev.source_excerpt)}", st["ev"]))
    s.append(Spacer(1, 14))
    s.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    s.append(Paragraph("Method and limitations", st["h1"]))
    s.append(Paragraph(escape(
        f"Section detection: {brief.classifier_version}. Summarizer: {brief.summarizer_id}. Generated {brief.generated_at}. "
        "'Evidence located' means a candidate source passage was retrieved by lexical similarity; it does not prove that the "
        "statement is correct. Summaries may omit or misstate details. " + DISCLAIMER), st["body"]))

    short = (md.get("title") or brief.document.filename)[:60]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"{short} | generated {brief.generated_at[:10]}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=f"Case brief - {short}").build(s, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
