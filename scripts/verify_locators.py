"""Locator-accuracy check on REAL judgment text: render each judgment to a multi-page PDF, run the pipeline, then verify
every evidence item (a) resolves to a stored SourceUnit and (b) its excerpt really appears on the cited PDF page.
This tests the page/offset mapping, not real-world scanned/odd-layout PDFs. Writes reports/locator_check.json."""
import argparse, glob, json, os, re, sys
import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.config import Config
from lbrief.classifier import RoleClassifier
from lbrief.summarizer import make_summarizer
from lbrief.pipeline import generate_brief

ap = argparse.ArgumentParser(); ap.add_argument("--mildsum", required=True); ap.add_argument("--out", default="reports/locator_check.json")
a = ap.parse_args()
ws = lambda s: re.sub(r"\s+", "", s)
cfg = Config(); clf = RoleClassifier(cfg); summ = make_summarizer(cfg)
tot = ok_unit = ok_page = 0; docs = []
for d in sorted(glob.glob(os.path.join(a.mildsum, "Sample_*"))):
    txt = open(os.path.join(d, "EN_Judgment.txt"), encoding="utf-8", errors="replace").read()
    pdf = f"/tmp/_loc_{os.path.basename(d)}.pdf"
    ss = getSampleStyleSheet()
    SimpleDocTemplate(pdf, pagesize=A4).build([x for para in txt.split("\n") if para.strip() for x in (Paragraph(escape(para), ss["Normal"]), Spacer(1, 5))])
    data = open(pdf, "rb").read(); doc = pymupdf.open(stream=data, filetype="pdf")
    pages = {i + 1: ws(p.get_text()) for i, p in enumerate(doc)}
    b = generate_brief(data, os.path.basename(pdf), cfg=cfg, classifier=clf, summarizer=summ)
    n = 0
    for sec in b.sections:
        for s in sec.statements:
            for eid in s.evidence_ids:
                ev = b.evidence[eid]; n += 1
                u = b.units.get(ev.source_id)
                ok_unit += bool(u and u.page_number == ev.page_number and u.raw_text.strip() == ev.source_excerpt)
                ok_page += ws(ev.source_excerpt) in pages.get(ev.page_number, "")
    tot += n; docs.append({"doc": os.path.basename(d), "pages": len(pages), "evidence_items": n})
res = {"evidence_items": tot, "resolve_to_source_unit": ok_unit, "excerpt_found_on_cited_page": ok_page, "documents": docs,
       "caveat": "PDFs were generated from the judgment text; verifies page/offset mapping, not arbitrary real-world PDF layouts."}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps({k: res[k] for k in ("evidence_items", "resolve_to_source_unit", "excerpt_found_on_cited_page")}))
