"""Human claim-level evidence audit (SRS 9.3). The core differentiator -- do not skip it.
  export : run the pipeline on PDFs and write a CSV for reviewers
  report : compute counts/rates from the filled CSV (+ threshold sweep)
Reviewers fill reviewer_1 (and reviewer_2 on a subset), then final_label, from:
  Supported | Partially supported | Unsupported | Contradicted | Not assessable
and optionally locator_ok (y/n: does the cited page/passage really exist in the PDF?)."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.config import Config
from lbrief.classifier import RoleClassifier
from lbrief.summarizer import make_summarizer
from lbrief.pipeline import generate_brief
from lbrief.evaluation import audit_stats, threshold_sweep

FIELDS = ["document", "section", "statement_id", "statement", "evidence_status", "has_evidence", "top_score",
          "pages", "source_ids", "source_excerpts", "flags", "reviewer_1", "reviewer_2", "final_label", "locator_ok", "notes"]

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
e = sub.add_parser("export"); e.add_argument("--pdfs", nargs="+", required=True); e.add_argument("--out", default="reports/audit_sheet.csv")
r = sub.add_parser("report"); r.add_argument("--csv", default="reports/audit_sheet.csv"); r.add_argument("--out", default="reports/audit_report.json")
a = ap.parse_args()

if a.cmd == "export":
    cfg = Config(); clf = RoleClassifier(cfg); summ = make_summarizer(cfg)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        for p in a.pdfs:
            b = generate_brief(open(p, "rb").read(), os.path.basename(p), cfg=cfg, classifier=clf, summarizer=summ)
            for sec in b.sections:
                for s in sec.statements:
                    evs = [b.evidence[i] for i in s.evidence_ids]
                    w.writerow({"document": os.path.basename(p), "section": sec.section_key, "statement_id": s.statement_id,
                                "statement": s.text, "evidence_status": s.evidence_status, "has_evidence": int(bool(evs)),
                                "top_score": round(evs[0].retrieval_score, 4) if evs else "",
                                "pages": ";".join(str(x.page_number) for x in evs), "source_ids": ";".join(x.source_id for x in evs),
                                "source_excerpts": " || ".join(x.source_excerpt for x in evs), "flags": " | ".join(s.flags)})
    print("wrote", a.out, "- now review it against the ORIGINAL PDFs")
else:
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    res = {"stats": audit_stats(rows), "threshold_sweep": threshold_sweep(rows)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res["stats"], indent=2))
