"""ROUGE vs reference briefs, with a LEAD baseline (first-N sentences, same length) so you can see whether the
pipeline beats trivial. Modes: --mildsum DIR (MILDSum sample layout Sample_*/EN_Judgment.txt + EN_Summary.txt) or
--pdfs DIR --refs DIR (X.pdf + X.txt). Only use references you are allowed to use. Writes reports/rouge.json."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.config import Config
from lbrief.classifier import RoleClassifier
from lbrief.summarizer import make_summarizer
from lbrief.pipeline import generate_brief
from lbrief.evaluation import rouge_scores
from lbrief.preprocess import split_sentences

ap = argparse.ArgumentParser()
ap.add_argument("--mildsum"); ap.add_argument("--pdfs"); ap.add_argument("--refs"); ap.add_argument("--out", default="reports/rouge.json")
a = ap.parse_args()
cfg = Config(); clf = RoleClassifier(cfg); summ = make_summarizer(cfg)
items = []          # (name, kwargs for generate_brief, reference, source_text)
if a.mildsum:
    for d in sorted(glob.glob(os.path.join(a.mildsum, "Sample_*"))):
        t = open(os.path.join(d, "EN_Judgment.txt"), encoding="utf-8", errors="replace").read()
        items.append((os.path.basename(d), dict(pasted_text=t), open(os.path.join(d, "EN_Summary.txt"), encoding="utf-8").read(), t))
elif a.pdfs and a.refs:
    import pymupdf
    for pdf in sorted(glob.glob(os.path.join(a.pdfs, "*.pdf"))):
        stem = os.path.splitext(os.path.basename(pdf))[0]; rp = os.path.join(a.refs, stem + ".txt")
        if os.path.exists(rp):
            data = open(pdf, "rb").read()
            items.append((stem, dict(data=data, filename=stem + ".pdf"), open(rp, encoding="utf-8").read(), "\n".join(p.get_text() for p in pymupdf.open(stream=data, filetype="pdf"))))
else:
    sys.exit("Give --mildsum DIR or --pdfs DIR --refs DIR")
P, L, R, names = [], [], [], []
for name, kw, ref, src in items:
    b = generate_brief(cfg=cfg, classifier=clf, summarizer=summ, **kw)
    pred = " ".join(s.text for sec in b.sections for s in sec.statements)
    sents = [src[s:e] for s, e in split_sentences(src)]
    lead, n = [], 0
    for s in sents:                       # lead baseline of equal word length
        if n >= len(pred.split()): break
        lead.append(s); n += len(s.split())
    P.append(pred); L.append(" ".join(lead)); R.append(ref); names.append(name)
res = rouge_scores(P, R)
res.update(lead_baseline={k: v for k, v in rouge_scores(L, R).items() if k.startswith("rouge")},
           summarizer=summ.model_id, classifier=clf.version, documents=names,
           reference="practitioner-written English summaries (MILDSum samples)" if a.mildsum else "user-supplied")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps({k: res[k] for k in ("n", "rouge1", "rouge2", "rougeL", "lead_baseline")}, indent=1))
