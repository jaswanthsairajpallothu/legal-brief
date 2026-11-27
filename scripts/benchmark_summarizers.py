"""Compare AT MOST TWO summarization checkpoints on your Indian judgment sample. Writes outputs + timings for
HUMAN review (faithfulness, party attribution, dates/numbers/negation). Requires torch + transformers and
network access to download weights. Select by human review first, ROUGE second, speed third.
  python scripts/benchmark_summarizers.py --pdfs j1.pdf j2.pdf --models nsi319/legal-pegasus facebook/bart-large-cnn"""
import argparse, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.config import Config
from lbrief.classifier import RoleClassifier
from lbrief.summarizer import TransformerSummarizer
from lbrief.pipeline import generate_brief

ap = argparse.ArgumentParser()
ap.add_argument("--pdfs", nargs="+", required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", default="reports/summarizer_benchmark.json")
a = ap.parse_args()
if len(a.models) > 2:
    sys.exit("Spec: benchmark at most two candidates.")
results = {}
for mid in a.models:
    cfg = Config(); cfg.summarizer_model_id = mid
    summ = TransformerSummarizer(cfg)
    clf = RoleClassifier(cfg)
    results[mid] = {}
    for p in a.pdfs:
        t0 = time.perf_counter()
        b = generate_brief(open(p, "rb").read(), os.path.basename(p), cfg=cfg, classifier=clf, summarizer=summ)
        results[mid][os.path.basename(p)] = {"seconds": round(time.perf_counter() - t0, 1),
            "sections": {s.section_key: [{"text": t.text, "status": t.evidence_status, "flags": t.flags} for t in s.statements] for s in b.sections}}
    del summ; gc.collect()                                   # one model in memory at a time
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(results, open(a.out, "w"), indent=2)
print("wrote", a.out)
