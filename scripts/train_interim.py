"""INTERIM classifier trained on the public JURIX-2019 corpus (50 labelled Indian Supreme Court judgments).
Use this until you have LegalSeg. Its label scheme is coarser than LegalSeg, so:
  trained (4 classes): Facts | Arguments | Reasoning | Decision
  rule-based on top : Issue ('Whether...' cues) and Arguments -> AoP/AoR by party cues   (NOT evaluated: no ground truth)
Evaluation = 5-fold GroupKFold BY DOCUMENT (never sentence-level splits), pooled predictions, vs the heuristic baseline.
Also tests whether adding relative document position helps (decided from CV, reported either way)."""
import argparse, glob, json, os, platform, sys
import joblib, numpy as np, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.classifier import heuristic_labels, add_position_tokens
from lbrief.evaluation import classification_metrics

COARSE = ["Facts", "Arguments", "Reasoning", "Decision"]
MAP = {"Facts": "Facts", "Ruling by Lower Court": "Facts", "Argument": "Arguments", "Ratio of the decision": "Reasoning",
       "Precedent": "Reasoning", "Statute": "Reasoning", "Ruling by Present Court": "Decision"}
HEUR2COARSE = {"Facts": "Facts", "AoP": "Arguments", "AoR": "Arguments", "Reasoning": "Reasoning", "Decision": "Decision"}


def load(dir_):
    docs = []
    for f in sorted(glob.glob(os.path.join(dir_, "*.txt"))):
        sents, labs = [], []
        for ln in open(f, encoding="utf-8", errors="replace"):
            if "\t" not in ln.strip("\n"):
                continue
            t, l = ln.rstrip("\n").rsplit("\t", 1)
            if t.strip() and l.strip() in MAP:
                sents.append(t.strip()); labs.append(MAP[l.strip()])
        docs.append((os.path.basename(f), sents, labs))
    return docs


def flatten(docs, pos):
    X, y, g = [], [], []
    for gi, (_, s, l) in enumerate(docs):
        X += add_position_tokens(s) if pos else s; y += l; g += [gi] * len(s)
    return X, np.array(y), np.array(g)


def make(C):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=500))])


def cv(docs, pos, C, seed):
    X, y, g = flatten(docs, pos)
    pred = np.empty(len(y), dtype=object); fold_f1 = []
    for tr, te in GroupKFold(5).split(X, y, g):
        p = make(C).fit([X[i] for i in tr], y[tr])
        pred[te] = p.predict([X[i] for i in te])
        fold_f1.append(classification_metrics(y[te].tolist(), pred[te].tolist(), COARSE)["macro_f1"])
    return classification_metrics(y.tolist(), pred.tolist(), COARSE), fold_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/external/semantic-segmentation/data/text")
    ap.add_argument("--out", default="models/role_classifier.joblib"); ap.add_argument("--report", default="reports/classifier_cv.json")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gate", type=float, default=0.45)
    a = ap.parse_args()
    docs = load(a.corpus)
    if len(docs) < 10:
        sys.exit("Corpus not found; run scripts/fetch_public_data.py first.")
    n_sent = sum(len(d[1]) for d in docs)
    print(f"{len(docs)} documents, {n_sent} sentences")
    # heuristic baseline on the SAME sentences (doc-sequential context kept)
    y_true, y_heur = [], []
    for _, s, l in docs:
        y_true += l; y_heur += [HEUR2COARSE.get(x, "None") for x, _ in heuristic_labels(s)]
    heur = classification_metrics(y_true, y_heur, COARSE)
    print(f"heuristic baseline: macro-F1={heur['macro_f1']:.3f}")
    results = {}
    for pos in (False, True):
        for C in (1.0, 4.0, 16.0):
            m, folds = cv(docs, pos, C, a.seed)
            results[f"pos={pos},C={C}"] = {"macro_f1": m["macro_f1"], "fold_macro_f1_mean": float(np.mean(folds)),
                                          "fold_macro_f1_std": float(np.std(folds)), "metrics": m}
            print(f"pos={pos!s:5} C={C:>4}: pooled macro-F1={m['macro_f1']:.3f}  folds {np.mean(folds):.3f}+-{np.std(folds):.3f}")
    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    use_pos, C = "pos=True" in best_key, float(best_key.split("C=")[1])
    X, y, _ = flatten(docs, use_pos)
    final = make(C).fit(X, y)
    meta = {"version": f"tfidf-lr-jurix-interim-{'pos' if use_pos else 'nopos'}-C{C:g}", "label_scheme": "coarse-jurix",
            "use_position": use_pos, "gate": a.gate, "C": C, "seed": a.seed, "classes": list(final.classes_),
            "dataset": "JURIX-2019 Indian SC rhetorical roles, 50 docs / %d sentences (Law-AI/semantic-segmentation, MIT)" % n_sent,
            "label_mapping": MAP, "sklearn": sklearn.__version__, "python": platform.python_version()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); os.makedirs(os.path.dirname(a.report), exist_ok=True)
    joblib.dump({"pipeline": final, **meta}, a.out)
    json.dump({"meta": meta, "note": "5-fold GroupKFold by document, pooled predictions. Config chosen on the same CV (mildly optimistic). "
               "Coarse JURIX labels, NOT LegalSeg; Issue and AoP/AoR are rule-based and unevaluated.",
               "cv_best": best_key, "cv_all": {k: {kk: v[kk] for kk in ("macro_f1", "fold_macro_f1_mean", "fold_macro_f1_std")} for k, v in results.items()},
               "validation": results[best_key]["metrics"], "heuristic_baseline": heur}, open(a.report, "w"), indent=2)
    print("best:", best_key, "->", a.out)


if __name__ == "__main__":
    main()
