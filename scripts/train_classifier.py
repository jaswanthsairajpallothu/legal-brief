"""Train the rhetorical-role classifier: TF-IDF (1-2 grams) + class-weighted Logistic Regression.

YOU must first obtain LegalSeg (accept the Hugging Face terms; train.csv / val.csv / test.csv from L-NLProc).
Schema per the authors' reference code (github.com/ShubhamKumarNigam/LegalSeg): columns `Text`, `Label`, `Index`
(document id); labels Facts / Issue / Arguments of Petitioner / Arguments of Respondent / Reasoning / Decision,
with empty or 0 meaning None. Defaults below follow that; still run `head -3 train.csv` to confirm your copy.
Unknown labels abort with a list, never silently dropped.

Example:
  python scripts/train_classifier.py --train train.csv --val val.csv --doc-col Index --max-per-class 60000
"""
import argparse, json, os, platform, sys, time
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.config import LABELS
from lbrief.evaluation import classification_metrics

ALIASES = {
    "facts": "Facts", "fact": "Facts", "issue": "Issue", "issues": "Issue",
    "aop": "AoP", "argumentsofpetitioner": "AoP", "argumentofpetitioner": "AoP", "petitionerargument": "AoP",
    "aor": "AoR", "argumentsofrespondent": "AoR", "argumentofrespondent": "AoR", "respondentargument": "AoR",
    "reasoning": "Reasoning", "decision": "Decision", "none": "None", "other": "None", "nan": "None", "": "None", "0": "None", "00": "None",
}


def norm_label(x) -> str:
    return ALIASES.get("".join(ch for ch in str(x).lower() if ch.isalnum()), "?" + str(x))


def load(path, text_col, label_col):
    df = pd.read_csv(path)
    if text_col not in df or label_col not in df:
        sys.exit(f"{path}: columns {list(df.columns)} do not include --text-col={text_col} / --label-col={label_col}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna(subset=["text"])
    df["label"] = df["label"].map(norm_label)
    bad = sorted(set(df.label[df.label.str.startswith("?")]))
    if bad:
        sys.exit(f"Unknown labels in {path}: {bad}. Extend ALIASES deliberately; they are not dropped silently.")
    return df


def cap_per_class(df, cap, seed):
    if not cap:
        return df
    return df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--data"); ap.add_argument("--split-col")
    ap.add_argument("--text-col", default="Text"); ap.add_argument("--label-col", default="Label")
    ap.add_argument("--doc-col", default=None, help="document-id column (e.g. Index); used to warn on train/val overlap")
    ap.add_argument("--max-per-class", type=int, default=0, help="cap COMMON classes in TRAIN only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.5, 1.0, 4.0])
    ap.add_argument("--out", default="models/role_classifier.joblib")
    ap.add_argument("--report", default="reports/classifier_val.json")
    ap.add_argument("--dataset-note", default="LegalSeg (record revision + date accessed here)")
    a = ap.parse_args()

    if a.data:
        df = load(a.data, a.text_col, a.label_col)
        split = pd.read_csv(a.data, usecols=[a.split_col])[a.split_col].astype(str).str.lower().values
        tr, va = df[split == "train"], df[pd.Series(split).isin(["val", "validation", "dev"]).values]
    else:
        if not (a.train and a.val):
            sys.exit("Provide --train and --val (official document-level splits) or --data with --split-col.")
        tr, va = load(a.train, a.text_col, a.label_col), load(a.val, a.text_col, a.label_col)
        if a.doc_col:
            d1 = set(pd.read_csv(a.train, usecols=[a.doc_col])[a.doc_col]); d2 = set(pd.read_csv(a.val, usecols=[a.doc_col])[a.doc_col])
            print(f"document ids: train={len(d1)} val={len(d2)} overlap={len(d1 & d2)}"
                  + ("  <-- WARNING: ids overlap; confirm Index is globally unique per document, not a per-file counter" if d1 & d2 else ""))
    tr = cap_per_class(tr, a.max_per_class, a.seed)       # validation is NEVER downsampled
    print("train class counts:", tr.label.value_counts().to_dict())
    best = None
    for C in a.c_grid:
        t0 = time.time()
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)),
                         ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=300, random_state=a.seed))])
        pipe.fit(tr.text.tolist(), tr.label.tolist())          # vectorizer fitted on TRAIN only
        m = classification_metrics(va.label.tolist(), pipe.predict(va.text.tolist()))
        print(f"C={C}: val macro-F1={m['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], C, pipe, m)
    _, C, pipe, metrics = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    meta = {"version": f"tfidf-lr-C{C}-seed{a.seed}", "C": C, "seed": a.seed, "dataset": a.dataset_note,
            "train_rows": len(tr), "val_rows": len(va), "max_per_class": a.max_per_class, "label_distribution_train": tr.label.value_counts().to_dict(),
            "sklearn": sklearn.__version__, "python": platform.python_version(), "classes": list(pipe.classes_)}
    joblib.dump({"pipeline": pipe, **meta}, a.out)
    json.dump({"meta": meta, "validation": metrics}, open(a.report, "w"), indent=2)
    print("saved", a.out, "| validation macro-F1", round(metrics["macro_f1"], 4), "(TEST split untouched; run evaluate_classifier.py once)")


if __name__ == "__main__":
    main()
