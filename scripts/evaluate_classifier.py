"""Evaluate on the official TEST split. Run ONCE, after all decisions are made on validation."""
import argparse, json, os, sys
import joblib, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbrief.evaluation import classification_metrics
from train_classifier import load

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True); ap.add_argument("--model", default="models/role_classifier.joblib")
ap.add_argument("--text-col", default="Text"); ap.add_argument("--label-col", default="Label")
ap.add_argument("--out", default="reports/classifier_test.json")
a = ap.parse_args()
bundle = joblib.load(a.model)
df = load(a.test, a.text_col, a.label_col)               # never downsampled
m = classification_metrics(df.label.tolist(), bundle["pipeline"].predict(df.text.tolist()))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"model": bundle["version"], "dataset": bundle.get("dataset"), "test": m}, open(a.out, "w"), indent=2)
print(f"TEST macro-F1={m['macro_f1']:.4f} weighted-F1={m['weighted_f1']:.4f} acc(secondary)={m['accuracy_secondary']:.4f}")
for k, v in m["per_class"].items():
    print(f"  {k:10s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f} n={v['support']}")
