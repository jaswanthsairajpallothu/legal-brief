"""Metrics (SRS section 9). ROUGE/BLEU-style overlap never proves legal correctness; audit stats report counts
beside percentages and are never collapsed into one 'faithfulness score'."""
from __future__ import annotations
from collections import Counter
from typing import Sequence
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, cohen_kappa_score)
from .config import LABELS

SUPPORT_LABELS = ["Supported", "Partially supported", "Unsupported", "Contradicted", "Not assessable"]


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict:
    rep = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy_secondary": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "per_class": {l: {k: float(rep[l][k]) if k != "support" else int(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for l in labels},
        "confusion_labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }


def rouge_scores(preds: Sequence[str], refs: Sequence[str]) -> dict:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {"n": len(preds), **{k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()},
            "note": "Overlap metrics only; they do not establish legal correctness or faithfulness."}


def audit_stats(rows: list[dict]) -> dict:
    """rows: dicts with final_label, has_evidence ('1'/'0'), optional locator_ok ('y'/'n'), reviewer_1/2."""
    total = len(rows)
    linked = sum(1 for r in rows if str(r.get("has_evidence")) == "1")
    labelled = [r for r in rows if r.get("final_label")]
    assessable = [r for r in labelled if r["final_label"] != "Not assessable"]
    cnt = Counter(r["final_label"] for r in assessable)
    n = len(assessable)

    def rate(k):
        return {"count": cnt.get(k, 0), "of": n, "rate": (cnt.get(k, 0) / n) if n else None}

    loc = [r for r in rows if r.get("locator_ok") in ("y", "n")]
    both = [(r["reviewer_1"], r["reviewer_2"]) for r in rows if r.get("reviewer_1") and r.get("reviewer_2")]
    kappa = None
    if len(both) >= 5 and len({a for a, _ in both} | {b for _, b in both}) > 1:
        kappa = float(cohen_kappa_score([a for a, _ in both], [b for _, b in both]))
    return {
        "statements_total": total,
        "evidence_link_coverage": {"count": linked, "of": total, "rate": linked / total if total else None},
        "labelled": len(labelled), "assessable": n,
        "supported": rate("Supported"), "partially_supported": rate("Partially supported"),
        "unsupported": rate("Unsupported"), "contradicted": rate("Contradicted"),
        "locator_accuracy": {"count": sum(1 for r in loc if r["locator_ok"] == "y"), "of": len(loc)},
        "inter_reviewer_kappa": kappa, "double_reviewed": len(both),
    }


def threshold_sweep(rows: list[dict], thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) -> list[dict]:
    """For each retrieval threshold: coverage and the share of linked statements humans judged
    Supported/Partially supported. Use it to calibrate link_weak/strong thresholds."""
    out = []
    lab = [r for r in rows if r.get("final_label") and r["final_label"] != "Not assessable" and r.get("top_score") not in (None, "")]
    for t in thresholds:
        kept = [r for r in lab if float(r["top_score"]) >= t]
        ok = sum(1 for r in kept if r["final_label"] in ("Supported", "Partially supported"))
        out.append({"threshold": t, "kept": len(kept), "of": len(lab), "precision_support_or_partial": (ok / len(kept)) if kept else None})
    return out
