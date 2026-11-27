import logging
import streamlit as st
from lbrief.config import DISCLAIMER, STATUS_LOCATED, STATUS_REVIEW, STATUS_NONE
from lbrief.parser import InputError
from lbrief.pipeline import generate_brief, STAGES
from lbrief.exporter import build_pdf
from lbrief import insights
from ui.common import cfg, ICON, STAGE_HELP, load_models, e, page_label
import pandas as pd

import json
from pathlib import Path

def metrics_tables(d, title):
    st.markdown(f"**{title}**")
    c = st.columns(4)
    c[0].metric("Macro-F1 (primary)", f"{d['macro_f1']:.3f}"); c[1].metric("Weighted-F1", f"{d['weighted_f1']:.3f}")
    c[2].metric("Accuracy (secondary)", f"{d['accuracy_secondary']:.3f}"); c[3].metric("Sentences", d["n"])
    st.dataframe(pd.DataFrame(d["per_class"]).T.rename(columns={"f1-score": "f1"}), width="stretch")
    labs = d["confusion_labels"]
    st.caption("Confusion matrix (rows = true, columns = predicted)")
    st.dataframe(pd.DataFrame(d["confusion_matrix"], index=labs, columns=labs), width="stretch")


def page_method():
    st.title("Methodology and evaluation")
    st.caption("Automated overlap metrics do not prove legal correctness or faithfulness.")
    brief = st.session_state.get("brief")
    if brief:
        st.markdown(f"**Current run** · section detection `{brief.classifier_version}` · summarizer `{brief.summarizer_id}` · "
                    f"generation `{brief.generation_params}`")
        st.dataframe(pd.DataFrame({"Stage": list(brief.timings), "Seconds": list(brief.timings.values())}), hide_index=True)
    R = Path("reports")
    shown = False
    if (R / "classifier_test.json").exists():
        shown = True
        d = json.load(open(R / "classifier_test.json"))
        st.subheader("Rhetorical-role classifier: official LegalSeg test split")
        st.caption(f"Model {d.get('model')} · dataset {d.get('dataset')}")
        metrics_tables(d["test"], "Test results")
    if (R / "classifier_cv.json").exists():
        shown = True
        d = json.load(open(R / "classifier_cv.json"))
        st.subheader("Interim classifier: 5-fold cross-validation by document")
        st.caption(d["note"])
        st.caption(f"Dataset: {d['meta']['dataset']}")
        c = st.columns(3)
        c[0].metric("Macro-F1, selected model", f"{d['validation']['macro_f1']:.3f}")
        c[1].metric("Macro-F1, cue-phrase heuristic", f"{d['heuristic_baseline']['macro_f1']:.3f}")
        c[2].metric("Selected configuration", d["cv_best"])
        st.dataframe(pd.DataFrame([{"configuration": k, **v} for k, v in d["cv_all"].items()]), hide_index=True, width="stretch")
        metrics_tables(d["validation"], "Per-class results (pooled over folds)")
    elif (R / "classifier_val.json").exists():
        shown = True
        d = json.load(open(R / "classifier_val.json"))
        st.subheader("Rhetorical-role classifier: validation only (test split not yet run)")
        metrics_tables(d["validation"], "Validation results")
    if (R / "rouge.json").exists():
        shown = True
        d = json.load(open(R / "rouge.json"))
        st.subheader("Summarization overlap (ROUGE)")
        rows = [{"system": f"This pipeline ({d.get('summarizer', '')})", **{k: d[k] for k in ("n", "rouge1", "rouge2", "rougeL")}}]
        if d.get("lead_baseline"):
            rows.append({"system": "Lead-sentences baseline (same length)", "n": d["n"], **d["lead_baseline"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Reference: {d.get('reference', 'n/a')}. " + d.get("note", ""))
    if (R / "locator_check.json").exists():
        shown = True
        d = json.load(open(R / "locator_check.json"))
        st.subheader("Evidence locator check (real judgment text)")
        st.write(f"{d['excerpt_found_on_cited_page']} of {d['evidence_items']} evidence items were found on the cited page; "
                 f"{d['resolve_to_source_unit']} of {d['evidence_items']} resolve to a stored source unit.")
        st.caption(d["caveat"])
    if (R / "audit_report.json").exists():
        shown = True
        a = json.load(open(R / "audit_report.json"))["stats"]
        st.subheader("Human claim-level evidence audit")
        rows = [("Statements audited", a["statements_total"], "", ""),
                ("Evidence-link coverage", a["evidence_link_coverage"]["count"], a["evidence_link_coverage"]["of"], a["evidence_link_coverage"]["rate"])]
        for k, lab in (("supported", "Supported"), ("partially_supported", "Partially supported"), ("unsupported", "Unsupported"), ("contradicted", "Contradicted")):
            rows.append((lab, a[k]["count"], a[k]["of"], a[k]["rate"]))
        rows.append(("Locator accuracy", a["locator_accuracy"]["count"], a["locator_accuracy"]["of"], ""))
        st.dataframe(pd.DataFrame(rows, columns=["Measure", "Count", "Of", "Rate"]).astype(str), hide_index=True, width="stretch")
        st.caption(f"Reviewer agreement (Cohen's kappa): {a['inter_reviewer_kappa']} on {a['double_reviewed']} double-reviewed statements.")
    if not shown:
        st.info("No evaluation reports found in `reports/`. Run `scripts/train_classifier.py`, `scripts/evaluate_classifier.py` and "
                "`scripts/audit.py`. Until then, no performance claim can be made.")
    st.subheader("What this system does not claim")
    st.markdown("- It does not give legal advice or verify legal correctness.\n- “Evidence located” means a candidate passage was retrieved by lexical similarity, not that the statement is proven.\n"
                "- High ROUGE does not mean high faithfulness.\n- A model trained on US legal or news text is not automatically reliable on Indian judgments.\n"
                "- A hosted demo processes uploads on the server; only local mode keeps documents on your machine.")
    st.subheader("Data and models")
    st.markdown("Classifier data: LegalSeg (L-NLProc; check current terms and record the revision). Summarizer: one pretrained checkpoint chosen by human review. "
                "Dataset files and model weights are not redistributed with this application.")


page_method()
