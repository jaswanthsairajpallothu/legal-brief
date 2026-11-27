import json
import os, subprocess, sys
from lbrief.insights import basic_stats, top_terms, entities
from lbrief.pipeline import generate_brief

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs"))
from worked_examples import edit_distance, bigram_prob_laplace


def test_insights_real_outputs():
    b = generate_brief(open(os.path.join(ROOT, "data", "synthetic_sample.pdf"), "rb").read(), "s.pdf")
    s = basic_stats(b.units, b.roles, b.document.page_count)
    assert s["sentences"] == len(b.units) and sum(s["role_counts"].values()) == s["sentences"]
    assert top_terms([u.normalized_text for u in b.units.values()])
    ents, why = entities([u.normalized_text for u in b.units.values()])
    assert ents is not None or "not installed" in why          # never fake entities


def test_worked_examples():
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3
    p = bigram_prob_laplace("the", "court", ["the court held", "the court said", "the appeal failed"])
    assert abs(p - (2 + 1) / (3 + len({"the", "court", "held", "said", "appeal", "failed"}))) < 1e-9


def test_rouge_script(tmp_path):
    (tmp_path / "refs").mkdir(); (tmp_path / "pdfs").mkdir()
    import shutil
    shutil.copy(os.path.join(ROOT, "data", "synthetic_sample.pdf"), tmp_path / "pdfs" / "j1.pdf")
    (tmp_path / "refs" / "j1.txt").write_text("The appeal was allowed and the dismissal order was set aside; reinstatement was directed.")
    out = tmp_path / "rouge.json"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "evaluate_summaries.py"), "--pdfs", str(tmp_path / "pdfs"),
                           "--refs", str(tmp_path / "refs"), "--out", str(out)], cwd=ROOT, stdout=subprocess.DEVNULL)
    import json
    d = json.load(open(out))
    assert d["n"] == 1 and 0 <= d["rouge1"] <= 1 and "overlap" in d["note"].lower()


# ---- interim (trained) classifier -------------------------------------------------
import pytest
from lbrief.config import Config
from lbrief.classifier import RoleClassifier, refine_coarse, add_position_tokens, heuristic_labels

ARTIFACT = os.path.join(ROOT, "models", "role_classifier.joblib")


def test_position_tokens_cover_deciles():
    t = add_position_tokens([f"s{i}" for i in range(100)])
    assert t[0].endswith("zzpos0") and t[99].endswith("zzpos9") and len(t) == 100


def test_refine_coarse_rules():
    texts = ["Whether the order is valid?", "The appellant submitted that the inquiry was unfair.",
             "He further said it was biased.", "The respondent argued otherwise.", "Something vague.", "Appeal dismissed."]
    out = refine_coarse(texts, ["Reasoning", "Arguments", "Arguments", "Arguments", "Reasoning", "Decision"], [.9, .9, .9, .9, .2, .9], 0.45)
    labs = [l for l, _ in out]
    assert labs == ["Issue", "AoP", "AoP", "AoR", "None", "Decision"]      # Issue rule, party cue, inheritance, gate


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="interim model not trained")
def test_interim_artifact_is_default_and_flagged():
    clf = RoleClassifier(Config())
    assert clf.is_trained and clf.is_interim and clf.version.startswith("tfidf-lr-jurix-interim")
    b = generate_brief(open(os.path.join(ROOT, "data", "synthetic_sample.pdf"), "rb").read(), "s.pdf", classifier=clf)
    assert any("interim classifier" in w for w in b.warnings)
    labels = {r.predicted_label for r in b.roles.values()}
    assert labels <= {"Facts", "Issue", "AoP", "AoR", "Reasoning", "Decision", "None"} and len(labels) >= 3


def test_heuristic_is_weak_and_reports_exist():
    """Guards the honesty claim: the heuristic stand-in must not be presented as the model."""
    rep = os.path.join(ROOT, "reports", "classifier_cv.json")
    if not os.path.exists(rep):
        pytest.skip("run scripts/train_interim.py")
    d = json.load(open(rep))
    assert d["validation"]["macro_f1"] > d["heuristic_baseline"]["macro_f1"]
