import os
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = open(os.path.join(ROOT, "tests", "paste_sample.txt"), encoding="utf-8").read()


def fresh():
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=90).run()


def test_home_renders():
    at = fresh()
    assert not at.exception
    assert any("Create a case brief" in t.value for t in at.title)


def test_empty_submit_shows_error():
    at = fresh()
    at.button[0].click().run()
    assert at.error and not at.exception


def test_short_paste_shows_plain_error():
    at = fresh()
    at.text_area[0].set_value("too short")
    at.button[0].click().run()
    assert at.error and "at least" in at.error[0].value and not at.exception


def test_paste_flow_reaches_brief_and_evidence_navigation():
    at = fresh()
    at.text_area[0].set_value(TEXT)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["brief"] is not None
    assert any("Ramesh" in t.value for t in at.title)             # now on the brief page
    view = [b for b in at.button if b.label == "View source"]
    assert view
    view[0].click().run()
    assert any("Source evidence" in s.value for s in at.subheader)
    assert any("Source text (original)" in m.value for m in at.markdown)


def test_methodology_page_without_and_with_reports(tmp_path, monkeypatch):
    import json, shutil
    monkeypatch.chdir(tmp_path)
    for d in ("assets", "views", "ui", "lbrief"):
        shutil.copytree(os.path.join(ROOT, d), tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(ROOT, "app.py"), tmp_path / "app.py")
    at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60)
    at.run(); at.switch_page("views/methodology.py").run()
    assert not at.exception and any("No evaluation reports" in i.value for i in at.info)
    (tmp_path / "reports").mkdir()
    cm = {"n": 10, "accuracy_secondary": .5, "macro_f1": .4, "weighted_f1": .45,
          "per_class": {"Facts": {"precision": .5, "recall": .5, "f1-score": .5, "support": 5}},
          "confusion_labels": ["Facts"], "confusion_matrix": [[5]]}
    json.dump({"model": "m", "dataset": "d", "test": cm}, open(tmp_path / "reports" / "classifier_test.json", "w"))
    at.run()
    assert not at.exception and any("official LegalSeg test split" in s.value for s in at.subheader)
