import io, os, subprocess, sys, json
import pymupdf, pytest
from lbrief.config import Config, STATUS_NONE, STATUS_REVIEW, STATUS_LOCATED
from lbrief.parser import parse_pdf, parse_text, InputError
from lbrief.preprocess import split_sentences, build_units
from lbrief.chunking import chunk_units, approx_token_count
from lbrief.models import SourceUnit
from lbrief.linker import link_statements, detail_flags
from lbrief.pipeline import generate_brief
from lbrief.exporter import build_pdf
from lbrief.summarizer import TransformerSummarizer, ExtractiveSummarizer
from lbrief.evaluation import audit_stats, classification_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "synthetic_sample.pdf")


@pytest.fixture(scope="module")
def sample_bytes():
    if not os.path.exists(SAMPLE):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "make_synthetic_sample.py"), SAMPLE])
    return open(SAMPLE, "rb").read()


# AT-01 / AT-02 / encrypted / corrupt / paste
def test_valid_pdf_has_page_numbers(sample_bytes):
    doc, pages = parse_pdf(sample_bytes, "s.pdf", Config())
    assert doc.page_count == len(pages) == 4 and [p.page_number for p in pages] == [1, 2, 3, 4]


def test_image_only_pdf_rejected():
    d = pymupdf.open(); d.new_page(); d.new_page()
    with pytest.raises(InputError, match="OCR"):
        parse_pdf(d.tobytes(), "scan.pdf", Config())


def test_encrypted_pdf_rejected(sample_bytes):
    d = pymupdf.open(stream=sample_bytes, filetype="pdf")
    enc = d.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(InputError, match="password"):
        parse_pdf(enc, "e.pdf", Config())


@pytest.mark.parametrize("blob", [b"", b"not a pdf", b"%PDF-1.4 garbage garbage"])
def test_corrupt_inputs_fail_gracefully(blob):
    with pytest.raises(InputError):
        parse_pdf(blob, "x.pdf", Config())


def test_short_paste_rejected():
    with pytest.raises(InputError):
        parse_text("too short", Config())


# preprocessing: offsets resolve into raw text; legal abbreviations survive
def test_offsets_and_abbreviations(sample_bytes):
    cfg = Config(); doc, pages = parse_pdf(sample_bytes, "s.pdf", cfg)
    units = build_units(doc.document_id, pages, cfg)
    pg = {p.page_number: p for p in pages}
    for u in units:
        assert pg[u.page_number].raw_text[u.start_offset:u.end_offset].strip() == u.raw_text.strip()
    assert any("Rs. 45,000" in u.normalized_text for u in units)
    assert not any(u.normalized_text.strip() in ("Rs.", "No.") for u in units)


def test_running_header_and_page_number_removed(sample_bytes):
    cfg = Config(); doc, pages = parse_pdf(sample_bytes, "s.pdf", cfg)
    units = build_units(doc.document_id, pages, cfg)
    assert not any("SYNTHETIC SAMPLE" in u.normalized_text and "IN THE SUPREME COURT OF INDIA -" in u.normalized_text for u in units)
    assert not any(u.normalized_text.lower().startswith("page ") and len(u.normalized_text) < 10 for u in units)


# AT-05: chunking never exceeds limit and never drops text
def test_chunks_respect_limit_and_keep_all_text():
    units = [SourceUnit(f"s{i}", "d", 1, i, "word " * 50, "word " * 50, 0, 1) for i in range(40)]
    units.append(SourceUnit("big", "d", 1, 99, "w " * 3000, "w " * 3000, 0, 1))
    chunks = chunk_units(units, 300)
    assert all(c["tokens"] <= 300 + 5 for c in chunks)
    assert sum(len(c["text"].split()) for c in chunks) == sum(len(u.normalized_text.split()) for u in units)
    assert "big" in {i for c in chunks for i in c["source_ids"]}


# AT-03/06/13: end-to-end, locator accuracy = 100%
def test_end_to_end_locator_accuracy(sample_bytes):
    b = generate_brief(sample_bytes, "s.pdf")
    keys = {s.section_key: s for s in b.sections}
    assert all(keys[k].statements for k in ("facts", "issues", "arguments_petitioner", "arguments_respondent", "reasoning", "decision"))
    assert b.metadata["year"] == "2019" and b.metadata["court"] == "Supreme Court of India"
    n = 0
    for s in b.sections:
        for st in s.statements:
            for eid in st.evidence_ids:
                ev = b.evidence[eid]; u = b.units[ev.source_id]
                assert ev.page_number == u.page_number and ev.source_excerpt.strip() == u.raw_text.strip()
                n += 1
    assert n > 0
    # party attribution preserved: no respondent sentence under petitioner section
    for st in keys["arguments_petitioner"].statements:
        assert "respondent State" not in st.text


# AT-04: missing section is not fabricated
def test_missing_section_not_fabricated():
    text = ("The appellant was appointed as a clerk on 12 March 2004. " * 5 + "He filed a writ petition before the High Court. " * 5)
    b = generate_brief(pasted_text=text * 2)
    sec = {s.section_key: s for s in b.sections}
    assert sec["arguments_respondent"].statements == [] and sec["arguments_respondent"].section_status == "Not confidently identified"
    assert sec["decision"].statements == []
    assert all(e.page_number == 1 for e in b.evidence.values())


# AT-07: weak / unrelated statement is never "Evidence located"
def test_weak_retrieval_never_marked_located():
    cfg = Config()
    cand = [SourceUnit("a", "d", 3, 0, "The appeal is allowed with costs.", "The appeal is allowed with costs.", 0, 5)]
    store = {}
    out = link_statements("decision", ["Quantum computing will transform agriculture in 2045."], cand, cfg, store, "x")
    assert out[0].evidence_status == STATUS_NONE and not out[0].evidence_ids


def test_detail_flags_catch_number_name_negation():
    f = detail_flags("The Court did not award Rs. 90,000 to Mehta.", "The court awarded Rs. 45,000 to the appellant.")
    joined = " ".join(f)
    assert "90000" in joined and "Mehta" in joined and "negation" in joined


def test_numbers_present_not_flagged():
    assert detail_flags("It was filed on 18 January 2018.", "The writ was filed on 18 January 2018.") == []


# Transformer path with injected generator (weights not available offline)
def test_transformer_backend_logic_with_stub(sample_bytes):
    calls = []
    def fake(text):
        calls.append(text); return "The appellant was dismissed from service. The appellant was dismissed from service. "
    cfg = Config(); summ = TransformerSummarizer(cfg, generate_fn=fake, count_fn=approx_token_count)
    b = generate_brief(sample_bytes, "s.pdf", cfg=cfg, summarizer=summ)
    assert calls and all(approx_token_count(c) <= cfg.chunk_token_limit for c in calls)
    for s in b.sections:
        texts = [t.text.lower() for t in s.statements]
        assert len(texts) == len(set(texts))                 # exact-duplicate removal
    empty = TransformerSummarizer(cfg, generate_fn=lambda t: "", count_fn=approx_token_count)
    b2 = generate_brief(sample_bytes, "s.pdf", cfg=cfg, summarizer=empty)
    assert all(s.statements == [] and s.section_status == "Not confidently identified" for s in b2.sections)


# AT-08: export renders, includes disclaimer
def test_pdf_export(sample_bytes):
    b = generate_brief(sample_bytes, "s.pdf")
    data = build_pdf(b)
    d = pymupdf.open(stream=data, filetype="pdf")
    text = " ".join(p.get_text() for p in d)
    assert data[:5] == b"%PDF-" and "Not legal advice" in text and "Page 1" in text and "Court's Reasoning" in text


def test_audit_stats_counts():
    rows = [{"has_evidence": "1", "final_label": "Supported"}, {"has_evidence": "1", "final_label": "Partially supported"},
            {"has_evidence": "0", "final_label": "Unsupported"}, {"has_evidence": "1", "final_label": "Not assessable"}]
    s = audit_stats(rows)
    assert s["assessable"] == 3 and s["supported"]["count"] == 1 and s["evidence_link_coverage"]["count"] == 3


# classifier training plumbing on SYNTHETIC rows (proves the script works, says nothing about accuracy)
def test_train_script_runs_on_synthetic_csv(tmp_path):
    import pandas as pd
    base = {"Facts": "The appellant filed a petition on 5 June 2012 before the court",
            "Issue": "Whether the order of dismissal is valid under the rules",
            "Arguments of Petitioner": "Learned counsel for the appellant submitted that the inquiry was unfair",
            "Arguments of Respondent": "Learned counsel for the respondent submitted that the inquiry was fair",
            "Reasoning": "We are of the view that the inquiry was defective in law",
            "Decision": "In the result the appeal is allowed and the order set aside", 0: "Page header text here"}
    rows = [(f"{t} number {i}", l) for l, t in base.items() for i in range(30)]
    df = pd.DataFrame(rows, columns=["Text", "Label"])
    df.iloc[:100].to_csv(tmp_path / "tr.csv", index=False); df.iloc[100:].to_csv(tmp_path / "va.csv", index=False)
    out = tmp_path / "m.joblib"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "train_classifier.py"), "--train", str(tmp_path / "tr.csv"),
                           "--val", str(tmp_path / "va.csv"), "--out", str(out), "--report", str(tmp_path / "r.json"), "--c-grid", "1.0"],
                          cwd=ROOT, stdout=subprocess.DEVNULL)
    cfg = Config(); cfg.classifier_path = str(out)
    from lbrief.classifier import RoleClassifier
    clf = RoleClassifier(cfg)
    assert clf.is_trained and clf.version.startswith("tfidf-lr")
    b = generate_brief(open(SAMPLE, "rb").read(), "s.pdf", cfg=cfg, classifier=clf)
    assert b.classifier_version.startswith("tfidf-lr") and not any("heuristic" in w for w in b.warnings)


def test_train_script_rejects_unknown_labels(tmp_path):
    import pandas as pd
    pd.DataFrame({"Text": ["a b c"] * 5, "Label": ["Weird"] * 5}).to_csv(tmp_path / "x.csv", index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_classifier.py"), "--train", str(tmp_path / "x.csv"),
                        "--val", str(tmp_path / "x.csv")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "Unknown labels" in (r.stdout + r.stderr)


# AT-10: logs never contain document text
def test_no_document_text_in_logs(sample_bytes, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        generate_brief(sample_bytes, "s.pdf")
    assert "Ramesh" not in caplog.text and "appellant" not in caplog.text
