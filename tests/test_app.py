import json
import os

import gradio as gr

from ui.gradio_app import create_app, process_document, render_brief, render_methodology, render_source


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = open(os.path.join(ROOT, "tests", "paste_sample.txt"), encoding="utf-8").read()


def _remove_temp(path):
    if path and os.path.exists(path):
        os.remove(path)


def test_gradio_app_has_judora_navigation():
    app = create_app()
    config = json.dumps(app.get_config_file(), default=str)
    assert isinstance(app, gr.Blocks)
    assert "Judora" in config
    assert "New judgment" in config
    assert "Case brief" in config
    assert "Methodology" in config


def test_empty_and_short_inputs_return_helpful_errors():
    brief, message, pdf = process_document(None, "")
    assert brief is None and "Upload a judgment PDF" in message and pdf is None

    brief, message, pdf = process_document(None, "Too short.")
    assert brief is None and "at least" in message and pdf is None


def test_pasted_judgment_renders_summary_sources_and_pdf():
    brief, message, pdf = process_document(None, TEXT, progress=None)
    try:
        assert brief is not None, message
        assert "Brief ready" in message
        assert os.path.isfile(pdf)

        title, metadata, warnings, content, selector, evidence, insights = render_brief(brief)
        assert "Ramesh" in title
        assert "Pasted judgment text" in metadata
        assert "Final Decision" in content
        assert "Source references" in content
        assert "Top TF-IDF terms" in insights

        statement_id = next(
            statement.statement_id
            for section in brief.sections
            for statement in section.statements
            if statement.evidence_ids
        )
        source = render_source(brief, statement_id)
        assert "Original judgment" in source
        assert "Pasted text" in source
    finally:
        _remove_temp(pdf)


def test_pdf_upload_uses_the_same_brief_pipeline():
    pdf_path = os.path.join(ROOT, "data", "synthetic_sample.pdf")
    brief, message, pdf = process_document(pdf_path, "", progress=None)
    try:
        assert brief is not None, message
        assert brief.document.source_type == "pdf"
        assert brief.document.page_count > 0
        assert os.path.isfile(pdf)
    finally:
        _remove_temp(pdf)


def test_methodology_keeps_limitations_visible():
    content = render_methodology()
    assert "Methodology and evaluation" in content
    assert "does not provide legal advice" in content
    assert "Macro-F1" in content
