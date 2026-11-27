"""Gradio presentation layer for the Judora case-brief workflow."""
from __future__ import annotations

from functools import lru_cache
import html
import json
import logging
import os
from pathlib import Path
import tempfile
import uuid

if os.environ.get("SPACE_ID"):
    os.environ.setdefault("LBRIEF_HOSTED", "1")

import gradio as gr

from lbrief import insights
from lbrief.classifier import RoleClassifier
from lbrief.config import (
    Config,
    DISCLAIMER,
    STATUS_LOCATED,
    STATUS_NONE,
    STATUS_REVIEW,
)
from lbrief.exporter import build_pdf
from lbrief.parser import InputError
from lbrief.pipeline import STAGES, generate_brief
from lbrief.summarizer import make_summarizer

ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = Path(os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "judora-gradio"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR))

cfg = Config()
log = logging.getLogger("lbrief")

STAGE_LABELS = {
    STAGES[0]: "Validating the document and extracting its text",
    STAGES[1]: "Preparing sentences and page references",
    STAGES[2]: "Identifying judgment sections",
    STAGES[3]: "Selecting candidate passages",
    STAGES[4]: "Preparing the section summaries",
    STAGES[5]: "Linking statements to source passages",
}
SECTION_NUMBER = {
    "facts": "01",
    "issues": "02",
    "arguments_petitioner": "03",
    "arguments_respondent": "04",
    "reasoning": "05",
    "decision": "06",
}


@lru_cache(maxsize=1)
def load_models():
    return RoleClassifier(cfg), make_summarizer(cfg)


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _page_label(brief, evidence) -> str:
    if brief.document.source_type == "pasted":
        return "Pasted text"
    return f"Page {evidence.page_number}"


def _temp_pdf_path(brief) -> str:
    path = TEMP_DIR / f"judora-{uuid.uuid4().hex}.pdf"
    path.write_bytes(build_pdf(brief))
    return str(path)


def _brief_from_inputs(pdf_file, pasted_text, progress=None):
    if not pdf_file and not (pasted_text or "").strip():
        raise InputError("Upload a judgment PDF or paste its text to get started.")

    if pdf_file:
        pdf_path = Path(pdf_file)
        if pdf_path.stat().st_size > cfg.max_upload_mb * 1024 * 1024:
            raise InputError(f"That file is over the {cfg.max_upload_mb} MB upload limit.")
        data = pdf_path.read_bytes()
        filename = pdf_path.name
        text = None
    else:
        if len(pasted_text.strip()) < cfg.min_paste_chars:
            raise InputError(f"Please paste at least {cfg.min_paste_chars} characters of judgment text.")
        data = None
        filename = "Pasted judgment text"
        text = pasted_text

    classifier, summarizer = load_models()

    def report_stage(stage):
        if progress is not None:
            progress(None, desc=STAGE_LABELS.get(stage, stage))

    return generate_brief(
        data=data,
        filename=filename,
        pasted_text=text,
        cfg=cfg,
        classifier=classifier,
        summarizer=summarizer,
        progress=report_stage,
    )


def process_document(pdf_file, pasted_text, progress=gr.Progress(track_tqdm=False)):
    try:
        brief = _brief_from_inputs(pdf_file, pasted_text, progress)
        return brief, "Brief ready. Open the Case brief tab to review the summary and its source passages.", _temp_pdf_path(brief)
    except InputError as exc:
        return None, f"**Check the document:** {_e(exc)}", None
    except Exception as exc:
        if type(exc).__name__ in ("StopException", "RerunException"):
            raise
        log.error("processing failed: %s", type(exc).__name__)
        return None, "We couldn't process this document. Try a text-based English PDF or paste the judgment text. Details are in the server log; document text is not logged.", None


def _status_badge(status: str) -> str:
    classes = {
        STATUS_LOCATED: "located",
        STATUS_REVIEW: "review",
        STATUS_NONE: "missing",
    }
    css_class = classes.get(status, "missing")
    return f"<span class='status-badge {css_class}'>{_e(status)}</span>"


def _source_choices(brief):
    choices = []
    for section in brief.sections:
        for statement in section.statements:
            excerpt = " ".join(statement.text.split())
            if len(excerpt) > 90:
                excerpt = excerpt[:87].rstrip() + "..."
            label = f"{section.title}: {excerpt}"
            choices.append((label, statement.statement_id))
    return choices


def _brief_summary_html(brief) -> str:
    sections = []
    source_rows = []
    seen_evidence = set()

    for section_index, section in enumerate(brief.sections, start=1):
        section_number = SECTION_NUMBER.get(section.section_key, f"{section_index:02d}")
        section_class = " final-verdict" if section.section_key == "decision" else ""
        if not section.statements:
            note = f"<p class='section-note'>{_e(section.note or section.section_status)}</p>"
            sections.append(
                f"<section class='summary-section{section_class}'>"
                f"<h3><span class='section-number'>{section_number}</span>{_e(section.title)}</h3>{note}</section>"
            )
            continue

        statement_html = []
        for statement in section.statements:
            labels = []
            for evidence_id in statement.evidence_ids:
                evidence = brief.evidence[evidence_id]
                label = _page_label(brief, evidence)
                if label not in labels:
                    labels.append(label)
                if evidence_id not in seen_evidence:
                    seen_evidence.add(evidence_id)
                    source_rows.append(
                        "<li class='reference-item'>"
                        f"<div class='reference-heading'>{_e(section.title)} <span>{_e(label)}</span></div>"
                        f"<p class='reference-for'>For: {_e(statement.text)}</p>"
                        f"<blockquote class='legal-document'>{_e(evidence.source_excerpt)}</blockquote>"
                        "</li>"
                    )

            refs = "".join(f"<span class='page-chip'>{_e(label)}</span>" for label in labels)
            flags = "".join(f"<li>{_e(flag)}</li>" for flag in statement.flags)
            flag_html = f"<ul class='review-notes'>{flags}</ul>" if flags else ""
            statement_html.append(
                "<article class='summary-statement'>"
                f"<p>{_e(statement.text)}</p>"
                f"<div class='statement-meta'>{_status_badge(statement.evidence_status)}{refs}</div>"
                f"{flag_html}</article>"
            )

        sections.append(
            f"<section class='summary-section{section_class}'>"
            f"<h3><span class='section-number'>{section_number}</span>{_e(section.title)}</h3>"
            f"{''.join(statement_html)}</section>"
        )

    references = (
        "<section class='source-references'><h3>Source references</h3>"
        "<p class='muted-copy'>Candidate passages are shown for comparison; they do not prove a statement is correct.</p>"
        f"<ol>{''.join(source_rows)}</ol></section>"
        if source_rows
        else ""
    )
    return (
        "<div class='judora-summary'>"
        f"{''.join(sections)}{references}"
        f"<p class='legal-disclaimer'>{_e(DISCLAIMER)}</p>"
        "</div>"
    )


def _metadata_html(brief) -> str:
    metadata = brief.metadata
    items = [
        ("Document", brief.document.filename),
        ("Pages", brief.document.page_count),
        ("Court", metadata.get("court") or "Not detected"),
        ("Case number", metadata.get("case_number") or "Not detected"),
        ("Year", metadata.get("year") or "Not detected"),
    ]
    return "<div class='case-metadata'>" + "".join(
        f"<div class='metadata-item'><span>{_e(label)}</span><strong>{_e(value)}</strong></div>"
        for label, value in items
    ) + "</div>"


def _warnings_html(brief) -> str:
    warnings = "".join(f"<li>{_e(warning)}</li>" for warning in brief.warnings)
    hosted = (
        "<div class='hosted-note'>This hosted app processes documents on the server. "
        "Do not upload confidential, sealed, privileged, or sensitive material.</div>"
        if cfg.hosted
        else ""
    )
    summary_mode = (
        "<div class='mode-note'>Extractive summary: each statement is a verbatim sentence from the judgment.</div>"
        if brief.summarizer_id.startswith("extractive")
        else ""
    )
    return (
        hosted
        + (f"<div class='warning-list'><ul>{warnings}</ul></div>" if warnings else "")
        + summary_mode
    )


def _brief_stats_html(brief) -> str:
    statements = [statement for section in brief.sections for statement in section.statements]
    needs_review = sum(s.evidence_status in (STATUS_REVIEW, STATUS_NONE) for s in statements)
    items = [
        ("Sections", len(brief.sections)),
        ("Statements", len(statements)),
        ("Need a closer look", needs_review),
    ]
    return "<div class='stat-grid'>" + "".join(
        f"<div class='stat-card'><span>{_e(label)}</span><strong>{value}</strong></div>"
        for label, value in items
    ) + "</div>"


def _empty_source_html() -> str:
    return (
        "<div class='empty-evidence'><h3>Check a source passage</h3>"
        "<p>Select a statement above to inspect its candidate source text, page, and review notes.</p></div>"
    )


def render_brief(brief, detect_entities=False):
    if brief is None:
        return (
            "<div class='empty-brief'><h2>Your case brief will appear here</h2>"
            "<p>Add a judgment in the New judgment tab to get started.</p></div>",
            "",
            "",
            "",
            gr.update(choices=[], value=None),
            _empty_source_html(),
            _insights_html(None, False),
        )
    return (
        f"<div class='brief-title'><div class='eyebrow'>Case overview</div><h2>{_e(brief.metadata.get('title') or 'Untitled judgment')}</h2></div>",
        _metadata_html(brief),
        _warnings_html(brief),
        _brief_stats_html(brief) + _brief_summary_html(brief),
        gr.update(choices=_source_choices(brief), value=None),
        _empty_source_html(),
        _insights_html(brief, detect_entities),
    )


def render_source(brief, statement_id):
    if brief is None or not statement_id:
        return _empty_source_html()

    statement = next(
        (
            current
            for section in brief.sections
            for current in section.statements
            if current.statement_id == statement_id
        ),
        None,
    )
    if statement is None:
        return "<div class='error-note'>That statement is no longer available. Generate the brief again.</div>"

    section = next(section for section in brief.sections if section.section_key == statement.section_key)
    evidence_html = []
    order = list(brief.units)
    for index, evidence_id in enumerate(statement.evidence_ids, start=1):
        evidence = brief.evidence[evidence_id]
        role = brief.roles[evidence.source_id].predicted_label
        position = order.index(evidence.source_id)
        surrounding = []
        if position > 0:
            surrounding.append(f"<p>{_e(brief.units[order[position - 1]].raw_text.strip())}</p>")
        surrounding.append(f"<p class='current-sentence'>{_e(brief.units[evidence.source_id].raw_text.strip())}</p>")
        if position + 1 < len(order):
            surrounding.append(f"<p>{_e(brief.units[order[position + 1]].raw_text.strip())}</p>")
        evidence_html.append(
            "<article class='source-card'>"
            f"<div class='source-label'>Original judgment | {_e(_page_label(brief, evidence))} | Candidate {index}</div>"
            f"<div class='source-meta'>Predicted role: {_e(role)}</div>"
            f"<blockquote class='legal-document'>{_e(evidence.source_excerpt)}</blockquote>"
            f"<p class='match-note'>Lexical match {evidence.retrieval_score:.2f}. A match score is not proof of correctness.</p>"
            "<details><summary>Show surrounding text</summary>"
            f"<div class='surrounding-text'>{''.join(surrounding)}</div></details>"
            "</article>"
        )

    flags = "".join(f"<li>{_e(flag)}</li>" for flag in statement.flags)
    flags_html = f"<ul class='review-notes'>{flags}</ul>" if flags else ""
    evidence_content = "".join(evidence_html) or (
        "<p class='error-note'>No candidate source passage was located for this statement.</p>"
    )
    return (
        "<div class='evidence-panel'>"
        f"<div class='source-label'>Judora summary | {_e(section.title)}</div>"
        f"<p class='selected-statement'>{_e(statement.text)}</p>"
        f"{_status_badge(statement.evidence_status)}{flags_html}{evidence_content}</div>"
    )


def _insights_html(brief, detect_entities):
    if brief is None:
        return "<p class='muted-copy'>Create a brief to see document-level insights.</p>"
    stats = insights.basic_stats(brief.units, brief.roles, brief.document.page_count)
    role_rows = "".join(
        f"<tr><td>{_e(role)}</td><td>{count}</td></tr>"
        for role, count in sorted(stats["role_counts"].items())
    )
    terms = insights.top_terms([unit.normalized_text for unit in brief.units.values()])
    content = (
        "<div class='insight-stats'>"
        f"<span><strong>{stats['pages']}</strong> Pages</span>"
        f"<span><strong>{stats['sentences']}</strong> Sentences</span>"
        f"<span><strong>{stats['words']}</strong> Words</span></div>"
        "<h4>Predicted section roles</h4>"
        f"<table class='data-table'><thead><tr><th>Role</th><th>Sentences</th></tr></thead><tbody>{role_rows}</tbody></table>"
        f"<p class='terms-line'><strong>Top TF-IDF terms:</strong> {_e(', '.join(terms) or 'Not available')}</p>"
    )
    if detect_entities:
        texts = [unit.normalized_text for unit in brief.units.values()]
        entities, reason = insights.entities(texts)
        if entities is None:
            content += f"<p class='info-note'>{_e(reason)}</p>"
        elif not entities:
            content += "<p class='muted-copy'>No entities of the displayed types were found.</p>"
        else:
            rows = []
            for label, items in entities.items():
                rows.append(
                    f"<tr><td>{_e(label)}</td><td>{_e('; '.join(f'{text} ({count})' for text, count in items))}</td></tr>"
                )
            content += (
                "<h4>Named entities</h4>"
                "<p class='muted-copy'>General-purpose NER may miss legal entities and statutes.</p>"
                f"<table class='data-table'><tbody>{''.join(rows)}</tbody></table>"
            )
    else:
        content += "<p class='muted-copy'>Named-entity detection is optional and may miss legal entities or statutes.</p>"
    return content


def _table(headers, rows):
    head = "".join(f"<th>{_e(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-scroll'><table class='data-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _read_report(name):
    path = ROOT / "reports" / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metrics_html(data):
    metrics = [
        ("Macro-F1", data.get("macro_f1")),
        ("Weighted-F1", data.get("weighted_f1")),
        ("Accuracy", data.get("accuracy_secondary")),
        ("Sentences", data.get("n")),
    ]
    cards = []
    for label, value in metrics:
        display = f"{value:.3f}" if isinstance(value, (int, float)) and label != "Sentences" else value
        cards.append(f"<div class='metric-card'><span>{_e(label)}</span><strong>{_e(display if display is not None else 'n/a')}</strong></div>")
    per_class = data.get("per_class", {})
    per_class_rows = [
        (label, values.get("precision"), values.get("recall"), values.get("f1-score"), values.get("support"))
        for label, values in per_class.items()
    ]
    result = f"<div class='metric-grid'>{''.join(cards)}</div>"
    if per_class_rows:
        result += _table(("Role", "Precision", "Recall", "F1", "Support"), per_class_rows)
    labels = data.get("confusion_labels", [])
    matrix = data.get("confusion_matrix", [])
    if labels and matrix:
        result += "<h4>Confusion matrix | rows are actual labels, columns are predicted labels</h4>"
        result += _table(["Actual / predicted", *labels], [[label, *row] for label, row in zip(labels, matrix)])
    return result


def render_methodology(brief=None):
    parts = [
        "<div class='methodology-page'>"
        "<div class='eyebrow'>How Judora works</div>"
        "<h2>Methodology and evaluation</h2>"
        "<p class='lede'>Transparency matters. Retrieval and overlap metrics do not establish legal correctness or summary faithfulness.</p>"
    ]
    if brief:
        timing_rows = [(stage, f"{seconds:.3f}") for stage, seconds in brief.timings.items()]
        parts.append(
            "<section class='method-section'><h3>Current run</h3>"
            f"<p>Section detection: {_e(brief.classifier_version)} | Summarizer: {_e(brief.summarizer_id)}</p>"
            + _table(("Stage", "Seconds"), timing_rows)
            + "</section>"
        )

    test_report = _read_report("classifier_test.json")
    if test_report:
        parts.append(
            "<section class='method-section'><h3>Rhetorical-role classifier: LegalSeg test split</h3>"
            f"<p class='muted-copy'>Model {_e(test_report.get('model', 'n/a'))} | Dataset {_e(test_report.get('dataset', 'n/a'))}</p>"
            f"{_metrics_html(test_report['test'])}</section>"
        )

    cv_report = _read_report("classifier_cv.json")
    if cv_report:
        validation = cv_report["validation"]
        parts.append(
            "<section class='method-section'><h3>Interim classifier: document-level cross-validation</h3>"
            f"<p>{_e(cv_report.get('note', ''))}</p>"
            "<div class='metric-grid'>"
            f"<div class='metric-card'><span>Selected Macro-F1</span><strong>{validation['macro_f1']:.3f}</strong></div>"
            f"<div class='metric-card'><span>Heuristic baseline Macro-F1</span><strong>{cv_report['heuristic_baseline']['macro_f1']:.3f}</strong></div>"
            f"<div class='metric-card'><span>Selected configuration</span><strong>{_e(cv_report.get('cv_best', 'n/a'))}</strong></div></div>"
            f"<p class='muted-copy'>Dataset: {_e(cv_report.get('meta', {}).get('dataset', 'n/a'))}. Configuration was selected on the same cross-validation folds, so results may be optimistic.</p>"
            f"{_metrics_html(validation)}</section>"
        )
    elif not test_report:
        val_report = _read_report("classifier_val.json")
        if val_report:
            parts.append(
                "<section class='method-section'><h3>Rhetorical-role classifier: validation split</h3>"
                f"{_metrics_html(val_report['validation'])}</section>"
            )

    rouge = _read_report("rouge.json")
    if rouge:
        rows = [[f"This pipeline ({rouge.get('summarizer', '')})", rouge.get("n", "n/a"),
                 rouge.get("rouge1", "n/a"), rouge.get("rouge2", "n/a"), rouge.get("rougeL", "n/a")]]
        if rouge.get("lead_baseline"):
            baseline = rouge["lead_baseline"]
            rows.append(["Lead-sentences baseline", rouge.get("n", "n/a"), baseline.get("rouge1", "n/a"),
                         baseline.get("rouge2", "n/a"), baseline.get("rougeL", "n/a")])
        parts.append(
            "<section class='method-section'><h3>Summarization overlap (ROUGE)</h3>"
            + _table(("System", "Samples", "ROUGE-1", "ROUGE-2", "ROUGE-L"), rows)
            + f"<p class='muted-copy'>{_e(rouge.get('note', 'Overlap is not evidence of faithfulness.'))}</p></section>"
        )

    locator = _read_report("locator_check.json")
    if locator:
        parts.append(
            "<section class='method-section'><h3>Evidence locator check</h3>"
            f"<p>{locator['excerpt_found_on_cited_page']} of {locator['evidence_items']} excerpts appeared on the cited page; "
            f"{locator['resolve_to_source_unit']} resolved to a stored source unit.</p>"
            f"<p class='muted-copy'>{_e(locator.get('caveat', ''))}</p></section>"
        )

    parts.extend(
        [
            "<section class='method-section'><h3>What this system does not claim</h3>"
            "<ul><li>It does not provide legal advice or verify legal correctness.</li>"
            "<li>“Evidence located” means a candidate passage was retrieved; it does not prove a statement is correct.</li>"
            "<li>Issue detection and party attribution use unevaluated rules.</li>"
            "<li>Automated overlap scores and a general-purpose NER model are not legal validation.</li></ul></section>",
            "<section class='method-section'><h3>Data and models</h3>"
            "<p>The default summarizer is extractive. The interim section classifier is trained on a small labeled corpus. "
            "Third-party datasets and transformer weights are not bundled; check their current terms before use.</p></section>",
            "</div>",
        ]
    )
    return "".join(parts)


def create_app():
    css = (ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    with gr.Blocks(
        title="Judora | Complex judgments. Clear understanding.",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
        delete_cache=(3600, 3600),
    ) as demo:
        brief_state = gr.State(None)
        gr.HTML(
            "<header class='judora-header'><div class='brand-copy'>"
            "<div class='brand-name'>Judora</div>"
            "<div class='brand-tagline'>Complex judgments. Clear understanding.</div>"
            "</div><div class='header-note'>Evidence-linked case briefs</div></header>"
        )
        gr.Markdown(
            "<div class='hero'><div class='eyebrow'>LEGAL JUDGMENT WORKSPACE</div>"
            "<h1>Understand the judgment.<br>Keep the evidence close.</h1>"
            "<p>Turn lengthy Indian court judgments into structured case briefs, with candidate source passages to check as you read.</p></div>"
        )

        with gr.Tabs():
            with gr.Tab("New judgment", id="new-judgment"):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.Markdown("### Start with a judgment")
                        gr.Markdown("Upload a text-based English PDF or paste the judgment text. Scanned or image-only PDFs are not supported.")
                        with gr.Tabs():
                            with gr.Tab("Upload PDF", id="upload-pdf"):
                                pdf_input = gr.File(
                                    label="Judgment PDF",
                                    file_types=[".pdf"],
                                    type="filepath",
                                    elem_classes=["judora-upload"],
                                )
                                gr.Markdown(
                                    f"PDF only | Up to {cfg.max_upload_mb} MB and {cfg.max_pages} pages | "
                                    "Selectable English text required",
                                    elem_classes=["help-copy"],
                                )
                            with gr.Tab("Paste text", id="paste-text"):
                                text_input = gr.Textbox(
                                    label="Judgment text",
                                    placeholder="Paste the judgment text here...",
                                    lines=12,
                                    max_lines=24,
                                    elem_classes=["judgment-input"],
                                )
                                gr.Markdown(
                                    f"At least {cfg.min_paste_chars} characters | Page references are unavailable for pasted text",
                                    elem_classes=["help-copy"],
                                )
                    with gr.Column(scale=5):
                        gr.HTML(
                            "<aside class='workflow-card'><div class='eyebrow'>Your workflow</div>"
                            "<ol><li><strong>Add a judgment</strong><span>Upload a PDF or paste text.</span></li>"
                            "<li><strong>Review the structure</strong><span>Facts, issues, arguments, reasoning and decision.</span></li>"
                            "<li><strong>Check the source</strong><span>Inspect candidate passages and review flags.</span></li></ol></aside>"
                        )
                        if cfg.hosted:
                            gr.Markdown(
                                "Hosted processing: documents are processed on the server. Do not upload confidential, "
                                "sealed, privileged, or sensitive material.",
                                elem_classes=["privacy-warning"],
                            )
                        else:
                            gr.Markdown(
                                "Local processing: source files are handled in memory and are not written to the project.",
                                elem_classes=["privacy-note"],
                            )
                with gr.Row(elem_classes=["action-row"]):
                    process_button = gr.Button("Generate case brief", variant="primary", elem_classes=["primary-action"])
                    generation_status = gr.Markdown(elem_classes=["generation-status"])

            with gr.Tab("Case brief", id="case-brief"):
                brief_title = gr.HTML(
                    "<div class='empty-brief'><h2>Your case brief will appear here</h2>"
                    "<p>Add a judgment in the New judgment tab to get started.</p></div>"
                )
                brief_metadata = gr.HTML()
                brief_warnings = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=8):
                        brief_content = gr.HTML()
                    with gr.Column(scale=4, elem_classes=["evidence-column"]):
                        gr.Markdown("### Trace a statement")
                        statement_select = gr.Dropdown(
                            label="Choose a summary statement",
                            choices=[],
                            interactive=True,
                            elem_classes=["source-selector"],
                        )
                        evidence_display = gr.HTML(_empty_source_html())
                        pdf_download = gr.File(label="Download case brief PDF", interactive=False)
                with gr.Accordion("Document insights", open=False):
                    gr.Markdown("A secondary view of the document. These general-purpose tools are not legal validation.")
                    detect_entities = gr.Checkbox(label="Detect named entities (optional)", value=False)
                    insights_display = gr.HTML(_insights_html(None, False))

            with gr.Tab("Methodology", id="methodology"):
                methodology_display = gr.HTML(render_methodology())

        process_event = process_button.click(
            fn=process_document,
            inputs=[pdf_input, text_input],
            outputs=[brief_state, generation_status, pdf_download],
            api_name=False,
        )
        process_event.then(
            fn=render_brief,
            inputs=[brief_state, detect_entities],
            outputs=[
                brief_title,
                brief_metadata,
                brief_warnings,
                brief_content,
                statement_select,
                evidence_display,
                insights_display,
            ],
            api_name=False,
        ).then(
            fn=render_methodology,
            inputs=brief_state,
            outputs=methodology_display,
            api_name=False,
        )
        statement_select.change(
            fn=render_source,
            inputs=[brief_state, statement_select],
            outputs=evidence_display,
            api_name=False,
        )
        detect_entities.change(
            fn=_insights_html,
            inputs=[brief_state, detect_entities],
            outputs=insights_display,
            api_name=False,
        )
    return demo


def launch(demo):
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        max_file_size=f"{cfg.max_upload_mb}mb",
        share=False,
    )
