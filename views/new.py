import logging
import streamlit as st
from lbrief.config import DISCLAIMER, STATUS_LOCATED, STATUS_REVIEW, STATUS_NONE
from lbrief.parser import InputError
from lbrief.pipeline import generate_brief, STAGES
from lbrief.exporter import build_pdf
from lbrief import insights
from ui.common import cfg, ICON, STAGE_HELP, load_models, e, page_label
import pandas as pd

def page_new():
    st.markdown("<div class='eyebrow'>For Indian court judgments</div>", unsafe_allow_html=True)
    st.title("Create a case brief")
    st.markdown(
        "<div class='hero-copy'>Get oriented faster with a structured summary, then check each statement "
        "against passages from the original judgment.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='process-steps' aria-label='How it works'>"
        "<div class='process-step'><span class='step-number' aria-hidden='true'>01</span><div>"
        "<div class='step-title'>Add a judgment</div><div class='step-copy'>Upload a text-based PDF or paste the text.</div></div></div>"
        "<div class='process-step'><span class='step-number' aria-hidden='true'>02</span><div>"
        "<div class='step-title'>Get the key sections</div><div class='step-copy'>Review facts, arguments, reasoning and decision.</div></div></div>"
        "<div class='process-step'><span class='step-number' aria-hidden='true'>03</span><div>"
        "<div class='step-title'>Check the original</div><div class='step-copy'>Open candidate passages and review anything flagged.</div></div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if cfg.hosted:
        st.warning(
            "This hosted demo processes documents on the hosting server. Do not upload confidential, sealed, privileged "
            "or sensitive material. Files are not stored after processing."
        )
    else:
        st.markdown(
            "<div class='trust-note'><span class='trust-dot' aria-hidden='true'></span><div>"
            "<strong>Private by default</strong>Processed locally on this machine. Uploaded content stays in memory "
            "for this session and is not written to disk.</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='section-intro'><div class='eyebrow'>Get started</div>"
        "<p>Choose the format that works best for you.</p></div>",
        unsafe_allow_html=True,
    )
    tab_pdf, tab_txt = st.tabs(["Upload PDF", "Paste text"])
    with tab_pdf:
        up = st.file_uploader("Choose a judgment PDF", type=["pdf"], help="Use an English PDF with selectable text.")
        st.caption(
            f"PDF with selectable English text | Up to {cfg.max_upload_mb} MB | "
            f"Up to {cfg.max_pages} pages. Scanned or image-only PDFs are not supported."
        )
    with tab_txt:
        txt = st.text_area(
            "Paste judgment text",
            height=220,
            help=f"Paste at least {cfg.min_paste_chars} characters. Page references are unavailable for pasted text.",
            placeholder="Paste the judgment text here...",
        )
        st.caption(f"{len(txt):,} characters | At least {cfg.min_paste_chars:,} required | Page references unavailable")
    if not st.button("Generate my case brief", type="primary"):
        return
    use_pdf = up is not None
    if not use_pdf and not txt.strip():
        st.error("Upload a PDF or paste judgment text first.")
        return
    box = st.empty()
    done: list = []

    def render(current=None):
        rows = []
        for s in STAGES:
            mark = "✓" if s in done else ("▶" if s == current else "○")
            rows.append(f"{mark} **{s}**" + (f" - {STAGE_HELP[s]}" if s == current else ""))
        box.markdown("  \n".join(rows))

    def progress(name):
        if name in STAGES and STAGES.index(name) > 0:
            done.append(STAGES[STAGES.index(name) - 1])
        render(name)

    try:
        with st.spinner("Processing..."):
            clf, summ = load_models()
            render(STAGES[0])
            result = generate_brief(data=up.getvalue() if use_pdf else None, filename=up.name if use_pdf else "pasted text",
                                    pasted_text=None if use_pdf else txt, cfg=cfg, classifier=clf, summarizer=summ, progress=progress)
        st.session_state.update(brief=result)
        for k in ("sel", "pdf", "ev_idx"):
            st.session_state.pop(k, None)
        st.switch_page("views/brief.py")
    except InputError as ex:
        box.empty()
        st.error(str(ex))
    except Exception as ex:                                       # no stack traces/paths for users
        if type(ex).__name__ in ("StopException", "RerunException"):
            raise
        logging.getLogger("lbrief").error("processing failed: %s", type(ex).__name__)
        box.empty()
        st.error("Processing failed. The document may have an unusual layout, or the model could not be loaded. "
                 "Try another file; details are in the server log (no document text is logged).")


page_new()
