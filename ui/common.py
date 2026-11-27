"""Shared UI helpers and constants."""
import html
import logging
from pathlib import Path
import streamlit as st
from lbrief.config import Config, STATUS_LOCATED, STATUS_REVIEW, STATUS_NONE
from lbrief.classifier import RoleClassifier
from lbrief.summarizer import make_summarizer
from lbrief.pipeline import STAGES

cfg = Config()
ROOT = Path(__file__).resolve().parent.parent
ICON = {STATUS_LOCATED: ("●", ""), STATUS_REVIEW: ("▲", "review"), STATUS_NONE: ("○", "none")}
STAGE_HELP = {
    STAGES[0]: "Checking the file and reading text page by page.",
    STAGES[1]: "Splitting text into sentences while keeping page numbers.",
    STAGES[2]: "Labelling each sentence: facts, issue, arguments, reasoning, decision.",
    STAGES[3]: "Grouping the relevant source passages for each brief section.",
    STAGES[4]: "Writing section summaries from the selected passages only.",
    STAGES[5]: "Matching each summary statement to candidate source passages.",
}


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return RoleClassifier(cfg), make_summarizer(cfg)          # exactly one classifier, one summarizer


def e(x):
    return html.escape(str(x))


def page_label(b, ev):
    return "Pasted text" if b.document.source_type == "pasted" else f"Page {ev.page_number}"


def header():
    mode = "Hosted demo: server-side processing" if cfg.hosted else "Local processing"
    st.markdown(
        "<div class='topbar'><div class='brand-lockup'><span class='brand-mark' aria-hidden='true'>CB</span>"
        "<div><div class='brand-name'>CaseBrief</div>"
        "<div class='brand-sub'>A clearer way to read judgments</div></div></div>"
        f"<span class='pill'>{e(mode)}</span></div>",
        unsafe_allow_html=True,
    )

