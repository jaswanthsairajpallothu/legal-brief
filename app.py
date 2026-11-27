"""Streamlit entrypoint (PRD section 12): New Judgment -> Current Case Brief (+ evidence panel) -> Methodology.
Uploaded bytes are processed in memory only; nothing is written to disk or logs."""
import logging
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Court Judgment & Legal Case Summarizer", layout="wide")
logging.basicConfig(level=logging.INFO)
st.markdown(f"<style>{(Path(__file__).parent / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)
from ui.common import header  # noqa: E402

header()
st.navigation([
    st.Page("views/new.py", title="New Judgment", url_path="new", default=True),
    st.Page("views/brief.py", title="Current Case Brief", url_path="brief"),
    st.Page("views/methodology.py", title="Methodology", url_path="methodology"),
]).run()
