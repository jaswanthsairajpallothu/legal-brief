import logging
import streamlit as st
from lbrief.config import DISCLAIMER, STATUS_LOCATED, STATUS_REVIEW, STATUS_NONE
from lbrief.parser import InputError
from lbrief.pipeline import generate_brief, STAGES
from lbrief.exporter import build_pdf
from lbrief import insights
from ui.common import cfg, ICON, STAGE_HELP, load_models, e, page_label
import pandas as pd

def page_brief():
    brief = st.session_state.get("brief")
    if brief is None:
        st.markdown("<div class='eyebrow'>Your workspace</div>", unsafe_allow_html=True)
        st.title("Current case brief")
        st.info("Your brief will appear here after you add a judgment.")
        st.page_link("views/new.py", label="Start with a judgment", icon=":material/upload_file:")
        return
    md = brief.metadata
    st.title(md.get("title") or "Untitled judgment")
    nd = lambda k: e(md.get(k) or "Not detected")
    st.markdown(
        f"<div class='case-meta'><span>{e(brief.document.filename)}</span>"
        f"<span>{brief.document.page_count} page(s)</span><span>Court: {nd('court')}</span>"
        f"<span>Case no.: {nd('case_number')}</span><span>Year: {nd('year')}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, _ = st.columns([1.1, 1.5, 4])
    if "pdf" not in st.session_state:
        st.session_state["pdf"] = build_pdf(brief)
    c1.download_button("Download PDF", st.session_state["pdf"], file_name="case_brief.pdf", mime="application/pdf", type="primary")
    if c2.button("Process another judgment"):
        for k in ("brief", "sel", "pdf", "ev_idx"):
            st.session_state.pop(k, None)
        st.switch_page("views/new.py")
    st.markdown(
        f"<div class='notice' role='note'><strong>{e(DISCLAIMER)}</strong> "
        "A located passage is a candidate for your review, not proof that a summary statement is correct.</div>",
        unsafe_allow_html=True,
    )
    for w in brief.warnings:
        st.warning(w)
    if brief.summarizer_id.startswith("extractive"):
        st.caption("Extractive summary · Statements are verbatim sentences from the judgment.")

    statements = [statement for section in brief.sections for statement in section.statements]
    needs_review = sum(s.evidence_status in (STATUS_REVIEW, STATUS_NONE) for s in statements)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sections in this brief", len(brief.sections))
    m2.metric("Summary statements", len(statements))
    m3.metric("Need a closer look", needs_review)
    st.caption("Use “View source” on any statement to compare it with the original passage.")

    main, side = st.columns([0.68, 0.32], gap="large")
    sel = st.session_state.get("sel")
    with main:
        for sec in brief.sections:
            st.markdown(f"<div class='sec-head'>{e(sec.title)}</div>", unsafe_allow_html=True)
            if not sec.statements:
                st.markdown(f"<div class='status none'>○ {e(sec.section_status)}</div><div class='meta' style='margin-left:10px'>{e(sec.note)}</div>",
                            unsafe_allow_html=True)
                continue
            for s in sec.statements:
                icon, cls = ICON.get(s.evidence_status, ("○", "none"))
                labels = list(dict.fromkeys(page_label(brief, brief.evidence[x]) for x in s.evidence_ids))      # unique, in order
                chips = "".join(f"<span class='chip'>{e(l)}</span>" for l in labels)
                st.markdown(f"<div class='stmt {'sel' if s.statement_id == sel else ''}'>{e(s.text)}</div>"
                            f"<div class='status {cls}'>{icon} {e(s.evidence_status)}</div>{chips}", unsafe_allow_html=True)
                if st.button("View source", key=f"v_{s.statement_id}"):
                    st.session_state["sel"], st.session_state["ev_idx"] = s.statement_id, 0
                    st.rerun()
    with side:
        st.markdown("<div class='ev-panel'></div>", unsafe_allow_html=True)
        render_evidence(brief, sel)
    st.write("")
    insights_panel(brief)


def render_evidence(brief, sel):
    st.subheader("Source evidence")
    stmt = next((s for sec in brief.sections for s in sec.statements if s.statement_id == sel), None)
    if stmt is None:
        st.caption("Choose “View source” beside a statement to compare it with the original passage.")
        return
    st.markdown("<div class='evlabel'>Generated statement</div>", unsafe_allow_html=True)
    st.write(stmt.text)
    for f in stmt.flags:
        st.markdown(f"<div class='status review'>▲ Review note: {e(f)}</div>", unsafe_allow_html=True)
    if not stmt.evidence_ids:
        st.markdown("<div class='status none'>○ No source passage was located for this statement.</div>", unsafe_allow_html=True)
        return
    n = len(stmt.evidence_ids)
    i = min(st.session_state.get("ev_idx", 0), n - 1)
    ev = brief.evidence[stmt.evidence_ids[i]]
    role = brief.roles[ev.source_id].predicted_label
    label = "Source text (original)"
    st.markdown(f"<div class='evlabel'>{label} · {i + 1} of {n}</div>"
                f"<span class='chip' style='margin-left:0'>{e(page_label(brief, ev))}</span><span class='role'>Predicted role: {e(role)}</span>"
                f"<div class='evidence'>{e(ev.source_excerpt)}</div>"
                f"<div class='meta'>Lexical match {ev.retrieval_score:.2f}. A match score is not proof that the statement is correct.</div>",
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("Previous source", disabled=i == 0, key="prev_src"):
        st.session_state["ev_idx"] = i - 1
        st.rerun()
    if b2.button("Next source", disabled=i >= n - 1, key="next_src"):
        st.session_state["ev_idx"] = i + 1
        st.rerun()
    order = list(brief.units)
    k = order.index(ev.source_id)
    with st.expander("Show surrounding text"):
        if k > 0:
            st.caption("Previous sentence")
            st.markdown(f"<div class='evidence'>{e(brief.units[order[k - 1]].raw_text.strip())}</div>", unsafe_allow_html=True)
        if k + 1 < len(order):
            st.caption("Next sentence")
            st.markdown(f"<div class='evidence'>{e(brief.units[order[k + 1]].raw_text.strip())}</div>", unsafe_allow_html=True)


def insights_panel(brief):
    with st.expander("NLP insights"):
        st.caption("Secondary view. General-purpose NER does not reliably detect every legal entity or statute.")
        stats = insights.basic_stats(brief.units, brief.roles, brief.document.page_count)
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages", stats["pages"]); m2.metric("Sentences", stats["sentences"]); m3.metric("Words", stats["words"])
        st.dataframe(pd.DataFrame({"Predicted role": list(stats["role_counts"]), "Sentences": list(stats["role_counts"].values())}),
                     hide_index=True, width="stretch")
        texts = [u.normalized_text for u in brief.units.values()]
        st.markdown("**Top TF-IDF terms:** " + (", ".join(insights.top_terms(texts)) or "n/a"))
        if st.checkbox("Detect named entities (spaCy, optional)"):
            ents, why = insights.entities(texts)
            if ents is None:
                st.info(why)
            elif not ents:
                st.write("No entities of the displayed types were found.")
            for lab, items in (ents or {}).items():
                st.markdown(f"**{lab}:** " + "; ".join(f"{t} ({c})" for t, c in items))


page_brief()
