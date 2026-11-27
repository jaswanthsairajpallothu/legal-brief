"""Statement-level evidence linking + review flags (SRS section 6).

RETRIEVAL IS NOT VERIFICATION. 'Evidence located' only means a candidate passage scored above a threshold.
Thresholds are uncalibrated defaults until you calibrate them against your human audit."""
from __future__ import annotations
import re
from .config import Config, STATUS_LOCATED, STATUS_REVIEW, STATUS_NONE
from .models import SourceUnit, EvidenceItem, SummaryStatement
from .ranking import similarity_matrix

_NEG = re.compile(r"\b(not|no|never|neither|nor|without|unless|cannot|can't|didn't|did not|except)\b", re.I)
_NUM = re.compile(r"\b\d[\d,./-]*\d\b|\b\d\b")
_CAPS = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-zA-Z]{2,}\b")


def _norm_num(s: str) -> str:
    return s.replace(",", "").strip(".")


def detail_flags(statement: str, source_text: str) -> list[str]:
    """Warnings only (never proof of error): numbers/names absent from candidates, or negation mismatch."""
    flags = []
    src_nums = {_norm_num(n) for n in _NUM.findall(source_text)}
    missing = sorted({_norm_num(n) for n in _NUM.findall(statement)} - src_nums)
    if missing:
        flags.append("Number/date not found in linked passage: " + ", ".join(missing[:5]))
    src_low = source_text.lower()
    names = {w for w in _CAPS.findall(statement) if w.lower() not in src_low}
    if names:
        flags.append("Name/term not found in linked passage: " + ", ".join(sorted(names)[:5]))
    if _NEG.search(statement) and not _NEG.search(source_text):
        flags.append("Statement contains negation/qualifier absent from linked passage")
    return flags


def link_statements(section_key: str, statements: list[str], candidates: list[SourceUnit], cfg: Config,
                    evidence_store: dict, id_prefix: str) -> list[SummaryStatement]:
    corpus = [u.normalized_text for u in candidates]
    sim = similarity_matrix(statements, corpus)
    out = []
    for i, text in enumerate(statements):
        sid = f"{id_prefix}-st{i}"
        if not corpus:
            out.append(SummaryStatement(sid, section_key, text, [], STATUS_NONE, ["No source passages in this section"]))
            continue
        row = sim[i]
        order = row.argsort()[::-1][: cfg.link_top_k]
        picked = [j for j in order if row[j] >= cfg.link_weak_threshold]
        if not picked:
            out.append(SummaryStatement(sid, section_key, text, [], STATUS_NONE, ["No passage passed the retrieval threshold"]))
            continue
        ev_ids = []
        for rank, j in enumerate(picked):
            u = candidates[j]
            eid = f"{sid}-ev{rank}"
            evidence_store[eid] = EvidenceItem(eid, u.source_id, u.page_number, u.raw_text.strip(), float(row[j]))
            ev_ids.append(eid)
        joined = " ".join(candidates[j].normalized_text for j in picked)
        flags = detail_flags(text, joined)
        top = float(row[picked[0]])
        status = STATUS_LOCATED if (top >= cfg.link_strong_threshold and not flags) else STATUS_REVIEW
        if top < cfg.link_strong_threshold:
            flags.insert(0, "Weak lexical match to source")
        out.append(SummaryStatement(sid, section_key, text, ev_ids, status, flags))
    return out
