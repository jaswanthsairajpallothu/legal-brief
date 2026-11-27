"""End-to-end orchestration. Stage names are real; no fake percentages. No raw judgment text is logged."""
from __future__ import annotations
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional
from .config import (Config, LABEL_TO_SECTION, SECTION_ORDER, SECTION_TITLES, DISCLAIMER_VERSION,
                     SECTION_AVAILABLE, SECTION_NOT_IDENTIFIED, STATUS_NONE)
from .models import BriefSection, CaseBrief, SummaryStatement
from .parser import parse_pdf, parse_text
from .preprocess import build_units
from .classifier import RoleClassifier
from .ranking import centrality_rank
from .summarizer import make_summarizer
from .linker import link_statements
from .metadata import extract_metadata

log = logging.getLogger("lbrief")          # logs carry counts/timings only, never document text

STAGES = ["Validating and extracting pages", "Preparing sentences", "Classifying sections",
          "Selecting source evidence", "Generating summaries", "Linking evidence"]


def _select(units, preds, cfg: Config) -> dict[str, list]:
    by_sec: dict[str, list] = {k: [] for k in SECTION_ORDER}
    pmap = {p.source_id: p for p in preds}
    for u in units:
        sec = LABEL_TO_SECTION.get(pmap[u.source_id].predicted_label)
        if sec:
            by_sec[sec].append(u)
    for sec, us in by_sec.items():                     # cap, centrality-ranked, then restore document order
        if len(us) > cfg.max_units_per_section:
            keep = sorted(centrality_rank([u.normalized_text for u in us])[: cfg.max_units_per_section])
            by_sec[sec] = [us[i] for i in keep]
    return by_sec


def generate_brief(data: Optional[bytes] = None, filename: str = "judgment.pdf", pasted_text: Optional[str] = None,
                   cfg: Optional[Config] = None, classifier: Optional[RoleClassifier] = None, summarizer=None,
                   progress: Optional[Callable[[str], None]] = None) -> CaseBrief:
    cfg = cfg or Config()
    timings: dict = {}
    t0 = time.perf_counter()

    def stage(name: str):
        nonlocal t0
        if progress:
            progress(name)
        return name

    def done(name: str):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()

    s = stage(STAGES[0])
    doc, pages = parse_pdf(data, filename, cfg) if pasted_text is None else parse_text(pasted_text, cfg)
    done(s)
    s = stage(STAGES[1])
    units = build_units(doc.document_id, pages, cfg)
    if not units:
        from .parser import InputError
        raise InputError("No usable sentences could be extracted from this document.")
    metadata = extract_metadata(pages)
    done(s)
    s = stage(STAGES[2])
    classifier = classifier or RoleClassifier(cfg)
    preds = classifier.predict(units)
    done(s)
    s = stage(STAGES[3])
    by_sec = _select(units, preds, cfg)
    done(s)

    summarizer = summarizer or make_summarizer(cfg)
    evidence: dict = {}
    sections: list[BriefSection] = []
    t_gen = t_link = 0.0
    stage(STAGES[4])
    t0 = time.perf_counter()
    gen_cache = {}
    for key in SECTION_ORDER:
        cands = by_sec[key]
        if len(cands) < cfg.min_units_per_section:
            continue
        gen_cache[key] = summarizer.summarize(key, cands)
    done(STAGES[4])
    stage(STAGES[5])
    for key in SECTION_ORDER:
        title = SECTION_TITLES[key]
        if key not in gen_cache:
            sections.append(BriefSection(key, title, [], SECTION_NOT_IDENTIFIED,
                                         "No source passages were confidently identified for this section."))
            continue
        stmts = link_statements(key, gen_cache[key], by_sec[key], cfg, evidence, f"{doc.document_id}-{key}")
        if not stmts:
            sections.append(BriefSection(key, title, [], SECTION_NOT_IDENTIFIED,
                                         "The summarizer returned no usable output for this section."))
        else:
            sections.append(BriefSection(key, title, stmts, SECTION_AVAILABLE))
    done(STAGES[5])

    warnings = []
    if doc.pages_without_text:
        warnings.append("Pages without extractable text were skipped: " + ", ".join(map(str, doc.pages_without_text)))
    if getattr(classifier, "is_interim", False):
        warnings.append("Section detection uses an interim classifier trained on 50 labelled Supreme Court judgments (JURIX 2019); "
                        "Issue detection and petitioner/respondent attribution are rule-based and unevaluated.")
    elif not classifier.is_trained:
        warnings.append(f"Section detection used the development heuristic ({classifier.version}), not the trained classifier.")
    log.info("brief generated: pages=%d units=%d timings=%s", doc.page_count, len(units), timings)
    return CaseBrief(
        brief_id=uuid.uuid4().hex[:12], document=doc, metadata=metadata, sections=sections, evidence=evidence,
        units={u.source_id: u for u in units}, roles={p.source_id: p for p in preds},
        summarizer_id=summarizer.model_id, classifier_version=classifier.version,
        generation_params=summarizer.params, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        disclaimer_version=DISCLAIMER_VERSION, timings=timings, warnings=warnings)
