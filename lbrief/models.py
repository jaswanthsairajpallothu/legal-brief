"""Data objects shared by every module (PRD/SRS section 10). The UI and PDF export both consume CaseBrief."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class PageText:
    document_id: str
    page_number: int
    raw_text: str
    normalized_text: str = ""
    extraction_status: str = "ok"          # ok | no_text


@dataclass
class Document:
    document_id: str
    filename: str
    page_count: int
    source_type: str = "pdf"               # pdf | pasted
    extraction_status: str = "ok"
    pages_without_text: list = field(default_factory=list)


@dataclass
class SourceUnit:
    source_id: str
    document_id: str
    page_number: int
    unit_index: int
    raw_text: str
    normalized_text: str
    start_offset: int                      # offsets into PageText.raw_text
    end_offset: int


@dataclass
class RolePrediction:
    source_id: str
    predicted_label: str
    raw_score: float
    classifier_version: str


@dataclass
class EvidenceItem:
    evidence_id: str
    source_id: str
    page_number: int
    source_excerpt: str
    retrieval_score: float
    retrieval_method: str = "tfidf-cosine"


@dataclass
class SummaryStatement:
    statement_id: str
    section_key: str
    text: str
    evidence_ids: list = field(default_factory=list)
    evidence_status: str = ""
    flags: list = field(default_factory=list)   # human-readable review reasons


@dataclass
class BriefSection:
    section_key: str
    title: str
    statements: list = field(default_factory=list)
    section_status: str = ""
    note: str = ""


@dataclass
class CaseBrief:
    brief_id: str
    document: Document
    metadata: dict
    sections: list
    evidence: dict                          # evidence_id -> EvidenceItem
    units: dict                             # source_id -> SourceUnit (for source inspection)
    roles: dict                             # source_id -> RolePrediction
    summarizer_id: str
    classifier_version: str
    generation_params: dict
    generated_at: str
    disclaimer_version: str
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
