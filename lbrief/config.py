"""Central configuration. Every threshold here is a DEFAULT, not a validated value.
Calibrate link thresholds on your audit sample (scripts/audit.py) before trusting them."""
from __future__ import annotations
from dataclasses import dataclass, field
import os

# LegalSeg label set -> case brief section keys
LABELS = ["Facts", "Issue", "AoP", "AoR", "Reasoning", "Decision", "None"]
LABEL_TO_SECTION = {
    "Facts": "facts", "Issue": "issues",
    "AoP": "arguments_petitioner", "AoR": "arguments_respondent",
    "Reasoning": "reasoning", "Decision": "decision",
}
SECTION_ORDER = ["facts", "issues", "arguments_petitioner", "arguments_respondent", "reasoning", "decision"]
SECTION_TITLES = {
    "facts": "Facts", "issues": "Issues",
    "arguments_petitioner": "Arguments - Petitioner / Appellant",
    "arguments_respondent": "Arguments - Respondent",
    "reasoning": "Court's Reasoning", "decision": "Final Decision / Disposition",
}
DISCLAIMER = "AI-generated summary. Verify against the original judgment. Not legal advice."
DISCLAIMER_VERSION = "1"

STATUS_LOCATED = "Evidence located"
STATUS_REVIEW = "Needs review"
STATUS_NONE = "No evidence located"
SECTION_AVAILABLE = "Available"
SECTION_NOT_IDENTIFIED = "Not confidently identified"


@dataclass
class Config:
    # input limits
    max_upload_mb: int = 20
    max_pages: int = 150
    min_chars_per_text_page: int = 80
    min_text_page_ratio: float = 0.3       # below this -> treated as scanned/image-only
    min_paste_chars: int = 500
    # sentence units
    min_unit_chars: int = 15
    # section selection
    min_units_per_section: int = 1
    max_units_per_section: int = 60        # cap fed to summarizer (centrality-ranked; UNTESTED ranking)
    # chunking
    chunk_token_limit: int = 850           # safety margin under a 1024-token model
    # extractive fallback
    extractive_sentences: int = 4
    # evidence linking (UNCALIBRATED DEFAULTS)
    link_top_k: int = 3
    link_weak_threshold: float = 0.20
    link_strong_threshold: float = 0.45
    # models
    classifier_path: str = os.environ.get("LBRIEF_CLASSIFIER", "models/role_classifier.joblib")
    summarizer_backend: str = os.environ.get("LBRIEF_SUMMARIZER", "extractive")  # extractive | transformer
    summarizer_model_id: str = os.environ.get("LBRIEF_MODEL_ID", "facebook/bart-large-cnn")
    gen_params: dict = field(default_factory=lambda: dict(num_beams=4, do_sample=False,
                                                          max_length=160, min_length=30, no_repeat_ngram_size=3))
    hosted: bool = os.environ.get("LBRIEF_HOSTED", "0") == "1"
