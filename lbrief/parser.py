"""Input validation + page-preserving extraction (FR-01..FR-03). Everything stays in memory: no uploaded
bytes are written to disk."""
from __future__ import annotations
import uuid
import pymupdf
from .config import Config
from .models import Document, PageText


class InputError(Exception):
    """User-facing, plain-language input problem."""


def parse_pdf(data: bytes, filename: str, cfg: Config) -> tuple[Document, list[PageText]]:
    if not data:
        raise InputError("The file is empty.")
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise InputError(f"File is larger than the {cfg.max_upload_mb} MB limit.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise InputError("This does not look like a PDF file.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise InputError("The PDF could not be opened. It may be corrupted.")
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise InputError("The PDF is password-protected. Please upload an unlocked copy.")
        if doc.page_count == 0:
            raise InputError("The PDF has no pages.")
        if doc.page_count > cfg.max_pages:
            raise InputError(f"The PDF has {doc.page_count} pages; the limit is {cfg.max_pages}.")
        doc_id = uuid.uuid4().hex[:12]
        pages, empty = [], []
        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            ok = len(text.strip()) >= cfg.min_chars_per_text_page
            if not ok:
                empty.append(i)
            pages.append(PageText(doc_id, i, text, extraction_status="ok" if ok else "no_text"))
        text_pages = len(pages) - len(empty)
        if text_pages == 0 or text_pages / len(pages) < cfg.min_text_page_ratio:
            raise InputError("No usable text layer was found. This looks like a scanned or image-only PDF; "
                             "OCR is not supported in version 1.0.")
        d = Document(doc_id, filename, len(pages), "pdf", "ok" if not empty else "partial", empty)
        return d, pages
    finally:
        doc.close()


def parse_text(text: str, cfg: Config, filename: str = "pasted text") -> tuple[Document, list[PageText]]:
    text = text or ""
    if len(text.strip()) < cfg.min_paste_chars:
        raise InputError(f"Please paste at least {cfg.min_paste_chars} characters of judgment text.")
    doc_id = uuid.uuid4().hex[:12]
    return (Document(doc_id, filename, 1, "pasted"), [PageText(doc_id, 1, text)])
