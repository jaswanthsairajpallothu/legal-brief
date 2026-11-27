"""Best-effort metadata (FR-19). Never guesses: unmatched fields stay None -> shown as 'Not detected'."""
from __future__ import annotations
import re
from .models import PageText

_CASE_NO = re.compile(r"\b((?:Civil|Criminal)\s+Appeal|Writ\s+Petition(?:\s*\((?:Civil|Criminal|C|Crl)\.?\))?|"
                      r"Special\s+Leave\s+Petition(?:\s*\((?:Civil|Criminal|C|Crl)\.?\))?|SLP(?:\s*\((?:C|Crl)\.?\))?|"
                      r"Crl\.?\s*A\.?|C\.?A\.?|W\.?P\.?(?:\s*\(C\))?)\s*(?:No\.?|Nos\.?)?\s*(\d[\d,/ -]*\d|\d)\s*(?:of|/)\s*((?:19|20)\d{2})", re.I)
_TITLE = re.compile(r"^(.{3,90}?)\s+(?:v\.?|vs\.?|versus)\s+(.{3,90})$", re.I)
_COURT = re.compile(r"(SUPREME COURT OF INDIA|HIGH COURT OF [A-Z ,&]+|HIGH COURT AT [A-Z ,&]+|[A-Z ]+ HIGH COURT)", re.I)


def extract_metadata(pages: list[PageText]) -> dict:
    md = {"title": None, "court": None, "case_number": None, "year": None}
    if not pages:
        return md
    head = pages[0].raw_text
    lines = [l.strip() for l in head.splitlines() if l.strip()][:60]
    m = _COURT.search(head[:1500])
    if m:
        md["court"] = re.sub(r"\s+", " ", m.group(1)).strip().title().replace("Of India", "of India")
    m = _CASE_NO.search(head[:2500])
    if m:
        md["case_number"] = re.sub(r"\s+", " ", m.group(0)).strip()
        md["year"] = m.group(3)
    for i, l in enumerate(lines):
        t = _TITLE.match(l)
        if t:
            md["title"] = l
            break
        if i + 2 < len(lines) and re.fullmatch(r"(v\.?|vs\.?|versus)", lines[i + 1], re.I):
            md["title"] = f"{l} v. {lines[i + 2]}"
            break
    return md
