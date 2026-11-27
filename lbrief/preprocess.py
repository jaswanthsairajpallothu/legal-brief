"""Sentence segmentation with exact offsets into raw page text (FR-04, FR-05).
Raw text is never modified. normalized_text is a separate field. Legal abbreviations are protected."""
from __future__ import annotations
import re
from collections import Counter
from .config import Config
from .models import PageText, SourceUnit

ABBREV = {
    "mr", "mrs", "ms", "dr", "sri", "smt", "shri", "hon", "hon'ble", "no", "nos", "vs", "v", "viz", "i.e", "e.g",
    "sec", "secs", "s", "ss", "art", "arts", "cl", "sub", "para", "paras", "ors", "anr", "ltd", "pvt", "co", "rs",
    "inr", "etc", "cr", "crl", "c", "p", "cri", "pc", "ipc", "crpc", "cpc", "w.p", "s.l.p", "r", "dt", "ex", "st",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx", "vol", "ch",
    "sr", "jr", "j", "jj", "ld", "lt", "col", "capt", "prof", "kumari", "m/s", "u/s", "a.i.r", "air",
    "scc", "scr",
}
_CAND = re.compile(r"[.?!][\"'\)\]”’]*(?=\s+)")
_STARTS = re.compile(r"[\"'“‘\(\[]*[A-Z0-9]")


def _last_token(text: str, dot_pos: int) -> str:
    """Token immediately before the '.' at dot_pos (letters/dots/apostrophes/slashes)."""
    m = re.search(r"([\w./'’]+)$", text[max(0, dot_pos - 25):dot_pos])
    return (m.group(1) if m else "").lower().strip(".")


_PARTY_LINE = re.compile(r"^.{3,90}\s(v\.?|vs\.?|versus)\s.{3,90}$", re.I)


def _heading_breaks(text: str) -> list[int]:
    """Offsets just after heading-like lines (mostly-uppercase or 'A v. B' lines with no terminal
    punctuation, e.g. the cause title). Without this they fuse with the first real sentence."""
    out, pos = [], 0
    for ln in text.splitlines(keepends=True):
        body = ln.strip()
        pos += len(ln)
        if not body or len(body) > 90 or body[-1] in ".?!;:,":
            continue
        letters = [c for c in body if c.isalpha()]
        upper = letters and sum(c.isupper() for c in letters) / len(letters) >= 0.7
        if upper or _PARTY_LINE.match(body):
            out.append(pos)
    return out


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans into `text`. Whitespace-only spans dropped."""
    cuts = {}
    for m in _CAND.finditer(text):
        end = m.end()
        nxt = text[end:].lstrip()
        if not nxt or not _STARTS.match(nxt):
            continue
        if text[m.start()] == ".":
            tok = _last_token(text, m.start())
            if tok in ABBREV or re.fullmatch(r"[a-z]", tok):
                continue
            if re.fullmatch(r"\d{1,2}", tok) and re.search(r"(^|\n)[ \t]*\d{1,2}$", text[:m.start()]):
                continue                       # list marker such as "\n12. The appellant..."
        cuts[end] = True
    for b in _heading_breaks(text):
        cuts[b] = True
    spans, start = [], 0
    for end in sorted(cuts):
        spans.append((start, end))
        start = end
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        seg = text[s:e]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        if seg.strip():
            out.append((s + lead, e - trail))
    return out


def normalize(s: str) -> str:
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)      # re-join hyphenated line breaks
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def find_noise_lines(pages: list[PageText]) -> set[str]:
    """Lines (digits masked) repeated on >=50% of pages (min 3 pages) = running headers/footers. Conservative."""
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for p in pages:
        seen = set()
        for ln in p.raw_text.splitlines():
            n = re.sub(r"\d+", "#", normalize(ln)).lower()
            if n and len(n) <= 90 and n not in seen:
                seen.add(n)
                counts[n] += 1
    need = max(3, int(0.5 * len(pages)))
    return {k for k, v in counts.items() if v >= need}


def _mask_noise(raw: str, noise: set[str]) -> str:
    """Blank noise lines with spaces so offsets stay aligned with raw text."""
    out = []
    for ln in raw.splitlines(keepends=True):
        body = ln.rstrip("\r\n")
        n = re.sub(r"\d+", "#", normalize(body)).lower()
        is_pageno = bool(re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", body.strip().lower()))
        if n and (n in noise or is_pageno):
            out.append(" " * len(body) + ln[len(body):])
        else:
            out.append(ln)
    return "".join(out)


def build_units(doc_id: str, pages: list[PageText], cfg: Config) -> list[SourceUnit]:
    noise = find_noise_lines(pages)
    units: list[SourceUnit] = []
    idx = 0
    for p in pages:
        if p.extraction_status != "ok":
            continue
        work = _mask_noise(p.raw_text, noise)
        p.normalized_text = normalize(work)
        for s, e in split_sentences(work):
            norm = normalize(work[s:e])
            if len(norm) < cfg.min_unit_chars:
                continue
            units.append(SourceUnit(f"{doc_id}-p{p.page_number}-s{idx}", doc_id, p.page_number, idx,
                                    p.raw_text[s:e], norm, s, e))
            idx += 1
    return units
