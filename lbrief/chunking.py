"""Token-aware chunking (FR-09). Chunks keep source IDs. Nothing is silently dropped: a sentence longer than
the limit is split by words, never truncated."""
from __future__ import annotations
from typing import Callable
from .models import SourceUnit


def approx_token_count(text: str) -> int:
    return int(len(text.split()) * 1.4) + 2        # fallback only; use the real tokenizer when available


def chunk_units(units: list[SourceUnit], limit: int, count: Callable[[str], int] = approx_token_count) -> list[dict]:
    chunks: list[dict] = []
    cur_text, cur_ids, cur_n = [], [], 0

    def flush():
        nonlocal cur_text, cur_ids, cur_n
        if cur_text:
            chunks.append({"text": " ".join(cur_text), "source_ids": cur_ids, "tokens": cur_n})
        cur_text, cur_ids, cur_n = [], [], 0

    for u in units:
        t = u.normalized_text
        n = count(t)
        if n > limit:                                  # oversize sentence: split by words, keep source id
            flush()
            words, piece = t.split(), []
            for w in words:
                piece.append(w)
                if count(" ".join(piece)) >= limit:
                    chunks.append({"text": " ".join(piece), "source_ids": [u.source_id], "tokens": count(" ".join(piece))})
                    piece = []
            if piece:
                chunks.append({"text": " ".join(piece), "source_ids": [u.source_id], "tokens": count(" ".join(piece))})
            continue
        if cur_n + n > limit:
            flush()
        cur_text.append(t)
        if u.source_id not in cur_ids:
            cur_ids.append(u.source_id)
        cur_n += n
    flush()
    return chunks
