"""Section summarization (FR-10). ONE backend active at runtime.

- ExtractiveSummarizer: restrained fallback the spec allows. Statements ARE verbatim source sentences.
- TransformerSummarizer: one pretrained seq2seq model, loaded lazily. Deterministic generation settings.
  `generate_fn` can be injected (tests / custom runtimes), so the logic is testable without model weights."""
from __future__ import annotations
from typing import Callable, Optional
from .config import Config
from .models import SourceUnit
from .preprocess import split_sentences
from .ranking import centrality_rank
from .chunking import chunk_units, approx_token_count


class ExtractiveSummarizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_id = "extractive-centrality-v0"
        self.params = {"sentences": cfg.extractive_sentences}

    def count_tokens(self, text: str) -> int:
        return approx_token_count(text)

    def summarize(self, section_key: str, units: list[SourceUnit]) -> list[str]:
        if not units:
            return []
        order = centrality_rank([u.normalized_text for u in units])[: self.cfg.extractive_sentences]
        return [units[i].normalized_text for i in sorted(order)]       # keep document order


class TransformerSummarizer:
    def __init__(self, cfg: Config, generate_fn: Optional[Callable[[str], str]] = None,
                 count_fn: Optional[Callable[[str], int]] = None):
        self.cfg = cfg
        self.model_id = cfg.summarizer_model_id
        self.params = dict(cfg.gen_params)
        self._generate = generate_fn
        self._count = count_fn
        if generate_fn is None:
            self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as e:
            raise RuntimeError("Transformer backend needs `torch` and `transformers` (see requirements.txt).") from e
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        except Exception as e:           # no silent model substitution (SRS error handling)
            raise RuntimeError(f"Could not load summarization model '{self.model_id}': {e}") from e
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        max_in = min(getattr(self.tok, "model_max_length", 1024), 1024)
        self.cfg.chunk_token_limit = min(self.cfg.chunk_token_limit, int(max_in * 0.85))

        def gen(text: str) -> str:
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_in).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **self.params)
            return self.tok.decode(out[0], skip_special_tokens=True)
        self._generate = gen
        self._count = lambda t: len(self.tok(t, add_special_tokens=True)["input_ids"])

    def count_tokens(self, text: str) -> int:
        return (self._count or approx_token_count)(text)

    def summarize(self, section_key: str, units: list[SourceUnit]) -> list[str]:
        if not units:
            return []
        chunks = chunk_units(units, self.cfg.chunk_token_limit, self.count_tokens)
        seen, statements = set(), []
        for ch in chunks:
            out = (self._generate(ch["text"]) or "").strip()
            if not out:
                continue                                     # empty generation -> nothing invented
            for s, e in split_sentences(out):
                st = out[s:e].strip()
                if st and st.lower() not in seen:            # exact-duplicate removal only
                    seen.add(st.lower())
                    statements.append(st)
        return statements


def make_summarizer(cfg: Config):
    if cfg.summarizer_backend == "transformer":
        return TransformerSummarizer(cfg)
    return ExtractiveSummarizer(cfg)
