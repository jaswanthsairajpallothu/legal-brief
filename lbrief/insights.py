"""Secondary NLP insights (FR-20). Only real pipeline output. Entities need spaCy + en_core_web_sm (optional);
if missing we say so instead of showing placeholders. General-purpose NER misses legal entities/statutes."""
from __future__ import annotations
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

_NLP = None
KEEP = {"PERSON": "People", "ORG": "Organisations", "GPE": "Places", "DATE": "Dates", "LAW": "Laws", "NORP": "Groups", "MONEY": "Amounts"}


def basic_stats(units, roles, page_count) -> dict:
    texts = [u.normalized_text for u in units.values()]
    return {"pages": page_count, "sentences": len(texts), "words": sum(len(t.split()) for t in texts),
            "role_counts": dict(Counter(r.predicted_label for r in roles.values()))}


def top_terms(texts: list[str], k: int = 15) -> list[str]:
    try:
        v = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=2000).fit(texts)
    except ValueError:
        return []
    scores = v.transform(texts).mean(axis=0).A1
    names = v.get_feature_names_out()
    return [names[i] for i in scores.argsort()[::-1][:k]]


def entities(texts: list[str], max_units: int = 400):
    """Returns (dict label->[(text,count)], None) or (None, reason)."""
    global _NLP
    try:
        import spacy
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        return None, "spaCy model 'en_core_web_sm' is not installed (see requirements-ml.txt)."
    found = defaultdict(Counter)
    for doc in _NLP.pipe(texts[:max_units], batch_size=64):
        for e in doc.ents:
            if e.label_ in KEEP:
                found[KEEP[e.label_]][e.text.strip()] += 1
    return {k: v.most_common(8) for k, v in found.items()}, None
