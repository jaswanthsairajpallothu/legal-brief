"""Small TF-IDF helpers shared by section selection, extractive fallback and evidence linking."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def centrality_rank(texts: list[str]) -> list[int]:
    """Indices ordered by mean cosine similarity to the other sentences (most representative first)."""
    if len(texts) <= 1:
        return list(range(len(texts)))
    try:
        X = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True).fit_transform(texts)
    except ValueError:
        return list(range(len(texts)))
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    return list(np.argsort(-sim.mean(axis=1)))


def similarity_matrix(queries: list[str], corpus: list[str]) -> np.ndarray:
    """Cosine similarity (queries x corpus) with a vectorizer fitted on corpus+queries."""
    if not queries or not corpus:
        return np.zeros((len(queries), len(corpus)))
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus + queries)
    except ValueError:
        return np.zeros((len(queries), len(corpus)))
    return cosine_similarity(vec.transform(queries), vec.transform(corpus))
