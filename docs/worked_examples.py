"""Small worked examples for syllabus topics NOT implemented as product features (SRS section 20):
edit distance (Unit II) and bigram probabilities with Laplace smoothing (Unit II). Run: python docs/worked_examples.py"""


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute = 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bigram_prob_laplace(w1: str, w2: str, sentences: list[str]) -> float:
    """P(w2 | w1) = (count(w1 w2) + 1) / (count(w1) + V)  (add-one smoothing)."""
    toks = [s.lower().split() for s in sentences]
    vocab = {w for t in toks for w in t}
    c12 = sum(1 for t in toks for x, y in zip(t, t[1:]) if (x, y) == (w1, w2))
    c1 = sum(1 for t in toks for x in t[:-1] if x == w1)
    return (c12 + 1) / (c1 + len(vocab))


if __name__ == "__main__":
    print("edit_distance('appelant','appellant') =", edit_distance("appelant", "appellant"))
    corpus = ["the court held", "the court said", "the appeal failed"]
    print("P(court|the) unsmoothed = 2/3 =", round(2 / 3, 3), "| Laplace =", round(bigram_prob_laplace("the", "court", corpus), 3))
    print("P(appeal|the) Laplace =", round(bigram_prob_laplace("the", "appeal", corpus), 3), "(unseen-ish pair gets non-zero mass)")
