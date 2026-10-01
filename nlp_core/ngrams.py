"""Module 2, Exercise 14: n-grams (1-5), phrase mining, dictionary growth.

N-grams never cross a sentence boundary. They are mined on the lemma stream
BEFORE stop-word removal (so "add the numerators" stays a real phrase) and
filtered afterwards: an n-gram that starts or ends with a stop word or
punctuation is dropped. Mining after removal invents phrases that never occur
("add top bottom"); the comparison is part of the results.
"""
from __future__ import annotations

from collections import Counter

from .stats import is_word


def ngrams(sentences: list[list[str]], n: int) -> Counter:
    c: Counter = Counter()
    for s in sentences:
        for i in range(len(s) - n + 1):
            c[tuple(s[i:i + n])] += 1
    return c


def clean_edges(counts: Counter, stop: frozenset) -> Counter:
    """Keep n-grams made of words whose first and last words are not stop words."""
    out: Counter = Counter()
    for g, v in counts.items():
        if all(is_word(t) for t in g) and g[0] not in stop and g[-1] not in stop:
            out[g] = v
    return out


def summary(counts: Counter, top: int = 10) -> dict:
    total = sum(counts.values())
    uniq = len(counts)
    once = sum(1 for v in counts.values() if v == 1)
    return {"total": total, "unique": uniq, "singletons": once,
            "singleton_share": round(once / uniq, 4) if uniq else 0.0,
            "top": [{"ngram": " ".join(g), "count": v, "freq": round(v / total, 6) if total else 0}
                    for g, v in counts.most_common(top)]}


def collocations(sentences: list[list[str]], stop: frozenset, n: int = 2, min_freq: int = 3, top: int = 25) -> list[dict]:
    """PMI and log-likelihood ranked collocations (NLTK collocation finders)."""
    from nltk.collocations import (BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures,
                                   TrigramCollocationFinder)

    seq = []
    for s in sentences:
        seq.extend(s + ["</s>"])
    if n == 2:
        f, m = BigramCollocationFinder.from_words(seq), BigramAssocMeasures()
    else:
        f, m = TrigramCollocationFinder.from_words(seq), TrigramAssocMeasures()
    f.apply_freq_filter(min_freq)
    f.apply_ngram_filter(lambda *w: any(x == "</s>" or not is_word(x) for x in w) or w[0] in stop or w[-1] in stop)
    ll = dict(f.score_ngrams(m.likelihood_ratio))
    pmi = dict(f.score_ngrams(m.pmi))
    rows = [{"ngram": " ".join(g), "count": f.ngram_fd[g], "log_likelihood": round(ll[g], 2), "pmi": round(pmi[g], 3)}
            for g in ll]
    return sorted(rows, key=lambda r: -r["log_likelihood"])[:top]


def network(bigrams: Counter, top: int = 60) -> dict:
    """Nodes and weighted edges for the bigram network view."""
    edges = bigrams.most_common(top)
    deg: Counter = Counter()
    for (a, b), v in edges:
        deg[a] += v
        deg[b] += v
    return {"nodes": [{"id": k, "weight": v} for k, v in deg.items()],
            "edges": [{"source": a, "target": b, "weight": v} for (a, b), v in edges]}
