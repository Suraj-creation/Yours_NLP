"""Module 1 statistics, plus Zipf and Heaps curves for the website and report."""
from __future__ import annotations

import math
import re
from collections import Counter

WORD = re.compile(r"[^\W_]", re.U)


def is_word(tok: str) -> bool:
    return bool(WORD.search(tok))


def doc_stats(doc_id: str, text: str, tokens: list[str], sentences: list[str]) -> dict:
    words = [t for t in tokens if is_word(t)]
    c = Counter(w.lower() for w in words)
    return {"doc_id": doc_id, "sentences": len(sentences), "tokens": len(tokens), "word_tokens": len(words),
            "characters": len(text), "vocabulary": len(c),
            "type_token_ratio": round(len(c) / max(1, len(words)), 4),
            "hapax": sum(1 for v in c.values() if v == 1)}


def corpus_summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {"documents": n, "sentences": sum(r["sentences"] for r in rows), "tokens": sum(r["tokens"] for r in rows),
            "word_tokens": sum(r["word_tokens"] for r in rows), "characters": sum(r["characters"] for r in rows),
            "avg_doc_length_tokens": round(sum(r["tokens"] for r in rows) / max(1, n), 1),
            "avg_doc_length_characters": round(sum(r["characters"] for r in rows) / max(1, n), 1)}


def zipf(counts: Counter, max_points: int = 400) -> dict:
    """Rank-frequency points (log-spaced sample) and the least-squares slope in log-log space."""
    freqs = sorted(counts.values(), reverse=True)
    if not freqs:
        return {"points": [], "slope": None}
    xs = [math.log10(i + 1) for i in range(len(freqs))]
    ys = [math.log10(f) for f in freqs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-12, sum((x - mx) ** 2 for x in xs))
    idx = sorted({int(round(10 ** (k * math.log10(n) / max_points))) - 1 for k in range(max_points + 1)})
    ranked = counts.most_common()
    pts = [{"rank": i + 1, "freq": freqs[i], "term": ranked[i][0]} for i in idx if 0 <= i < n]
    return {"points": pts, "slope": round(slope, 3), "types": n}


def heaps(tokens: list[str], steps: int = 60) -> dict:
    """Vocabulary size V as tokens N are read; fit V = K * N^beta."""
    seen, pts = set(), []
    n = len(tokens)
    marks = {max(1, int(n * (k / steps))) for k in range(1, steps + 1)}
    for i, t in enumerate(tokens, 1):
        seen.add(t)
        if i in marks:
            pts.append({"tokens": i, "vocabulary": len(seen)})
    if len(pts) < 2:
        return {"points": pts, "K": None, "beta": None}
    xs = [math.log(p["tokens"]) for p in pts]
    ys = [math.log(p["vocabulary"]) for p in pts]
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-12, sum((x - mx) ** 2 for x in xs))
    K = math.exp(my - beta * mx)
    return {"points": pts, "K": round(K, 2), "beta": round(beta, 3)}
