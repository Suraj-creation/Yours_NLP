"""Module 4: positional inverted index.

term -> {doc_id: [positions]}. Positions are the ORIGINAL token positions, so a
stop word removed from the index leaves a gap instead of making two words look
adjacent. That is what keeps phrase search honest.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


class PositionalIndex:
    def __init__(self):
        self.postings: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.doc_len: dict[str, int] = {}
        self.doc_ids: list[str] = []

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, docs_terms: dict[str, list[tuple[str, int]]]) -> "PositionalIndex":
        idx = cls()
        for doc_id, pairs in docs_terms.items():
            idx.doc_ids.append(doc_id)
            idx.doc_len[doc_id] = len(pairs)
            for term, pos in pairs:
                idx.postings[term].setdefault(doc_id, []).append(pos)
        idx.postings = dict(idx.postings)
        return idx

    # ------------------------------------------------------------ stats
    @property
    def N(self) -> int:
        return len(self.doc_ids)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def cf(self, term: str) -> int:
        return sum(len(p) for p in self.postings.get(term, {}).values())

    def idf(self, term: str) -> float:
        d = self.df(term)
        return math.log10(self.N / d) if d else 0.0

    def tfidf(self, term: str, doc_id: str) -> float:
        tf = len(self.postings.get(term, {}).get(doc_id, []))
        return (1 + math.log10(tf)) * self.idf(term) if tf else 0.0

    def docs(self, term: str) -> set[str]:
        return set(self.postings.get(term, {}))

    @property
    def vocabulary(self) -> int:
        return len(self.postings)

    # ------------------------------------------------------------ phrase
    def phrase_docs(self, terms: list[tuple[str, int]]) -> dict[str, int]:
        """Docs containing the terms at the same relative offsets. Returns doc -> match count."""
        if not terms:
            return {}
        first, p0 = terms[0]
        cand = self.docs(first)
        for t, _ in terms[1:]:
            cand &= self.docs(t)
        out = {}
        for d in cand:
            starts = set(self.postings[first][d])
            for t, p in terms[1:]:
                off = p - p0
                starts &= {x - off for x in self.postings[t][d]}
                if not starts:
                    break
            if starts:
                out[d] = len(starts)
        return out

    # ------------------------------------------------------------ io
    def to_json(self, path: Path | None = None, with_positions: bool = True) -> dict:
        data = {
            "documents": self.doc_ids,
            "doc_length": self.doc_len,
            "vocabulary_size": self.vocabulary,
            "terms": {t: {"df": len(p), "cf": sum(len(v) for v in p.values()),
                          "postings": ({d: v for d, v in sorted(p.items())} if with_positions else sorted(p))}
                      for t, p in sorted(self.postings.items())},
        }
        if path:
            Path(path).write_text(json.dumps(data, ensure_ascii=False))
        return data

    @classmethod
    def from_json(cls, data: dict) -> "PositionalIndex":
        idx = cls()
        idx.doc_ids = list(data["documents"])
        idx.doc_len = dict(data["doc_length"])
        idx.postings = {t: {d: list(v) for d, v in e["postings"].items()} for t, e in data["terms"].items()}
        return idx
