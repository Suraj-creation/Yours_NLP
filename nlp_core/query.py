"""Modules 4-5: query processing and retrieval.

Operators are UPPERCASE only (AND, OR, NOT). Lower-case "not" and "and" are ordinary
words in this domain ("not a common denominator"), so they must stay searchable.

Grammar: expr := unit | expr AND expr | expr OR expr | NOT expr | ( expr )
         adjacent units without an operator are joined by AND (implicit AND);
         a unit is a quoted phrase, or a run of bare words matched as a phrase.
Precedence NOT > AND > OR. NOT is the complement against all document IDs.
Every unit is analysed by the SAME pipeline as the index, then matched by
positional intersection (a single-term unit is a plain postings lookup).
Boolean results are sets; they are ranked by TF-IDF so P@K and R@K have an order.
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field

from .index import PositionalIndex

TOKEN = re.compile(r'"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()"]+')
PREC = {"NOT": 3, "AND": 2, "OR": 1}


@dataclass
class Unit:
    text: str
    terms: list[tuple[str, int]] = field(default_factory=list)
    negated: bool = False


def lex(query: str) -> list[str]:
    """Split into operators, brackets, quoted phrases and word runs."""
    raw = TOKEN.findall(query)
    out, buf = [], []
    for t in raw:
        if t in ("AND", "OR", "NOT", "(", ")") or t.startswith('"'):
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(t)
        else:
            buf.append(t)
    if buf:
        out.append(" ".join(buf))
    # implicit AND between adjacent operands / closing-opening brackets
    fixed = []
    for t in out:
        if fixed and _is_operand_end(fixed[-1]) and (_is_operand_start(t)):
            fixed.append("AND")
        fixed.append(t)
    return fixed


def _is_operand_end(t):
    return t not in ("AND", "OR", "NOT", "(")


def _is_operand_start(t):
    return t not in ("AND", "OR", ")")


def to_postfix(tokens: list[str]) -> list[str]:
    out, ops = [], []
    for t in tokens:
        if t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if not ops:
                raise ValueError("unbalanced ')'")
            ops.pop()
        elif t in PREC:
            while ops and ops[-1] != "(" and (PREC[ops[-1]] > PREC[t] or (PREC[ops[-1]] == PREC[t] and t != "NOT")):
                out.append(ops.pop())
            ops.append(t)
        else:
            out.append(t)
    while ops:
        if ops[-1] == "(":
            raise ValueError("unbalanced '('")
        out.append(ops.pop())
    return out


def classify(query: str) -> str:
    """Query type label as used in Results Table I."""
    ops = set(re.findall(r"\b(AND|OR|NOT)\b", query))
    if "(" in query or ("OR" in ops and len(ops) > 1):
        return "Advanced Boolean"
    if "NOT" in ops:
        return "Boolean NOT"
    if "OR" in ops:
        return "Boolean OR"
    if "AND" in ops:
        return "Boolean AND"
    words = query.replace('"', " ").split()
    if len(words) == 1:
        return "Keyword (unigram)"
    return {2: "Phrase (bigram)", 3: "Phrase (trigram)"}.get(len(words), f"Phrase ({len(words)}-gram)")


class QueryEngine:
    def __init__(self, index: PositionalIndex, analyzer):
        """analyzer(text) -> [(term, position)] using the index pipeline's normalisation."""
        self.index = index
        self.analyzer = analyzer

    def _unit_docs(self, text: str) -> tuple[set[str], list[tuple[str, int]]]:
        terms = self.analyzer(text.strip('"'))
        if not terms:
            return set(), []
        if len(terms) == 1:
            return self.index.docs(terms[0][0]), terms
        return set(self.index.phrase_docs(terms)), terms

    def evaluate(self, query: str, mode: str = "auto") -> tuple[set[str], list[Unit]]:
        if mode == "keyword":  # every word must appear, anywhere (implicit AND of single terms)
            units = [Unit(w, self.analyzer(w)) for w in query.replace('"', " ").split()]
            docs = None
            for u in units:
                d = set(self.index.docs(u.terms[0][0])) if len(u.terms) == 1 else set(self.index.phrase_docs(u.terms))
                if not u.terms:
                    continue
                docs = d if docs is None else docs & d
            return docs or set(), units
        if mode == "phrase":
            d, terms = self._unit_docs(query.replace('"', ""))
            return d, [Unit(query, terms)]
        postfix = to_postfix(lex(query))
        stack: list[tuple[set[str], list[int]]] = []  # (docs, indices of the units inside)
        units: list[Unit] = []
        universe = set(self.index.doc_ids)
        for t in postfix:
            if t == "NOT":
                if not stack:
                    raise ValueError("NOT needs an operand")
                d, us = stack.pop()
                for i in us:
                    units[i].negated = not units[i].negated
                stack.append((universe - d, us))
            elif t in ("AND", "OR"):
                if len(stack) < 2:
                    raise ValueError(f"{t} needs two operands")
                (b, ub), (a, ua) = stack.pop(), stack.pop()
                stack.append((a & b if t == "AND" else a | b, ua + ub))
            else:
                d, terms = self._unit_docs(t)
                units.append(Unit(t, terms))
                stack.append((d, [len(units) - 1]))
        if len(stack) != 1:
            raise ValueError("malformed query")
        return stack[0][0], units

    def rank(self, docs: set[str], units: list[Unit]) -> list[tuple[str, float]]:
        pos_terms = [t for u in units if not u.negated for t, _ in u.terms]
        scored = [(d, round(sum(self.index.tfidf(t, d) for t in pos_terms), 4)) for d in docs]
        return sorted(scored, key=lambda x: (-x[1], x[0]))

    def search(self, query: str, mode: str = "auto", repeats: int = 20) -> dict:
        times = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            docs, units = self.evaluate(query, mode)
            ranked = self.rank(docs, units)
            times.append(time.perf_counter() - t0)
        return {"query": query, "type": classify(query) if mode == "auto" else mode, "results": ranked,
                "doc_ids": [d for d, _ in ranked], "count": len(ranked),
                "time_ms": round(statistics.median(times) * 1000, 4),
                "analysed": [{"unit": u.text, "terms": [t for t, _ in u.terms], "negated": u.negated} for u in units]}
