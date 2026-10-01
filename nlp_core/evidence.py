"""Evidence Preview: the bridge from this assessment to the research problem.

PROTOTYPE, not a learner model. It shows how the NLP pipeline turns one learner
utterance into a *Cognitive Evidence* record that keeps three things separate:

  observation     the raw text, untouched
  extracted       what the pipeline found (concepts, numbers, expressions, verb-object pairs)
  interpretation  candidate misconceptions, each with the detector that proposed it and a
                  strength; arithmetic detectors CHECK the maths (1/2+1/3=2/5 really is
                  (1+1)/(2+3)), retrieval proposes catalogue entries with document IDs

and a provenance block. Nothing here is a probability of what the learner knows;
that belongs to the persistent learner state of the later research phases.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from fractions import Fraction
from functools import lru_cache

from .config import DATA, GOLD

FRAC = r"(\d+)\s*/\s*(\d+)"
ADD = re.compile(FRAC + r"\s*\+\s*" + FRAC + r"\s*=\s*" + FRAC)
CMP = re.compile(FRAC + r"\s*(?:is\s+)?(>|<|bigger than|greater than|larger than|more than|smaller than|less than)\s*" + FRAC,
                 re.I)
DEC_CMP = re.compile(r"(\d*\.\d+)\s+(?:is\s+)?(bigger|greater|larger|more|smaller|less)\s+than\s+(\d*\.\d+)", re.I)
DEC_MUL = re.compile(r"(\d*\.\d+)\s*[x×*]\s*(\d*\.\d+)\s*=\s*(\d*\.\d+)")
EQUIV = re.compile(FRAC + r"\s*(?:=|is equal to|equals|is the same as)\s*" + FRAC, re.I)


def _f(a, b) -> Fraction | None:
    return Fraction(int(a), int(b)) if int(b) else None


def detect(text: str) -> list[dict]:
    """Rule detectors. Each returns the span it saw and an arithmetic check."""
    out = []
    for m in ADD.finditer(text):
        a, b, c, d, e, f = map(int, m.groups())
        if b and d and f:
            if (e, f) == (a + c, b + d) and Fraction(a, b) + Fraction(c, d) != Fraction(e, f):
                out.append({"hypothesis": "M01 adds numerators and adds denominators", "detector": "arithmetic: add-across",
                            "span": m.group(0), "check": f"{e}/{f} = ({a}+{c})/({b}+{d}); correct sum is {Fraction(a, b) + Fraction(c, d)}",
                            "strength": 0.9})
            elif Fraction(a, b) + Fraction(c, d) == Fraction(e, f):
                out.append({"hypothesis": "correct fraction addition", "detector": "arithmetic: sum check", "span": m.group(0),
                            "check": f"{a}/{b} + {c}/{d} = {Fraction(e, f)} (correct)", "strength": 0.9})
            else:
                out.append({"hypothesis": "SLIP or other error in fraction addition", "detector": "arithmetic: sum check",
                            "span": m.group(0), "check": f"correct sum is {Fraction(a, b) + Fraction(c, d)}, not {e}/{f}",
                            "strength": 0.5})
    for m in CMP.finditer(text):
        a, b, op, c, d = m.group(1), m.group(2), m.group(3).lower(), m.group(4), m.group(5)
        x, y = _f(a, b), _f(c, d)
        if x is None or y is None:
            continue
        says_bigger = op in (">", "bigger than", "greater than", "larger than", "more than")
        truth = x > y
        if says_bigger != truth and a == c:  # same numerators: judged by the denominator alone
            out.append({"hypothesis": "M02 larger denominator means larger fraction", "detector": "arithmetic: comparison",
                        "span": m.group(0), "check": f"{a}/{b} {'<' if x < y else '>'} {c}/{d}", "strength": 0.8})
        elif says_bigger != truth:
            out.append({"hypothesis": "incorrect fraction comparison", "detector": "arithmetic: comparison",
                        "span": m.group(0), "check": f"{a}/{b} {'<' if x < y else '>'} {c}/{d}", "strength": 0.5})
    for m in DEC_CMP.finditer(text):
        x, word, y = float(m.group(1)), m.group(2).lower(), float(m.group(3))
        says_bigger = word in ("bigger", "greater", "larger", "more")
        if says_bigger != (x > y):
            longer = len(m.group(1).split(".")[1]) > len(m.group(3).split(".")[1])
            out.append({"hypothesis": "M05 longer decimal is bigger" if longer == says_bigger else "incorrect decimal comparison",
                        "detector": "arithmetic: decimal comparison", "span": m.group(0),
                        "check": f"{x} {'<' if x < y else '>'} {y}", "strength": 0.8 if longer == says_bigger else 0.5})
    for m in DEC_MUL.finditer(text):
        x, y, z = map(float, m.groups())
        if abs(x * y - z) > 1e-9:
            whole = abs(float(m.group(1).replace(".", "") or 0) * float(m.group(2).replace(".", "") or 0) / 10 - z) < 1e-9
            out.append({"hypothesis": "M06 decimals multiplied as whole numbers" if whole else "decimal multiplication error",
                        "detector": "arithmetic: product check", "span": m.group(0), "check": f"{x} x {y} = {round(x * y, 6)}",
                        "strength": 0.8 if whole else 0.5})
    for m in EQUIV.finditer(text):
        a, b, c, d = map(int, m.groups())
        if b and d and Fraction(a, b) != Fraction(c, d) and c - a == d - b and c != a:
            out.append({"hypothesis": "M10 equivalent fractions by adding the same number", "detector": "arithmetic: equivalence",
                        "span": m.group(0), "check": f"{a}/{b} = {a / b:.3f} but {c}/{d} = {c / d:.3f}", "strength": 0.8})
    if re.search(r"add(ed|ing|s)?\s+(the\s+)?(tops?|numerators?).{0,40}(bottoms?|denominators?)", text, re.I):
        out.append({"hypothesis": "M01 adds numerators and adds denominators", "detector": "language: describes add-across",
                    "span": re.search(r"add.{0,60}(bottoms?|denominators?)", text, re.I).group(0), "check": "verbal description",
                    "strength": 0.6})
    return out


@lru_cache(maxsize=1)
def _catalogue():
    """Grounding catalogue: 55 MaE misconceptions (real) + 11 malrules M01-M11 (D25, synthetic, labelled)."""
    import csv

    from sklearn.feature_extraction.text import TfidfVectorizer

    items = []
    mae = json.loads(next(DATA.glob("D16_*")).read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    for r in mae:
        e = by_id.setdefault(r["Misconception ID"], {"id": r["Misconception ID"], "doc_id": "D16", "source": "MaE (real)",
                                                     "name": r["Misconception"].strip(), "examples": []})
        e["examples"].append(" ".join(f"{r.get('Question', '')} {r.get('Incorrect Answer', '')} {r.get('Explanation', '')}".split()))
    items += list(by_id.values())
    with open(GOLD / "synthetic_labels.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    mal: dict[str, dict] = {}
    for r in rows:
        if r["malrule_id"] in ("M00", "SLIP"):
            continue
        e = mal.setdefault(r["malrule_id"], {"id": r["malrule_id"], "doc_id": "D25", "source": "malrule (synthetic)",
                                             "name": r["malrule"], "examples": []})
        e["examples"].append(r["text"])
    items += list(mal.values())
    texts = [f"{i['name']} {' '.join(i['examples'])}" for i in items]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|\d+/\d+")
    mat = vec.fit_transform(texts)
    return items, vec, mat


def ground(text: str, k: int = 5) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity

    items, vec, mat = _catalogue()
    sims = cosine_similarity(vec.transform([text]), mat)[0]
    order = sims.argsort()[::-1][:k]
    return [{"id": items[i]["id"], "name": items[i]["name"], "doc_id": items[i]["doc_id"], "source": items[i]["source"],
             "score": round(float(sims[i]), 4), "example": items[i]["examples"][0][:160]} for i in order if sims[i] > 0]


def evidence_record(text: str) -> dict:
    from .cleaning import clean_text
    from .ner import entities
    from .pos import verb_object_pairs
    from .tokenizers import hybrid, token_rule_hits

    clean, _ = clean_text(text)
    ents = entities(clean, True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return {
        "evidence_id": "E_" + hashlib.sha1(clean.encode()).hexdigest()[:10],
        "status": "PROTOTYPE - candidate evidence, not a diagnosis",
        "observation": {"raw_text": text, "cleaned_text": clean, "channel": "language"},
        "extracted": {
            "tokens": hybrid(clean),
            "concepts": [e for e in ents if e["label"] in ("CONCEPT", "MISCONCEPTION", "KT_MODEL", "DATASET", "METRIC")],
            "other_entities": [e for e in ents if e["label"] not in ("CONCEPT", "MISCONCEPTION", "KT_MODEL", "DATASET", "METRIC")],
            "numbers_and_expressions": [h for h in token_rule_hits(clean)
                                        if h["rule"] in ("fraction", "expression", "mixed_number", "percent", "number")],
            "verb_object": verb_object_pairs(clean),
        },
        "interpretation": {"detectors": detect(clean), "retrieved_grounding": ground(clean)},
        "provenance": {"extractor": "nlp_core.evidence v1", "tokenizer": "hybrid", "ner": "spaCy + EntityRuler",
                       "catalogue": "D16 (MaE, 55 misconceptions) + D25 malrules (synthetic, labelled)", "created": now},
        "review_status": "unverified",
    }
