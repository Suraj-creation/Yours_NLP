"""Module 2, Exercise 13: named entity recognition.

Baseline: spaCy en_core_web_sm on the hybrid tokens. Custom: an EntityRuler placed
BEFORE the statistical NER, so domain matches (BKT, MathDial, "common denominator")
are fixed first and the model cannot relabel them (spaCy alone calls BKT an ORG).
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache

import yaml

from .config import CONFIG, GOLD

STANDARD = ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT"]
DOMAIN = ["CONCEPT", "MISCONCEPTION", "KT_MODEL", "DATASET", "METRIC"]
TYPES = STANDARD + DOMAIN


def lexicon() -> list[dict]:
    with open(CONFIG / "domain_lexicon.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_name(term: str) -> bool:
    return bool(re.search(r"[A-Z].*[A-Z]|\d", term))  # BKT, MathDial, ASSIST2009 -> exact case


def ruler_patterns() -> list[dict]:
    from .tokenizers import hybrid

    pats = []
    for row in lexicon():
        toks = hybrid(row["term"])
        if _is_name(row["term"]):
            pats.append({"label": row["type"], "pattern": [{"ORTH": t} for t in toks], "id": row["term"]})
            continue
        base = [{"LOWER": t.lower()} for t in toks]
        pats.append({"label": row["type"], "pattern": base, "id": row["term"]})
        last = toks[-1].lower()
        for plural in {last + "s", last + "es", re.sub(r"y$", "ies", last)} - {last}:
            pats.append({"label": row["type"], "pattern": base[:-1] + [{"LOWER": plural}], "id": row["term"]})
    # Tokenization -> NER dependency: the hybrid tokenizer keeps "$3,650" and "₹200" as ONE
    # token, but en_core_web_sm learned MONEY from split tokens ("$", "3,650"), so its MONEY
    # recall drops to zero. Two shape patterns restore it; ISO dates likewise.
    pats += [
        {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"^[₹$€£]\d[\d,]*(\.\d+)?$"}}], "id": "currency-token"},
        {"label": "MONEY", "pattern": [{"LOWER": {"IN": ["rs", "rs.", "inr"]}}, {"TEXT": {"REGEX": r"^\d[\d,]*(\.\d+)?$"}}],
         "id": "rupees"},
        {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"^\d{4}-\d{2}-\d{2}(T[\d:]+)?$"}}], "id": "iso-date"},
    ]
    extra = yaml.safe_load((CONFIG / "entity_patterns.yaml").read_text(encoding="utf-8"))
    for label, names in extra.items():
        for n in names:
            pats.append({"label": label, "pattern": [{"ORTH": t} for t in hybrid(n)], "id": n})
    return pats


@lru_cache(maxsize=None)
def pipeline(with_ruler: bool = True):
    import spacy

    from .tokenizers import HybridTokenizer

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = HybridTokenizer(nlp.vocab)
    if with_ruler:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        ruler.add_patterns(ruler_patterns())
    return nlp


def entities(text: str, with_ruler: bool = True, keep: list[str] | None = TYPES) -> list[dict]:
    doc = pipeline(with_ruler)(text)
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char,
             "source": "ruler" if with_ruler and e.ent_id_ else "model"}
            for e in doc.ents if keep is None or e.label_ in keep]


def load_gold() -> list[dict]:
    return [json.loads(line) for line in open(GOLD / "ner_gold.jsonl", encoding="utf-8")]


def evaluate(with_ruler: bool) -> tuple[dict, list[dict]]:
    """Scores on the gold set + a per-entity table (Required Results Format E)."""
    from .evaluate import ner_scores

    gold, pred, rows = [], [], []
    for g in load_gold():
        ents = entities(g["text"], with_ruler)
        gs = [tuple(e) for e in g["entities"]]
        ps = [(e["start"], e["end"], e["label"]) for e in ents]
        gold.append(gs)
        pred.append(ps)
        for (a, b, lab) in gs:
            match = next((p for p in ps if (p[0], p[1]) == (a, b)), None)
            overlap = next((p for p in ps if p[0] < b and a < p[1]), None)
            status = "correct" if match and match[2] == lab else ("incorrect" if (match or overlap) else "missed")
            got = match or overlap
            rows.append({"sentence": g["id"], "entity": g["text"][a:b], "gold_type": lab,
                         "predicted_type": got[2] if got else "", "predicted_text": g["text"][got[0]:got[1]] if got else "",
                         "status": status, "domain_type": lab in DOMAIN})
        for p in ps:
            if not any(p[0] < b and a < p[1] for a, b, _ in gs):
                rows.append({"sentence": g["id"], "entity": g["text"][p[0]:p[1]], "gold_type": "",
                             "predicted_type": p[2], "predicted_text": g["text"][p[0]:p[1]], "status": "spurious",
                             "domain_type": p[2] in DOMAIN})
    return ner_scores(gold, pred), rows


def displacy_html(text: str, with_ruler: bool = True) -> str:
    from spacy import displacy

    doc = pipeline(with_ruler)(text)
    colors = {"CONCEPT": "#c7e3ff", "MISCONCEPTION": "#ffd0d0", "KT_MODEL": "#e0d4ff", "DATASET": "#d4f5dc",
              "METRIC": "#fff1bf"}
    return displacy.render(doc, style="ent", options={"colors": colors}, jupyter=False)
