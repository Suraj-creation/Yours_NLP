"""Module 2, Exercises 5 and 12: POS tagging, a rule/dictionary tagger and ML taggers.

Default taggers: NLTK averaged perceptron (Penn tags) and spaCy (tag_ Penn, pos_ UPOS).
Custom 1 (rules + dictionary): an override layer on top of a default tagger.
Custom 2 (ML): NLTK Unigram->Bigram->Trigram backoff chain, and a CRF, both trained on
the Penn Treebank sample plus the domain training sentences.

The rule layer comes from Exercise 12's first step (run default taggers over corpus
sentences and list domain terms they tag inconsistently), not from the gold test set.
"""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache

from .config import CACHE, GOLD, SEED

FRACTIONISH = re.compile(r"^-?\d+(?:\s\d+)?/\d+$|^\d+(?:[.,]\d+)*%?$|^\d+(?:[./]\d+)*[+\-×*÷=][\d./+\-×*÷=a-z]+$")
MATH_IMPERATIVES = {"simplify", "find", "flip", "subtract", "multiply", "divide", "add", "write", "round", "plot",
                    "shade", "convert", "cross", "solve", "compare", "estimate", "locate", "evaluate", "model",
                    "translate", "rewrite", "draw", "name", "use", "order", "identify", "list", "graph", "reduce"}
# domain dictionary: word (lower) -> Penn tag
DOMAIN_LEXICON = {
    "lcd": "NN", "lcm": "NN", "gcf": "NN", "gcd": "NN",
    "bkt": "NNP", "dkt": "NNP", "sqkt": "NNP", "cikt": "NNP", "q-mckt": "NNP", "llmkt": "NNP",
    "mathdial": "NNP", "talkmoves": "NNP", "assistments": "NNP", "assist2009": "NNP", "ednet": "NNP",
    "mae": "NNP", "eedi": "NNP", "openstax": "NNP", "gpt-4": "NNP", "gpt-4o": "NNP", "chatgpt": "NNP",
    "halves": "NNS", "thirds": "NNS", "fourths": "NNS", "fifths": "NNS", "sixths": "NNS", "eighths": "NNS",
    "tenths": "NNS", "hundredths": "NNS", "quarters": "NNS",
    "numerator": "NN", "denominator": "NN", "numerators": "NNS", "denominators": "NNS",
    "reciprocal": "NN", "idk": "UH", "u": "PRP",
}
OPERATOR_WORDS = {"x", "×", "+", "=", "÷", "*"}


# ---------------------------------------------------------------- default taggers
def nltk_tag(tokens: list[str]) -> list[str]:
    import nltk

    return [t for _, t in nltk.pos_tag(tokens)]


def spacy_tag(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Returns (Penn tag_, UPOS pos_) for pre-tokenized input."""
    from spacy.tokens import Doc

    from .nlp_models import tagger_only

    nlp = tagger_only("hybrid")
    doc = Doc(nlp.vocab, words=tokens)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    return [t.tag_ for t in doc], [t.pos_ for t in doc]


# ---------------------------------------------------------------- custom 1: rules + dictionary
def rule_tag(tokens: list[str], base: str = "nltk") -> tuple[list[str], list[str]]:
    """Apply domain rules on top of a default tagger. Returns (tags, list of rule names fired)."""
    tags = nltk_tag(tokens) if base == "nltk" else spacy_tag(tokens)[0]
    fired = [""] * len(tokens)
    for i, w in enumerate(tokens):
        lw = w.lower()
        prev = tokens[i - 1].lower() if i else ""
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        new = None
        if FRACTIONISH.match(w):
            new, why = "CD", "number/fraction pattern"
        elif lw in OPERATOR_WORDS and i and i + 1 < len(tokens) and FRACTIONISH.match(tokens[i - 1]) \
                and FRACTIONISH.match(tokens[i + 1]):
            new, why = "SYM", "operator between numbers"
        elif lw in DOMAIN_LEXICON:
            new, why = DOMAIN_LEXICON[lw], "domain dictionary"
        elif (i == 0 or prev in {".", ":", ";", "and", "then", "first"}) and lw in MATH_IMPERATIVES \
                and tags[i] not in ("VB",):
            new, why = "VB", "math instruction verb (imperative)"
        elif lw in {"top", "bottom"} and nxt in {"number", "numbers", "part", "parts"}:
            new, why = "JJ", "top/bottom as modifier"
        elif lw == "over" and FRACTIONISH.match(prev) or (lw == "over" and prev in {"one", "two", "three", "four"}):
            new, why = "IN", "'over' as fraction bar"
        if new and new != tags[i]:
            tags[i] = new
            fired[i] = why
    return tags, fired


# ---------------------------------------------------------------- training data
def load_gold(split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(GOLD / "pos_gold.jsonl", encoding="utf-8")]
    return [r for r in rows if split is None or r["split"] == split]


def treebank_sents(limit: int | None = None) -> list[list[tuple[str, str]]]:
    from nltk.corpus import treebank

    out = []
    for s in treebank.tagged_sents():
        s = [(w, t) for w, t in s if t != "-NONE-"]
        if s:
            out.append(s)
    return out[:limit] if limit else out


def training_data(domain_weight: int = 5) -> list[list[tuple[str, str]]]:
    dom = [list(zip(g["tokens"], g["penn"])) for g in load_gold("train")]
    return treebank_sents() + dom * domain_weight


# ---------------------------------------------------------------- custom 2a: NLTK backoff chain
@lru_cache(maxsize=None)
def backoff_tagger():
    import nltk

    # Not pickled: NLTK's RegexpTagger wraps its patterns in a ReDoS guard that does not
    # survive unpickling. Training the chain takes a few seconds, so it is rebuilt.
    train = training_data()
    patterns = [(r"^-?\d+(?:\s\d+)?/\d+$", "CD"), (r"^-?\d+(?:[.,]\d+)*%?$", "CD"), (r".*ing$", "VBG"),
                (r".*ed$", "VBD"), (r".*es$", "VBZ"), (r".*ly$", "RB"), (r".*s$", "NNS"), (r"^[A-Z][a-z]+$", "NNP"),
                (r".*", "NN")]
    t0 = nltk.RegexpTagger(patterns)
    t1 = nltk.UnigramTagger(train, backoff=t0)
    t2 = nltk.BigramTagger(train, backoff=t1)
    return nltk.TrigramTagger(train, backoff=t2)


def backoff_tag(tokens: list[str]) -> list[str]:
    return [t or "NN" for _, t in backoff_tagger().tag(tokens)]


# ---------------------------------------------------------------- custom 2b: CRF
def _shape(w: str) -> str:
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


def word_features(sent: list[str], i: int) -> dict:
    w = sent[i]
    f = {"bias": 1.0, "lower": w.lower(), "suf3": w[-3:].lower(), "suf2": w[-2:].lower(), "pre2": w[:2].lower(),
         "is_title": w.istitle(), "is_upper": w.isupper(), "is_digit": w.isdigit(),
         "has_digit": any(c.isdigit() for c in w), "is_fraction": bool(FRACTIONISH.match(w)),
         "has_hyphen": "-" in w, "shape": _shape(w), "len": min(len(w), 12), "in_domain_lex": w.lower() in DOMAIN_LEXICON,
         "math_imperative": w.lower() in MATH_IMPERATIVES}
    if i == 0:
        f["BOS"] = True
    else:
        p = sent[i - 1]
        f.update({"-1:lower": p.lower(), "-1:shape": _shape(p), "-1:is_fraction": bool(FRACTIONISH.match(p))})
    if i == len(sent) - 1:
        f["EOS"] = True
    else:
        n = sent[i + 1]
        f.update({"+1:lower": n.lower(), "+1:shape": _shape(n), "+1:is_fraction": bool(FRACTIONISH.match(n))})
    return f


@lru_cache(maxsize=None)
def crf_tagger():
    import sklearn_crfsuite

    path = CACHE / "crf_tagger.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    train = training_data()
    X = [[word_features([w for w, _ in s], i) for i in range(len(s))] for s in train]
    y = [[t for _, t in s] for s in train]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.05, max_iterations=120,
                               all_possible_transitions=True)
    crf.fit(X, y)
    path.write_bytes(pickle.dumps(crf))
    return crf


def crf_tag(tokens: list[str]) -> list[str]:
    return crf_tagger().predict_single([word_features(tokens, i) for i in range(len(tokens))])


TAGGERS = {
    "nltk": ("NLTK averaged perceptron", nltk_tag),
    "spacy": ("spaCy en_core_web_sm", lambda t: spacy_tag(t)[0]),
    "rules_nltk": ("Rules + dictionary over NLTK", lambda t: rule_tag(t, "nltk")[0]),
    "rules_spacy": ("Rules + dictionary over spaCy", lambda t: rule_tag(t, "spacy")[0]),
    "backoff": ("NLTK trigram backoff (ML)", backoff_tag),
    "crf": ("CRF (ML)", crf_tag),
}


def tag_all(tokens: list[str]) -> dict:
    return {k: fn(tokens) for k, (_, fn) in TAGGERS.items()}


# ---------------------------------------------------------------- Exercise 5 helpers
def verb_object_pairs(text: str) -> list[dict]:
    """Verb + object noun phrase pairs from the dependency parse ("added" -> "the numerators")."""
    from .nlp_models import full

    doc = full("hybrid")(text)
    out = []
    for t in doc:
        if t.dep_ in ("dobj", "obj", "pobj") and t.head.pos_ in ("VERB", "AUX"):
            span = doc[t.left_edge.i: t.right_edge.i + 1]
            out.append({"verb": t.head.lemma_, "object": span.text})
    return out


def noun_phrases(text: str) -> list[str]:
    from .nlp_models import full

    return [nc.text for nc in full("hybrid")(text).noun_chunks]


__all__ = ["nltk_tag", "spacy_tag", "rule_tag", "backoff_tag", "crf_tag", "TAGGERS", "tag_all", "load_gold",
           "verb_object_pairs", "noun_phrases", "SEED"]
