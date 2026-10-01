"""Module 2, Exercises 7-10: stop words, stemming, lemmatization.

Stop-word decision (Ex 7): NLTK's English list deletes words this domain needs:
negation ("not", "no", "nor") and comparison/quantity words ("more", "than",
"over", "same", "each", "few", "most", "all", "both", "before", "after",
"above", "below", "under", "up", "down", "off", "out", "only", "very").
"3/4 is more than 2/3" and "not a common denominator" lose their meaning without
them. The custom list = NLTK minus those words, plus corpus noise found by
document frequency (speaker labels, figure/paper boilerplate).
"""
from __future__ import annotations

import re
from functools import lru_cache

from .config import CONFIG

KEEP = {"not", "no", "nor", "more", "most", "than", "over", "under", "same", "each", "few", "all", "both",
        "before", "after", "above", "below", "up", "down", "off", "out", "only", "very", "again", "further",
        "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "can't", "couldn't", "won't", "shouldn't"}
# Speaker labels such as "Teacher:" are frequent but are real domain words (queries use
# "teacher" and "student"), so they stay. Fillers and extraction labels go.
CORPUS_NOISE = {"et", "al", "fig", "pp", "arxiv", "doi", "eq", "authors", "venue",
                "um", "uh", "umm", "hmm", "mmm", "okay", "ok", "yeah"}


@lru_cache(maxsize=None)
def stopwords(name: str) -> frozenset:
    if name in ("none", None):
        return frozenset()
    if name == "nltk":
        from nltk.corpus import stopwords as sw

        return frozenset(sw.words("english"))
    if name == "spacy":
        from spacy.lang.en.stop_words import STOP_WORDS

        return frozenset(STOP_WORDS)
    if name == "custom":
        p = CONFIG / "stopwords_custom.txt"
        if p.exists():
            return frozenset(w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#"))
        return frozenset(set(stopwords("nltk")) - KEEP | CORPUS_NOISE)
    raise ValueError(name)


def write_custom_stopwords() -> list[str]:
    words = sorted(set(stopwords("nltk")) - KEEP | CORPUS_NOISE)
    header = ["# Custom stop-word list (Exercise 7).",
              "# = NLTK English list MINUS negation/comparison words this domain needs:",
              "#   " + ", ".join(sorted(KEEP & set(stopwords("nltk")))),
              "# PLUS corpus noise found by document frequency and inspection:",
              "#   " + ", ".join(sorted(CORPUS_NOISE))]
    (CONFIG / "stopwords_custom.txt").write_text("\n".join(header + words) + "\n")
    stopwords.cache_clear()
    return words


# ---------------------------------------------------------------- stemmers
@lru_cache(maxsize=None)
def stemmer(name: str):
    from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer

    return {"porter": PorterStemmer(), "snowball": SnowballStemmer("english"), "lancaster": LancasterStemmer(),
            # one hand-written suffix rule set (Exercise 8 "custom"): strip common inflections only
            "regexp": RegexpStemmer(r"(?:ing|ed|es|s)$", min=5)}[name]


def stem(word: str, name: str = "porter") -> str:
    if not re.search(r"[A-Za-z]", word) or re.search(r"\d", word):
        return word  # never stem numbers, fractions or expressions
    return stemmer(name).stem(word)


STEMMERS = ["porter", "snowball", "lancaster", "regexp"]

# ---------------------------------------------------------------- lemmatizers
_PENN2WN = {"J": "a", "V": "v", "N": "n", "R": "r"}


def penn_to_wordnet(tag: str) -> str:
    return _PENN2WN.get(tag[:1], "n")


@lru_cache(maxsize=None)
def _wnl():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def lemma_wordnet(word: str, penn_tag: str | None = None) -> str:
    if re.search(r"\d", word):
        return word
    w = word.lower()
    return _wnl().lemmatize(w, penn_to_wordnet(penn_tag)) if penn_tag else _wnl().lemmatize(w)


def lemmatize_tokens(tokens: list[str], method: str = "spacy") -> list[str]:
    """method: wordnet_nopos | wordnet_pos | spacy"""
    if method == "wordnet_nopos":
        return [lemma_wordnet(t) for t in tokens]
    if method == "wordnet_pos":
        import nltk

        return [lemma_wordnet(w, t) for w, t in nltk.pos_tag(tokens)]
    if method == "spacy":
        from spacy.tokens import Doc

        from .nlp_models import tagger_only

        nlp = tagger_only("hybrid")
        doc = Doc(nlp.vocab, words=tokens)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        return [t.lemma_.lower() if not re.search(r"\d", t.text) else t.text for t in doc]
    raise ValueError(method)


LEMMATIZERS = ["wordnet_nopos", "wordnet_pos", "spacy"]


def normalize_term(t: str) -> str:
    """Final index-term normalisation shared by index and query: lowercase and strip possessive 's."""
    t = t.lower()
    return t[:-2] if t.endswith("'s") and len(t) > 3 else t
