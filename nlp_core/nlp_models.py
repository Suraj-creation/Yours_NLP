"""Cached spaCy / NLTK resources so every module shares one loaded copy."""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

SPACY_MODEL = "en_core_web_sm"


def ensure_nltk() -> None:
    import nltk

    if os.environ.get("VERCEL"):
        nltk.data.path.insert(0, "/tmp/nltk_data")
    need = {"tokenizers/punkt_tab": "punkt_tab", "corpora/stopwords": "stopwords", "corpora/wordnet": "wordnet",
            "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
            "corpora/treebank": "treebank", "taggers/universal_tagset": "universal_tagset", "corpora/words": "words"}
    for path, pkg in need.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True, download_dir="/tmp/nltk_data" if os.environ.get("VERCEL") else None)


@lru_cache(maxsize=None)
def blank():
    """Tokenizer-only English pipeline (Lab 3: spacy.blank)."""
    import spacy

    return spacy.blank("en")


@lru_cache(maxsize=None)
def sentencizer():
    nlp = blank().__class__()  # fresh blank English
    nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=None)
def full(tokenizer: str = "hybrid"):
    """Full pretrained pipeline (Lab 3: spacy.load). tokenizer='hybrid' swaps in our tokenizer."""
    import spacy

    nlp = spacy.load(SPACY_MODEL)
    if tokenizer == "hybrid":
        from .tokenizers import HybridTokenizer

        nlp.tokenizer = HybridTokenizer(nlp.vocab)
    return nlp


@lru_cache(maxsize=None)
def tagger_only(tokenizer: str = "hybrid"):
    """POS + lemma only (parser and NER disabled) for fast processing of the heavy datasets."""
    import spacy

    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    if tokenizer == "hybrid":
        from .tokenizers import HybridTokenizer

        nlp.tokenizer = HybridTokenizer(nlp.vocab)
    return nlp
