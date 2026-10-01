"""Module 3: configurable pipelines, run end to end and compared.

A pipeline is a small config (tokenizer, stop-word list, normaliser, order). Running it
over the corpus produces the index terms with their ORIGINAL positions, a log of
token and vocabulary counts after every stage, and an analyzer that applies exactly
the same steps to queries.

Intermediate results are cached per (tokenizer) and per (tokenizer, normaliser), so
the website's pipeline designer can re-run a new order in about a second.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from .cleaning import clean_text, split_sentences
from .config import CACHE, load_yaml
from .index import PositionalIndex
from .loaders import Document, load_corpus
from .preprocess import lemma_wordnet, normalize_term, stem, stopwords
from .query import QueryEngine
from .stats import is_word

STAGES = ["tokenized", "lowercased", "stopwords_removed", "normalized", "index_terms"]
SENT_GAP = 1  # positions skip between sentences so phrases never cross a sentence boundary


# ---------------------------------------------------------------- tokenization of one sentence
def _tokens(sent: str, tokenizer: str) -> list[str]:
    if tokenizer.startswith("tiktoken"):
        from .bpe import tiktoken_pieces

        name = {"tiktoken_cl100k": "cl100k_base", "tiktoken_gpt2": "gpt2", "tiktoken_o200k": "o200k_base"}[tokenizer]
        return [p.strip() for p in tiktoken_pieces(sent, name) if p.strip()]
    from .tokenizers import tokenize

    return tokenize(sent, tokenizer)


def _normalize_sentences(sents: list[list[str]], normalizer: str) -> list[list[str]]:
    """Normalise every token of every sentence (POS-aware ones see the full cased sentence)."""
    if normalizer in ("porter", "snowball", "lancaster", "regexp"):
        return [[stem(t.lower(), normalizer) for t in s] for s in sents]
    if normalizer == "wordnet_nopos":
        return [[lemma_wordnet(t) for t in s] for s in sents]
    if normalizer == "wordnet_cf":
        return [[lemma_context_free(t) for t in s] for s in sents]
    if normalizer == "wordnet_pos":
        import nltk

        tagged = nltk.pos_tag_sents(sents)
        return [[lemma_wordnet(w, t) for w, t in s] for s in tagged]
    if normalizer == "spacy":
        from spacy.tokens import Doc

        from .nlp_models import tagger_only

        nlp = tagger_only("hybrid")
        docs = (Doc(nlp.vocab, words=s) if s else Doc(nlp.vocab, words=["."]) for s in sents)
        out = []
        for s, d in zip(sents, nlp.pipe(docs, batch_size=256)):
            out.append([(t.lemma_ if not re.search(r"\d", t.text) else t.text) for t in d][: len(s)] if s else [])
        return out
    if normalizer == "none":
        return [[t for t in s] for s in sents]
    raise ValueError(normalizer)


def lemma_context_free(word: str) -> str:
    """Context-free lemma: WordNet verb -> noun -> adjective, first form that changes.

    The same word always gets the same lemma, in a document or in a two-word query.
    (spaCy's lemma depends on the sentence: "Tracing" in a title is tagged NNP and kept,
    while the query word "tracing" becomes "trace", so the phrase never matches.)
    """
    if re.search(r"\d", word) or not word.isalpha():
        return word.lower()
    w = word.lower()
    for pos in ("v", "n", "a"):
        lem = lemma_wordnet(w, {"v": "VB", "n": "NN", "a": "JJ"}[pos])
        if lem != w:
            return lem
    return w


# ---------------------------------------------------------------- corpus-level caches
@lru_cache(maxsize=4)
def cleaned_corpus(key: str = "assessment") -> tuple:
    docs = load_corpus()
    return tuple((d.doc_id, clean_text(d.text, d.fmt)[0]) for d in docs)


def _cache_key(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def _disk(name: str, key: str, build):
    path = CACHE / f"{name}_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    val = build()
    path.write_bytes(pickle.dumps(val))
    return val


def corpus_fingerprint(corpus: tuple) -> str:
    return hashlib.md5("".join(t for _, t in corpus).encode()).hexdigest()[:10]


def tokenized(corpus: tuple, tokenizer: str) -> dict[str, list[list[str]]]:
    key = _cache_key(corpus_fingerprint(corpus), tokenizer)
    return _disk("tok", key, lambda: {did: [_tokens(s, tokenizer) for s in split_sentences(text)]
                                      for did, text in corpus})


def normalized(corpus: tuple, tokenizer: str, normalizer: str) -> dict[str, list[list[str]]]:
    key = _cache_key(corpus_fingerprint(corpus), tokenizer, normalizer)

    def build():
        toks = tokenized(corpus, tokenizer)
        return {did: _normalize_sentences(sents, normalizer) for did, sents in toks.items()}

    return _disk("norm", key, build)


# ---------------------------------------------------------------- the pipeline
@dataclass
class Pipeline:
    name: str
    tokenizer: str = "hybrid"
    stopwords: str = "custom"
    normalizer: str = "spacy"
    order: str = "norm_then_stop"
    label: str = ""
    description: str = ""
    compound_parts: bool = False
    posthoc: bool = False

    @classmethod
    def from_config(cls, name: str) -> "Pipeline":
        c = load_yaml("pipelines.yaml")[name]
        return cls(name=name, **{k: c[k] for k in ("tokenizer", "stopwords", "normalizer", "order", "label",
                                                   "description", "compound_parts", "posthoc") if k in c})

    def config(self) -> dict:
        return {"tokenizer": self.tokenizer, "stopwords": self.stopwords, "normalizer": self.normalizer,
                "order": self.order, "compound_parts": self.compound_parts}

    def _parts(self, tok: str, stop: frozenset) -> list[str]:
        """Parts of a hyphenated compound, indexed at the compound's position (LLM-based -> llm, based)."""
        if not self.compound_parts or "-" not in tok or not re.search(r"[A-Za-z]", tok):
            return []
        out = []
        for part in tok.split("-"):
            if part and is_word(part):
                nm = lemma_context_free(part) if self.normalizer == "wordnet_cf" else part.lower()
                t = normalize_term(nm)
                if t not in stop and t not in out:
                    out.append(t)
        return out

    # ---- the per-token decision, shared by corpus and query processing
    def _keep(self, surface: str, norm: str, stop: frozenset) -> str | None:
        if not is_word(surface):
            return None
        term = normalize_term(norm)
        check = surface.lower() if self.order == "stop_then_norm" else term
        if check in stop:
            return None
        return term

    def run(self, corpus: tuple | None = None) -> "PipelineRun":
        corpus = corpus or cleaned_corpus()
        t0 = time.perf_counter()
        toks = tokenized(corpus, self.tokenizer)
        norms = normalized(corpus, self.tokenizer, self.normalizer)
        stop = stopwords(self.stopwords)
        docs_terms, surfaces, sent_streams = {}, {}, {}
        stage_tokens = Counter()
        stage_vocab: dict[str, set] = {s: set() for s in STAGES}
        for did, sents in toks.items():
            pairs, pos, surf = [], 0, {}
            stream = []
            for s, ns in zip(sents, norms[did]):
                sent_norm = []
                for tok, nm in zip(s, ns):
                    stage_tokens["tokenized"] += 1
                    stage_vocab["tokenized"].add(tok)
                    stage_tokens["lowercased"] += 1
                    stage_vocab["lowercased"].add(tok.lower())
                    stop_hit = (tok.lower() in stop) if self.order == "stop_then_norm" else (normalize_term(nm) in stop)
                    if not stop_hit:
                        stage_tokens["stopwords_removed"] += 1
                        stage_vocab["stopwords_removed"].add(tok.lower())
                    stage_tokens["normalized"] += 1
                    stage_vocab["normalized"].add(normalize_term(nm))
                    term = self._keep(tok, nm, stop)
                    sent_norm.append(normalize_term(nm) if is_word(tok) else tok)
                    if term:
                        pairs.append((term, pos))
                        stage_tokens["index_terms"] += 1
                        stage_vocab["index_terms"].add(term)
                        surf.setdefault(term, Counter())[tok.lower()] += 1
                        for part in self._parts(tok, stop):
                            if part != term:
                                pairs.append((part, pos))
                                stage_tokens["index_terms"] += 1
                                stage_vocab["index_terms"].add(part)
                                surf.setdefault(part, Counter())[tok.lower()] += 1
                    pos += 1
                pos += SENT_GAP
                stream.append(sent_norm)
            docs_terms[did] = pairs
            surfaces[did] = surf
            sent_streams[did] = stream
        index = PositionalIndex.build(docs_terms)
        seconds = time.perf_counter() - t0
        stages = [{"stage": s, "tokens": stage_tokens[s], "vocabulary": len(stage_vocab[s])} for s in STAGES]
        return PipelineRun(self, index, docs_terms, surfaces, sent_streams, stages, seconds, stop)

    def analyzer(self):
        stop = stopwords(self.stopwords)

        def analyze(text: str) -> list[tuple[str, int]]:
            sents = [_tokens(s, self.tokenizer) for s in split_sentences(text)] or [[]]
            norms = _normalize_sentences(sents, self.normalizer)
            out, pos = [], 0
            for s, ns in zip(sents, norms):
                for tok, nm in zip(s, ns):
                    term = self._keep(tok, nm, stop)
                    if term:
                        out.append((term, pos))
                    pos += 1
                pos += SENT_GAP
            return out

        return analyze


@dataclass
class PipelineRun:
    pipeline: Pipeline
    index: PositionalIndex
    docs_terms: dict
    surfaces: dict
    sent_streams: dict  # normalised tokens per sentence BEFORE stop-word removal (for n-grams)
    stages: list
    seconds: float
    stop: frozenset = field(default_factory=frozenset)

    def engine(self) -> QueryEngine:
        return QueryEngine(self.index, self.pipeline.analyzer())

    def term_counts(self) -> Counter:
        c = Counter()
        for pairs in self.docs_terms.values():
            c.update(t for t, _ in pairs)
        return c

    def summary(self) -> dict:
        return {"run": self.pipeline.name, "label": self.pipeline.label, **self.pipeline.config(),
                "token_count": sum(len(p) for p in self.docs_terms.values()), "vocabulary_size": self.index.vocabulary,
                "index_seconds": round(self.seconds, 2)}


def run_named(name: str) -> PipelineRun:
    return Pipeline.from_config(name).run()


def all_run_names() -> list[str]:
    return list(load_yaml("pipelines.yaml"))


def build_run_for_documents(docs: list[Document], pipeline: Pipeline) -> PipelineRun:
    corpus = tuple((d.doc_id, clean_text(d.text, d.fmt)[0]) for d in docs)
    return pipeline.run(corpus)
