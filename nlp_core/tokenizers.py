"""Module 2, Exercises 2, 3 and 6: tokenization.

Built-in tokenizers (NLTK variants, spaCy), a rule-based Custom tokenizer and a
Hybrid tokenizer (spaCy + protected domain spans). The domain rules exist
because default tokenizers break exactly the strings this domain depends on:

  1 fractions            3/4, 12/5
  2 expressions          1/2+1/3=2/5, 2x+9, 3-5
  3 mixed numbers        2 1/4, 3 1/2 (after cleaning turns 3½ into "3 1/2")
  4 hyphenated terms     knowledge-tracing, k-means, LLM-based, teacher-described
  5 acronyms, abbrev.    B.Tech, GPT-4o, Q-MCKT, ASSIST2009, e.g., Ms.
  6 currency, percent    ₹50,000, $3,650, 12.5%
  7 decimals, ratios     0.75, 3:5, 16:02:10, 2026-09-01
  8 learner informal     don't, cant, idk, u  (contractions kept whole)

Exercise 6 decision: this domain needs numbers (2/5 vs 5/6 separates a
misconception from a slip) and dates (tutoring timestamps), so both are kept as
single tokens rather than split or dropped.
"""
from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------- domain patterns
_CUR = r"(?:₹|\$|€|£)"
_ATOM = r"(?:\d+(?:[./]\d+)*[a-z]?(?![A-Za-z0-9])|[a-z](?![A-Za-z]))"
RULES: dict[str, str] = {
    "url_email": r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+",
    "datetime": r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?",
    "currency": _CUR + r"\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|" + _CUR + r"\s?\d+(?:\.\d+)?",
    "mixed_number": r"(?<![\d/.])\d+\s\d+/\d+(?![\d/])",
    "expression": r"(?<![\w/.])(?=[^\s]*\d)" + _ATOM + r"(?:[+\-×·*÷=^<>]" + _ATOM + r")+",
    "fraction": r"(?<![\w/])\d+/\d+(?![\d/])",
    "percent": r"\d+(?:\.\d+)?%",
    "ratio_time": r"(?<![\d:])\d+:\d+(?::\d+)?(?![\d:])",
    "number": r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|(?<![\w.])\.\d+",
    "ordinal": r"\d+(?:st|nd|rd|th)\b",
    "abbreviation": r"(?:e\.g\.|i\.e\.|etc\.|vs\.|Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|Fig\.|No\.|St\.|Rs\.|"
                    r"B\.Tech|M\.Tech|B\.Sc|M\.Sc|Ph\.D\.?|a\.m\.|p\.m\.)|(?:[A-Z]\.){2,}",
    "compound": r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)+",
    "alnum": r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*",
    "informal": r"(?i:\b(?:dont|cant|wont|isnt|doesnt|didnt|aint|wasnt|shouldnt|couldnt|wouldnt|havent|"
                r"hasnt|thats|whats|youre|theyre|idk|gonna|wanna|gotta)\b)",
}
DEFAULT_RULES = list(RULES)
_GENERIC = r"[^\W\d_]+|\d+|\.\.\.|[^\w\s]"


@lru_cache(maxsize=64)
def _pattern(rules: tuple[str, ...]) -> re.Pattern:
    alts = [RULES[r] for r in rules] + [_GENERIC]
    return re.compile("|".join(f"(?:{a})" for a in alts))


@lru_cache(maxsize=64)
def _protect(rules: tuple[str, ...]) -> re.Pattern | None:
    alts = [RULES[r] for r in rules if r != "url_email"]
    return re.compile("|".join(f"(?:{a})" for a in alts)) if alts else None


# ---------------------------------------------------------------- built-in tokenizers
def nltk_word(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    return word_tokenize(text)


def nltk_wordpunct(text: str) -> list[str]:
    from nltk.tokenize import wordpunct_tokenize

    return wordpunct_tokenize(text)


def nltk_treebank(text: str) -> list[str]:
    from nltk.tokenize import TreebankWordTokenizer

    return TreebankWordTokenizer().tokenize(text)


def nltk_tweet(text: str) -> list[str]:
    from nltk.tokenize import TweetTokenizer

    return TweetTokenizer().tokenize(text)


def whitespace(text: str) -> list[str]:
    return text.split()


def spacy_tok(text: str) -> list[str]:
    from .nlp_models import blank

    return [t.text for t in blank()(text)]


def custom(text: str, rules: list[str] | None = None) -> list[str]:
    """Rule-based tokenizer: ordered regex alternatives (earlier rules win)."""
    return _pattern(tuple(rules or DEFAULT_RULES)).findall(text)


# ---------------------------------------------------------------- hybrid
def _base_spacy(vocab=None):
    """spaCy English tokenizer with the letter-hyphen-letter infix removed."""
    import spacy
    from spacy.lang.char_classes import ALPHA, HYPHENS
    from spacy.util import compile_infix_regex

    from spacy.util import compile_prefix_regex

    nlp = spacy.blank("en") if vocab is None else spacy.blank("en", vocab=vocab)
    hyphen_rule = r"(?<=[{a}0-9])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS)
    infixes = [p for p in nlp.Defaults.infixes if p != hyphen_rule]
    # Found by the corpus audit (Exercise 2): spaCy keeps comma lists of fractions such as
    # "5/4,3/2,5/5" and "2013,44%" as one token, and treats "·" (times) as a letter joint.
    infixes += [r"(?<=[0-9]),(?=-?[0-9]+/)", r"(?<=/[0-9]),(?=-?[0-9])", r"(?<=/[0-9][0-9]),(?=-?[0-9])",
                r"(?<=[0-9]),(?=[0-9]+(?:\.[0-9]+)?%)", r"(?<=[0-9])/(?=\()"]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    # a leading minus sign is its own token ("-1/4" -> "-", "1/4"), as in the gold conventions
    nlp.tokenizer.prefix_search = compile_prefix_regex(list(nlp.Defaults.prefixes) + [r"-(?=[0-9])"]).search
    for abbr in ["B.Tech", "M.Tech", "B.Sc", "M.Sc", "Ph.D", "Ph.D.", "Rs."]:
        from spacy.symbols import ORTH

        nlp.tokenizer.add_special_case(abbr, [{ORTH: abbr}])
    return nlp.tokenizer


class HybridTokenizer:
    """spaCy tokenization + regex-protected domain spans merged back into single tokens.

    Callable(text) -> spacy Doc, so it can replace nlp.tokenizer and the tagger,
    parser and NER then run on the domain-correct tokens.
    """

    def __init__(self, vocab=None, rules: list[str] | None = None):
        self.base = _base_spacy(vocab)
        self.vocab = self.base.vocab
        self.rules = tuple(rules or DEFAULT_RULES)

    def __call__(self, text: str):
        doc = self.base(text)
        pat = _protect(self.rules)
        if pat is None:
            return doc
        spans, last_end = [], -1
        for m in pat.finditer(text):
            if m.start() < last_end or m.end() - m.start() < 2:
                continue
            sp = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if sp is not None and len(sp) > 1 and (not spans or sp.start >= spans[-1].end):
                spans.append(sp)
                last_end = m.end()
        if spans:
            with doc.retokenize() as r:
                for sp in spans:
                    r.merge(sp)
        return doc

    def to_disk(self, *a, **k):  # spaCy serialisation hooks (not needed)
        pass

    def from_disk(self, *a, **k):
        return self


@lru_cache(maxsize=16)
def _hybrid(rules: tuple[str, ...]) -> HybridTokenizer:
    return HybridTokenizer(rules=list(rules))


def hybrid(text: str, rules: list[str] | None = None) -> list[str]:
    return [t.text for t in _hybrid(tuple(rules or DEFAULT_RULES))(text)]


def mwe(tokens: list[str], phrases: list[str]) -> list[str]:
    """NLTK MWETokenizer: join known multi-word terms ("knowledge tracing" -> knowledge_tracing)."""
    from nltk.tokenize import MWETokenizer

    return MWETokenizer([tuple(p.split()) for p in phrases], separator="_").tokenize(tokens)


TOKENIZERS = {
    "whitespace": whitespace,
    "nltk": nltk_word,
    "nltk_wordpunct": nltk_wordpunct,
    "nltk_treebank": nltk_treebank,
    "nltk_tweet": nltk_tweet,
    "spacy": spacy_tok,
    "custom": custom,
    "hybrid": hybrid,
}
LABELS = {"whitespace": "Whitespace", "nltk": "NLTK word_tokenize", "nltk_wordpunct": "NLTK wordpunct",
          "nltk_treebank": "NLTK Treebank", "nltk_tweet": "NLTK Tweet", "spacy": "spaCy",
          "custom": "Custom (regex rules)", "hybrid": "Hybrid (spaCy + rules)"}


def tokenize(text: str, name: str = "hybrid", rules: list[str] | None = None) -> list[str]:
    fn = TOKENIZERS[name]
    return fn(text, rules) if name in ("custom", "hybrid") else fn(text)


# ---------------------------------------------------------------- alignment (for boundary scoring)
_QUOTE_FORMS = {"``": '"', "''": '"'}


def align(tokens: list[str], text: str) -> list[tuple[int, int]]:
    """Map tokens back to character spans in text (NLTK rewrites quotes, so try variants)."""
    spans, cur = [], 0
    for tok in tokens:
        cands = [tok, _QUOTE_FORMS.get(tok, tok), tok.replace("``", '"').replace("''", '"')]
        best = None
        for c in cands:
            i = text.find(c, cur)
            if i != -1 and (best is None or i < best[0]):
                best = (i, i + len(c))
        if best is None:  # token not recoverable: zero-width span at cursor
            spans.append((cur, cur))
            continue
        spans.append(best)
        cur = best[1]
    return spans


def token_rule_hits(text: str, rules: list[str] | None = None) -> list[dict]:
    """Which domain rule protected which string (shown on the website)."""
    out = []
    for name in rules or DEFAULT_RULES:
        for m in re.finditer(RULES[name], text):
            out.append({"rule": name, "text": m.group(0), "start": m.start(), "end": m.end()})
    return sorted(out, key=lambda x: x["start"])
