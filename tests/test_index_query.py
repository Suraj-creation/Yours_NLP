"""Modules 4-5: positional index and Boolean / phrase query processing."""
import math
import re

import pytest

from nlp_core.index import PositionalIndex
from nlp_core.query import QueryEngine, classify, lex, to_postfix

DOCS = {
    "D1": "Add the numerators. Keep the common denominator.",
    "D2": "The denominator is not common here.",
    "D3": "A common misconception: add the tops and add the bottoms.",
    "D4": "Knowledge tracing models predict answers.",
}


def analyzer(text):
    """Toy analyser: lower-case words; a sentence end leaves a one-position gap."""
    out, pos = [], 0
    for sent in re.split(r"[.:]", text):
        for w in re.findall(r"[a-z]+", sent.lower()):
            out.append((w, pos))
            pos += 1
        pos += 1
    return out


@pytest.fixture(scope="module")
def eng():
    return QueryEngine(PositionalIndex.build({d: analyzer(t) for d, t in DOCS.items()}), analyzer)


def docs(eng, q, mode="auto"):
    return eng.evaluate(q, mode)[0]


def test_keyword(eng):
    assert docs(eng, "denominator") == {"D1", "D2"}


def test_phrase_needs_adjacent_positions(eng):
    assert docs(eng, "common denominator") == {"D1"}
    assert docs(eng, '"add the tops"') == {"D3"}


def test_phrase_never_crosses_a_sentence_boundary(eng):
    # D1: "...numerators. Keep..." must not match "numerators keep"
    assert docs(eng, "numerators keep") == set()


def test_boolean_operators(eng):
    assert docs(eng, "denominator AND common") == {"D1", "D2"}
    assert docs(eng, "misconception OR tracing") == {"D3", "D4"}
    assert docs(eng, "denominator AND NOT here") == {"D1"}
    assert docs(eng, "NOT denominator") == {"D3", "D4"}  # complement within the collection


def test_brackets_and_precedence(eng):
    assert docs(eng, "(tracing OR misconception) AND add") == {"D3"}
    assert docs(eng, "tracing OR misconception AND add") == {"D3", "D4"}  # AND binds tighter


def test_lowercase_operators_are_words(eng):
    # "not common" is a phrase in D2, not a negation
    assert docs(eng, "not common") == {"D2"}


def test_keyword_mode_is_implicit_and_of_terms(eng):
    assert docs(eng, "numerators denominator", mode="keyword") == {"D1"}


def test_unbalanced_brackets_raise():
    with pytest.raises(ValueError):
        to_postfix(lex("(fraction AND misconception"))
    with pytest.raises(ValueError):
        to_postfix(lex("fraction )"))


def test_lexer_inserts_implicit_and():
    assert lex('fraction "common denominator"') == ["fraction", "AND", '"common denominator"']


@pytest.mark.parametrize("q,label", [
    ("denominator", "Keyword (unigram)"), ("common denominator", "Phrase (bigram)"),
    ("added the numerators", "Phrase (trigram)"), ("a AND b", "Boolean AND"), ("a OR b", "Boolean OR"),
    ("a AND NOT b", "Boolean NOT"), ("a AND (b OR c)", "Advanced Boolean"),
])
def test_classify(q, label):
    assert classify(q) == label


def test_tfidf_formula_and_ranking(eng):
    idx = eng.index
    tf = len(idx.postings["add"]["D3"])
    assert tf == 2
    assert idx.tfidf("add", "D3") == pytest.approx((1 + math.log10(2)) * math.log10(4 / 2))
    res = eng.search("add", repeats=1)
    assert res["doc_ids"] == ["D3", "D1"]  # higher tf ranks first


def test_json_round_trip(eng):
    data = eng.index.to_json()
    back = PositionalIndex.from_json(data)
    assert back.postings == eng.index.postings and back.N == eng.index.N
