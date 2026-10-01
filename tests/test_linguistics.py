"""Exercises 5, 7-13: stop words, stemming, lemmas, POS rules, NER, BPE, n-grams, evidence."""
from collections import Counter

import pytest

from nlp_core import bpe as B
from nlp_core import pos as P
from nlp_core.ngrams import ngrams
from nlp_core.preprocess import lemmatize_tokens, stem, stopwords


def test_custom_stoplist_keeps_mathematical_words():
    custom, nltk = stopwords("custom"), stopwords("nltk")
    for w in ("not", "more", "than", "over", "under", "same", "each"):
        assert w in nltk and w not in custom
    for w in ("et", "al", "um", "uh"):
        assert w in custom


def test_stemmers_and_lemmatizers():
    assert stem("denominators", "porter") == "denomin"
    assert stem("numerator", "porter") == stem("numerical", "porter")  # the documented over-stemming error
    assert stem("3/4", "porter") == "3/4"  # numbers are never stemmed
    assert [x.lower() for x in lemmatize_tokens(["The", "students", "were", "adding"], "wordnet_pos")] == ["the", "student", "be", "add"]
    assert lemmatize_tokens(["students", "were", "adding"], "wordnet_nopos")[1:] == ["were", "adding"]


def test_rule_tagger_fixes_domain_errors():
    tags, fired = P.rule_tag(["Simplify", "12/8", "."])
    assert tags[0] == "VB" or fired[0] == ""
    tags, fired = P.rule_tag(["Multiply", "2/3", "x", "3/4", "."])
    assert tags[2] == "SYM" and fired[2] == "operator between numbers"
    tags, _ = P.rule_tag(["The", "LCD", "is", "12", "."])
    assert tags[1] == "NN"


def test_ml_taggers_run():
    toks = ["Round", "3.456", "to", "the", "nearest", "tenth", "."]
    assert len(P.backoff_tag(toks)) == len(toks)
    assert len(P.crf_tag(toks)) == len(toks)


def test_ner_ruler_adds_domain_and_money():
    from nlp_core.ner import entities

    text = "We compare BKT and DKT on MathDial using AUC; the tiles cost Rs 2,500 and $3,650."
    model = {(e["text"], e["label"]) for e in entities(text, False)}
    ruler = {(e["text"], e["label"]) for e in entities(text, True)}
    assert ("BKT", "KT_MODEL") in ruler and ("MathDial", "DATASET") in ruler and ("AUC", "METRIC") in ruler
    assert ("Rs 2,500", "MONEY") in ruler and ("$3,650", "MONEY") in ruler
    assert ("BKT", "KT_MODEL") not in model


def test_tiktoken_ranks_are_the_real_ones():
    enc = B.encoding("gpt2")
    assert enc.n_vocab == 50257 and enc.encode("Hello") == [15496]
    assert B.encoding("cl100k_base").n_vocab == 100277


def test_simple_bpe_learns_and_segments():
    words = Counter("denominator denominators numerator numerators fraction fractions".split() * 20)
    bpe = B.SimpleBPE().fit(words, 60)
    pieces = bpe.encode_word("denominators")
    assert pieces[-1].endswith("</w>") and "".join(pieces).replace("</w>", "") == "denominators"
    assert len(pieces) <= 3


def test_simple_tokenizer_v2_maps_unknown_words():
    tok = B.SimpleTokenizerV2("the LCD of 4 and 6 is 12 .")
    assert "<|unk|>" in tok.decode(tok.encode("the LCD is obviously 12 ."))


def test_ngrams_stay_inside_sentences():
    grams = ngrams([["add", "the", "tops"], ["then", "add"]], 2)
    assert ("tops", "then") not in grams and grams[("add", "the")] == 1


@pytest.mark.parametrize("text,malrule", [
    ("I did 1/2 + 1/3 = 2/5 because you add the tops and bottoms.", "M01"),
    ("1/8 is bigger than 1/6 because 8 is bigger than 6.", "M02"),
    ("0.25 is bigger than 0.3 because 25 is more than 3.", "M05"),
    ("0.4 × 0.2 = 0.8, you just multiply the numbers.", "M06"),
    ("2/3 is equal to 4/5 because I added 2 to the top and 2 to the bottom.", "M10"),
])
def test_evidence_detectors(text, malrule):
    from nlp_core.evidence import evidence_record

    rec = evidence_record(text)
    assert any(d["hypothesis"].startswith(malrule) for d in rec["interpretation"]["detectors"])
    assert rec["status"].startswith("PROTOTYPE") and rec["review_status"] == "unverified"


def test_evidence_correct_work_names_no_misconception():
    from nlp_core.evidence import evidence_record

    dets = evidence_record("1/2 + 1/3 = 5/6 because 3/6 + 2/6 = 5/6.")["interpretation"]["detectors"]
    assert dets and all(d["hypothesis"].startswith("correct") for d in dets)
