"""Exercise 2/3/6: the domain rules keep mathematical tokens whole."""
import pytest

from nlp_core import tokenizers as T
from nlp_core.evaluate import boundary_scores
from nlp_core.config import GOLD
import json


@pytest.mark.parametrize("name", ["custom", "hybrid"])
@pytest.mark.parametrize("text,token", [
    ("I ate 3/4 of it", "3/4"),
    ("3 1/2 is the same as 7/2", "3 1/2"),
    ("I did 1/2+1/3=2/5, idk", "1/2+1/3=2/5"),
    ("She paid $3,650 for it", "$3,650"),
    ("a 46% increase", "46%"),
    ("the ratio 3:5", "3:5"),
    ("on 2026-09-01 we met", "2026-09-01"),
    ("an LLM-based tutor", "LLM-based"),
    ("knowledge-tracing models", "knowledge-tracing"),
    ("mail a.b@x.org today", "a.b@x.org"),
    ("see https://x.org/a?b=1 now", "https://x.org/a?b=1"),
    ("the 1st and 2nd", "1st"),
])
def test_domain_tokens_kept_whole(name, text, token):
    assert token in T.tokenize(text, name)


def test_nltk_splits_what_the_rules_protect():
    toks = T.tokenize("She paid $3,650 and 3 1/2 cups", "nltk")
    assert "$3,650" not in toks and "3 1/2" not in toks


def test_hybrid_splits_comma_lists_and_leading_minus_like_the_gold():
    assert T.tokenize("5/4,3/2 and -3/4", "hybrid") == ["5/4", ",", "3/2", "and", "-", "3/4"]


def test_rule_switch_changes_custom_output():
    all_rules = list(T.RULES)
    no_currency = [r for r in all_rules if r != "currency"]
    assert "$3,650" in T.tokenize("paid $3,650", "custom", all_rules)
    assert "$3,650" not in T.tokenize("paid $3,650", "custom", no_currency)


def test_align_spans_point_back_into_text():
    text = "Aarav wrote 1/2 + 1/3 = 2/5, idk why."
    toks = T.tokenize(text, "hybrid")
    for tok, (a, b) in zip(toks, T.align(toks, text)):
        assert text[a:b].replace(" ", "") == tok.replace(" ", "")


def test_every_registered_tokenizer_runs():
    for name in T.TOKENIZERS:
        assert T.tokenize("Simplify 12/8 now.", name)


def test_gold_scores_match_reported_ordering():
    gold = [json.loads(x) for x in open(GOLD / "tokenization_gold.jsonl", encoding="utf-8")]
    assert len(gold) == 40
    hy = boundary_scores(lambda s: T.tokenize(s, "hybrid"), gold)
    nl = boundary_scores(lambda s: T.tokenize(s, "nltk"), gold)
    ws = boundary_scores(lambda s: T.tokenize(s, "whitespace"), gold)
    assert hy["f1"] > 0.99 > nl["f1"] > ws["f1"]
