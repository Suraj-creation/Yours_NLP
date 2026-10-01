"""Module 6 metrics, checked against hand-computed values."""
import pytest

from nlp_core.evaluate import cohen_kappa, confusion_matrix, ner_scores, prf, retrieval_scores, tagging_accuracy


def test_prf():
    r = prf(8, 2, 4)
    assert r["precision"] == pytest.approx(0.8, abs=1e-3) and r["recall"] == pytest.approx(8 / 12, abs=1e-3)
    assert r["f1"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12), abs=1e-3)


def test_retrieval_scores_by_hand():
    r = retrieval_scores(["D1", "D2", "D3", "D4"], {"D1", "D3", "D9"})
    assert r["precision"] == 0.5 and r["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert r["p@3"] == pytest.approx(0.6667, abs=1e-4) and r["r@3"] == pytest.approx(0.6667, abs=1e-4)
    assert r["ap"] == pytest.approx((1 / 1 + 2 / 3) / 3, abs=1e-4)
    assert r["hits"] == 2 and r["retrieved"] == 4 and r["relevant"] == 3


def test_empty_result():
    r = retrieval_scores([], {"D1"})
    assert r["precision"] == 0 and r["recall"] == 0 and r["f1"] == 0


def test_cohen_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    # 10 items, agreement 0.8, both 50/50 marginals -> kappa 0.6
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.6)


def test_tagging_and_confusion():
    acc = tagging_accuracy([["VB", "CD"]], [["NN", "CD"]])
    assert acc["accuracy"] == 0.5
    cm = confusion_matrix([["VB", "CD"]], [["NN", "CD"]], ["CD", "NN", "VB"])
    assert cm == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]


def test_ner_scores_outcomes():
    gold = [[(0, 5, "PERSON"), (10, 13, "KT_MODEL"), (20, 25, "MONEY")]]
    pred = [[(0, 5, "PERSON"), (10, 13, "ORG"), (30, 33, "DATE")]]
    s = ner_scores(gold, pred)["overall"]
    assert (s["correct"], s["incorrect"], s["missed"], s["spurious"]) == (1, 1, 1, 1)
