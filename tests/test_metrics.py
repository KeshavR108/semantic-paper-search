import math

import pytest

from papersearch import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k():
    ranked = [["a", "b", "c", "d"], ["x", "y", "z", "w"]]
    relevant = [{"a", "d", "q"}, {"y"}]
    assert recall_at_k(ranked, relevant, 4) == pytest.approx((2 / 3 + 1) / 2)
    assert recall_at_k(ranked, relevant, 1) == pytest.approx((1 / 3 + 0) / 2)


def test_mrr_at_k():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
    relevant = [{"a"}, {"z"}, {"nope"}]
    assert mrr_at_k(ranked, relevant, 3) == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert mrr_at_k(ranked, relevant, 2) == pytest.approx((1 + 0 + 0) / 3)


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([["n", "r1", "n2", "r2"]], [{"r1", "r2"}], 10) == pytest.approx(dcg / ideal)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([["a", "b", "c"]], [{"a", "b"}], 10) == pytest.approx(1.0)
    assert ndcg_at_k([["c", "d"]], [{"a"}], 10) == 0.0


def test_ndcg_rewards_higher_ranks():
    top = ndcg_at_k([["r", "n", "n"]], [{"r"}], 10)
    low = ndcg_at_k([["n", "n", "r"]], [{"r"}], 10)
    assert top == pytest.approx(1.0) and low == pytest.approx(0.5)


def test_ndcg_ideal_is_capped_at_k():
    assert ndcg_at_k([["a", "b"]], [{"a", "b", "c"}], 2) == pytest.approx(1.0)
