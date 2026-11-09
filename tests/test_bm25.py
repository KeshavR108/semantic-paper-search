import math

import numpy as np
import pytest

from papersearch import BM25, tokenize

DOCS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "dogs and cats are pets",
    "quantum computing uses qubits",
    "cat cat cat cat cat cat cat cat",
]


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("The Cat-sat on: the MAT!") == ["cat", "sat", "mat"]
    assert tokenize("IL-6 & TNF2") == ["il", "6", "tnf2"]


def test_matches_hand_computed_score():
    bm = BM25(DOCS, k1=1.2, b=0.75)
    idf = math.log(1 + (5 - 3 + 0.5) / (3 + 0.5))
    lens = [3, 3, 3, 4, 8]
    avg = sum(lens) / 5
    def expected(tf, dl):
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avg))
    s = bm.scores("cat")
    assert s[0] == pytest.approx(expected(1, 3))
    assert s[4] == pytest.approx(expected(8, 8))
    assert s[2] == 0 and s[3] == 0


def test_only_documents_sharing_a_word_get_a_score():
    bm = BM25(DOCS)
    s = bm.scores("qubits")
    assert s[3] > 0 and np.count_nonzero(s) == 1
    assert not bm.scores("automobile").any()


def test_rare_words_weigh_more_than_common_ones():
    bm = BM25(DOCS)
    assert bm.idf("qubits") > bm.idf("cat") > 0


def test_term_frequency_saturates():
    bm = BM25(["cat", "cat " * 8, "dog", "bird"], b=0.0)
    s = bm.scores("cat")
    assert s[0] < s[1] < 2.2 * s[0]


def test_length_normalization_penalizes_long_documents():
    short = "telomerase activity"
    long = "telomerase activity " + "filler " * 50
    bm = BM25([short, long, "other words", "more other words"])
    s = bm.scores("telomerase")
    assert s[0] > s[1]


def test_search_orders_best_first_with_deterministic_ties():
    bm = BM25(["apple pie", "apple pie", "banana", "apple"])
    ids, scores = bm.search("apple", k=3)
    assert ids.tolist() == [3, 0, 1]
    assert scores[0] > scores[1] == scores[2]
    assert len(bm.search("apple", k=99)[0]) == 4
