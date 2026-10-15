"""Neighbor selection on a hand-made 2D example (squared L2), base at the origin.

            e2 (0, 3.2)
            |
    base ●──────● r (2, 0)  ● e1 (3, 0.3)

Squared distance to base:  r: 4,  e1: 9.09,  e2: 10.24   (so sorted: r, e1, e2)

- e1 is closer to r (1.09) than to base (9.09): reachable via base -> r -> e1,
  so a direct link is redundant.
- e2 is closer to base (10.24) than to r (14.24): a genuinely new direction.

Simple selection (M=2) keeps the 2 nearest: r, e1 (both point right).
The heuristic (M=2) keeps r, e2 (two directions).
"""

import numpy as np
import pytest

from hnsw import HNSWIndex

BASE, R, E1, E2 = 0, 1, 2, 3


@pytest.fixture
def index():
    idx = HNSWIndex(dim=2, M=4)
    idx.vectors = np.array([(0, 0), (2, 0), (3, 0.3), (0, 3.2)], dtype=np.float32)
    idx.count = 4
    return idx


@pytest.fixture
def candidates(index):
    d = index._dists(index.vectors[BASE], [R, E1, E2])
    return sorted(zip(d, [R, E1, E2]))


def ids(selected):
    return [e for _, e in selected]


def test_candidates_are_sorted_as_expected(candidates):
    assert ids(candidates) == [R, E1, E2]


def test_simple_takes_nearest(index, candidates):
    assert ids(index.select_neighbors_simple(candidates, 2)) == [R, E1]


def test_heuristic_drops_redundant_keeps_diverse(index, candidates):
    assert ids(index.select_neighbors_heuristic(candidates, 2)) == [R, E2]


def test_heuristic_without_keep_pruned_leaves_slot_empty(index, candidates):
    assert ids(index.select_neighbors_heuristic(candidates, 3, keep_pruned=False)) == [R, E2]


def test_heuristic_keep_pruned_refills_up_to_M(index, candidates):
    assert ids(index.select_neighbors_heuristic(candidates, 3, keep_pruned=True)) == [R, E2, E1]


def test_heuristic_never_exceeds_M(index, candidates):
    assert len(index.select_neighbors_heuristic(candidates, 1)) == 1
