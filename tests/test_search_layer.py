import numpy as np
import pytest

from hnsw import HNSWIndex

POINTS = [(0, 0), (2, 0), (-1, 0), (0, 3), (3, 3), (5, 0.5)]
LINKS = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0, 4], 4: [3, 5], 5: [4]}
Q = np.array([5.0, 0.0], dtype=np.float32)


@pytest.fixture
def index():
    idx = HNSWIndex(dim=2, M=4)
    idx.vectors = np.array(POINTS, dtype=np.float32)
    idx.count = len(POINTS)
    idx.levels = [0] * len(POINTS)
    idx.graph = [[LINKS[n]] for n in range(len(POINTS))]
    idx.entry_point, idx.max_level = 0, 0
    return idx


def test_ef1_is_greedy_and_gets_stuck_in_local_minimum(index):
    assert index.search_layer(Q, [0], ef=1, layer=0) == [(9.0, 1)]


def test_wider_beam_escapes_local_minimum(index):
    result = index.search_layer(Q, [0], ef=3, layer=0)
    assert result == [(0.25, 5), (9.0, 1), (13.0, 4)]


def test_results_sorted_nearest_first_and_capped_at_ef(index):
    result = index.search_layer(Q, [0], ef=6, layer=0)
    dists = [d for d, _ in result]
    assert dists == sorted(dists)
    assert len(result) == 6
    assert {e for _, e in result} == set(range(6))


def test_visited_set_is_fresh_each_call(index):
    first = index.search_layer(Q, [0], ef=3, layer=0)
    second = index.search_layer(Q, [0], ef=3, layer=0)
    assert first == second


def test_more_entry_points_than_ef(index):
    result = index.search_layer(Q, [4, 2, 0], ef=2, layer=0)
    assert [e for _, e in result] == [5, 4]
