"""search_layer on a hand-built 6-node graph (2D, squared L2), query q = (5, 0).

    3 (0,3) ---- 4 (3,3)
    |              \\
    0 (0,0) -- 1 (2,0)   5 (5,0.5)
    |
    2 (-1,0)

Squared distances to q:  0:25  1:9  2:36  3:34  4:13  5:0.25

Node 1 is a trap: it's the closest neighbor of the start node, but a dead end.
The true nearest node, 5, is only reachable by first moving AWAY from q (0 -> 3).
"""

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
    # Greedy (ef=1) moves to node 1 (dist 9) and stops: node 1's only neighbor is 0.
    assert index.search_layer(Q, [0], ef=1, layer=0) == [(9.0, 1)]


def test_wider_beam_escapes_local_minimum(index):
    # ef=3 keeps node 3 (dist 34) in the beam long enough to walk 3 -> 4 -> 5.
    result = index.search_layer(Q, [0], ef=3, layer=0)
    assert result == [(0.25, 5), (9.0, 1), (13.0, 4)]


def test_results_sorted_nearest_first_and_capped_at_ef(index):
    result = index.search_layer(Q, [0], ef=6, layer=0)
    dists = [d for d, _ in result]
    assert dists == sorted(dists)
    assert len(result) == 6          # ef >= N on a connected graph: everything found
    assert {e for _, e in result} == set(range(6))


def test_visited_set_is_fresh_each_call(index):
    first = index.search_layer(Q, [0], ef=3, layer=0)
    second = index.search_layer(Q, [0], ef=3, layer=0)
    assert first == second


def test_more_entry_points_than_ef(index):
    # W starts with {4:13, 2:36, 0:25}, is trimmed to the best 2 {4, 0},
    # then expanding 4 finds 5 and evicts 0.
    result = index.search_layer(Q, [4, 2, 0], ef=2, layer=0)
    assert [e for _, e in result] == [5, 4]
