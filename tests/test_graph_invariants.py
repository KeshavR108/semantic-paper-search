from collections import deque

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hnsw import HNSWIndex


def reachable_on_layer0(idx: HNSWIndex) -> int:
    """Count nodes reachable from the entry point by following layer-0 links
    (links are directed, so this is what a search can actually reach)."""
    seen = {idx.entry_point}
    queue = deque([idx.entry_point])
    while queue:
        for e in idx.graph[queue.popleft()][0]:
            if e not in seen:
                seen.add(e)
                queue.append(e)
    return len(seen)


def check_invariants(idx: HNSWIndex, min_reachable: float):
    n = idx.count
    assert len(idx.levels) == len(idx.graph) == n

    for node in range(n):
        assert len(idx.graph[node]) == idx.levels[node] + 1
        for layer, nbrs in enumerate(idx.graph[node]):
            cap = idx.M0 if layer == 0 else idx.M
            assert len(nbrs) <= cap, f"node {node} layer {layer} has {len(nbrs)} > {cap} links"
            assert node not in nbrs, f"self-loop at node {node}"
            assert len(set(nbrs)) == len(nbrs), f"duplicate neighbor at node {node}"
            for e in nbrs:
                assert 0 <= e < n, f"neighbor {e} does not exist"
                assert idx.levels[e] >= layer, f"neighbor {e} not on layer {layer}"

    assert idx.levels[idx.entry_point] == idx.max_level == max(idx.levels)

    reached = reachable_on_layer0(idx)
    assert reached >= min_reachable * n, f"only {reached}/{n} reachable on layer 0"


@pytest.fixture(scope="module")
def built_5k():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 16)).astype(np.float32)
    out = {}
    for selection in ("heuristic", "simple"):
        idx = HNSWIndex(dim=16, M=8, ef_construction=100, seed=1, neighbor_selection=selection)
        idx.add_batch(X)
        out[selection] = idx
    return out


def test_invariants_5k_heuristic(built_5k):
    check_invariants(built_5k["heuristic"], min_reachable=0.999)


def test_invariants_5k_simple(built_5k):
    # Simple selection is only a benchmark baseline. Its structure must still be
    # valid, but it is NOT expected to reach the 99.9% connectivity bar; see below.
    check_invariants(built_5k["simple"], min_reachable=0.0)


def test_heuristic_keeps_graph_better_connected_than_simple(built_5k):
    # Simple selection prunes each node's links down to its nearest neighbors.
    # A node whose only incoming links get pruned away becomes unreachable.
    # The heuristic keeps links in diverse directions, so fewer nodes are cut off.
    h = reachable_on_layer0(built_5k["heuristic"])
    s = reachable_on_layer0(built_5k["simple"])
    print(f"reachable on layer 0: heuristic {h}/5000, simple {s}/5000")
    assert h >= s


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(1, 300),
    dim=st.integers(1, 12),
    M=st.integers(2, 12),
    ef_construction=st.integers(1, 64),
    metric=st.sampled_from(["l2", "cosine"]),
    selection=st.sampled_from(["heuristic", "simple"]),
    seed=st.integers(0, 2**31 - 1),
)
def test_invariants_random_params(n, dim, M, ef_construction, metric, selection, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    idx = HNSWIndex(dim, metric=metric, M=M, ef_construction=ef_construction,
                    seed=seed, neighbor_selection=selection)
    idx.add_batch(X)
    # Structural invariants must always hold; connectivity is checked loosely
    # here because tiny ef_construction can legitimately produce weaker graphs.
    check_invariants(idx, min_reachable=0.0)
