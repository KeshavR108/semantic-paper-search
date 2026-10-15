import numpy as np
import pytest

from hnsw.brute_force import knn
from hnsw.distance import (
    cosine_dist,
    cosine_dist_batch,
    cosine_dist_pairwise,
    l2_sq,
    l2_sq_batch,
    l2_sq_pairwise,
    normalize,
)


def naive_l2_sq(a, b):
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def naive_dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 16)), rng.standard_normal(16)


def test_l2_single_and_batch_match_naive(data):
    X, q = data
    expected = np.array([naive_l2_sq(q, x) for x in X])
    assert l2_sq(q, X[0]) == pytest.approx(expected[0], abs=1e-6)
    np.testing.assert_allclose(l2_sq_batch(q, X), expected, atol=1e-6)


def test_cosine_single_and_batch_match_naive(data):
    X, q = data
    Xn, qn = normalize(X), normalize(q)
    expected = np.array([1.0 - naive_dot(qn, x) for x in Xn])
    assert cosine_dist(qn, Xn[0]) == pytest.approx(expected[0], abs=1e-6)
    np.testing.assert_allclose(cosine_dist_batch(qn, Xn), expected, atol=1e-6)


def test_pairwise_match_naive(data):
    X, _ = data
    n = len(X)
    exp_l2 = np.array([[naive_l2_sq(X[i], X[j]) for j in range(n)] for i in range(n)])
    np.testing.assert_allclose(l2_sq_pairwise(X), exp_l2, atol=1e-6)
    Xn = normalize(X)
    exp_cos = np.array([[1.0 - naive_dot(Xn[i], Xn[j]) for j in range(n)] for i in range(n)])
    np.testing.assert_allclose(cosine_dist_pairwise(Xn), exp_cos, atol=1e-6)


def test_float32_pairwise_close_to_float64_reference():
    # Also guards against the spurious Accelerate matmul warnings hiding real errors.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 64)).astype(np.float32)
    ref = ((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1)
    D = l2_sq_pairwise(X)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(D, ref, rtol=1e-4, atol=1e-3)


def test_cosine_self_and_opposite():
    x = normalize(np.array([3.0, -4.0, 12.0]))
    assert cosine_dist(x, x) == pytest.approx(0.0, abs=1e-9)
    assert cosine_dist(x, -x) == pytest.approx(2.0, abs=1e-9)


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.zeros(4))


# ---- brute force ------------------------------------------------------------------

def naive_knn(queries, base, k, dist):
    out = []
    for q in queries:
        d = [(dist(q, x), i) for i, x in enumerate(base)]
        out.append([i for _, i in sorted(d)[:k]])  # sorted by (dist, id): ties -> smaller id
    return np.array(out)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_brute_force_matches_naive(metric):
    rng = np.random.default_rng(7)
    base, queries = rng.standard_normal((200, 8)), rng.standard_normal((20, 8))
    if metric == "l2":
        expected = naive_knn(queries, base, 10, naive_l2_sq)
    else:
        bn, qn = normalize(base), normalize(queries)
        expected = naive_knn(qn, bn, 10, lambda a, b: 1.0 - naive_dot(a, b))
    ids, dists = knn(queries, base, 10, metric)
    np.testing.assert_array_equal(ids, expected)
    assert np.all(np.diff(dists, axis=1) >= 0)  # nearest first


def test_brute_force_ties_prefer_smaller_id():
    base = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [5.0, 5.0]])
    q = np.zeros((1, 2))  # equidistant (distance 1) from ids 0..3
    ids, _ = knn(q, base, 2)
    np.testing.assert_array_equal(ids, [[0, 1]])
    ids, _ = knn(q, base, 4)
    np.testing.assert_array_equal(ids, [[0, 1, 2, 3]])


def test_brute_force_k_larger_than_n():
    base = np.eye(3)
    ids, dists = knn(np.ones((1, 3)), base, 10)
    assert ids.shape == (1, 3) and dists.shape == (1, 3)


def test_brute_force_chunking_matches_unchunked(monkeypatch):
    import hnsw.brute_force as bf
    rng = np.random.default_rng(3)
    base, queries = rng.standard_normal((500, 8)), rng.standard_normal((37, 8))
    full = knn(queries, base, 5)
    monkeypatch.setattr(bf, "_CHUNK_ENTRIES", 500 * 4)  # 4 queries per chunk
    chunked = knn(queries, base, 5)
    np.testing.assert_array_equal(full[0], chunked[0])
