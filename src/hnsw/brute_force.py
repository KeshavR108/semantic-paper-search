"""Exact k-nearest-neighbor search. This is the ground truth that HNSW recall is
measured against, so it favors correctness over speed:

- distances are computed in float64 so rounding can't reorder near-ties;
- ties are broken by the smaller id, so results are deterministic;
- queries are processed in chunks so the (queries x base) distance matrix
  never exceeds a fixed memory budget.
"""

import numpy as np

from .distance import METRICS, _quiet_matmul, normalize

# Max number of float64 entries in one chunk's distance matrix (~80 MB).
_CHUNK_ENTRIES = 10_000_000


def knn(queries: np.ndarray, base: np.ndarray, k: int, metric: str = "l2"):
    """Return (ids, dists), each shaped (n_queries, min(k, n_base)), nearest first."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    if k < 1:
        raise ValueError("k must be >= 1")

    Q = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    X = np.asarray(base, dtype=np.float64)
    if metric == "cosine":
        Q, X = normalize(Q), normalize(X)

    n_base = X.shape[0]
    k = min(k, n_base)
    x_sq = np.einsum("ij,ij->i", X, X)
    chunk = max(1, _CHUNK_ENTRIES // n_base)

    all_ids = np.empty((Q.shape[0], k), dtype=np.int64)
    all_dists = np.empty((Q.shape[0], k), dtype=np.float64)

    for start in range(0, Q.shape[0], chunk):
        q = Q[start:start + chunk]
        with _quiet_matmul():
            if metric == "l2":
                q_sq = np.einsum("ij,ij->i", q, q)
                D = np.maximum(q_sq[:, None] - 2.0 * (q @ X.T) + x_sq[None, :], 0.0)
            else:
                D = 1.0 - q @ X.T

        for row, d in enumerate(D):
            ids = _top_k_with_ties(d, k)
            all_ids[start + row] = ids
            all_dists[start + row] = d[ids]

    return all_ids, all_dists


def _top_k_with_ties(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest values of d, sorted by (distance, id).

    argpartition finds the k-th smallest value in O(n) without sorting everything.
    Any element tied with that value might belong in the top k, so we keep all
    of them, then sort the short list by distance and break ties by smaller id.
    """
    if k < len(d):
        kth = d[np.argpartition(d, k - 1)[k - 1]]
        cand = np.flatnonzero(d <= kth)
    else:
        cand = np.arange(len(d))
    order = np.lexsort((cand, d[cand]))  # last key is primary: distance, then id
    return cand[order][:k]
