import numpy as np

from .distance import METRICS, _quiet_matmul, normalize

_CHUNK_ENTRIES = 10_000_000


def knn(queries: np.ndarray, base: np.ndarray, k: int, metric: str = "l2"):
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
    if k < len(d):
        kth = d[np.argpartition(d, k - 1)[k - 1]]
        cand = np.flatnonzero(d <= kth)
    else:
        cand = np.arange(len(d))
    order = np.lexsort((cand, d[cand]))
    return cand[order][:k]
