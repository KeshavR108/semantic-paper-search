"""Distance functions. Every function returns a *distance*: lower means more similar.

Two metrics are supported:
- "l2":     squared Euclidean distance. We skip the sqrt because sqrt is monotonic,
            so it never changes which vector is closer, and it costs time.
- "cosine": 1 - dot(a, b), which equals cosine distance ONLY when a and b have
            length 1. The index normalizes vectors once at insert time and
            normalizes each query, so this precondition always holds inside it.
"""

import numpy as np

METRICS = ("l2", "cosine")

# NumPy 2.x linked against Apple's Accelerate BLAS can emit spurious
# "divide by zero / overflow / invalid value in matmul" warnings on Apple
# Silicon even though the results are correct (checked against a float64
# reference in tests/test_distance.py). We silence them only around matmuls.
def _quiet_matmul():
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


# ---- single pair ---------------------------------------------------------------

def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    # Assumes a and b are already unit length.
    return float(1.0 - np.dot(a, b))


# ---- one query vs. many vectors (the hot path inside the index) ------------------

def l2_sq_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared L2 distance from q to every row of X. Shape (n,)."""
    diff = X - q
    # einsum computes the row-wise dot product diff[i] . diff[i] without
    # building another (n, d) temporary like (diff ** 2).sum(1) would.
    return np.einsum("ij,ij->i", diff, diff)


def cosine_dist_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine distance from unit vector q to every unit row of X. Shape (n,)."""
    with _quiet_matmul():
        return 1.0 - X @ q


# ---- all pairs within one set (used by the neighbor-selection heuristic) --------

def l2_sq_pairwise(X: np.ndarray) -> np.ndarray:
    """Squared L2 distance between every pair of rows of X. Shape (n, n).

    Uses ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, so the whole matrix comes from
    one matrix multiply. Rounding can make tiny results slightly negative, so clip.
    """
    sq = np.einsum("ij,ij->i", X, X)
    with _quiet_matmul():
        D = sq[:, None] - 2.0 * (X @ X.T) + sq[None, :]
    return np.maximum(D, 0.0)


def cosine_dist_pairwise(X: np.ndarray) -> np.ndarray:
    with _quiet_matmul():
        return 1.0 - X @ X.T


# ---- helpers ---------------------------------------------------------------------

def normalize(X: np.ndarray) -> np.ndarray:
    """Scale each row (or a single vector) to length 1. Zero vectors are rejected
    because they have no direction, so cosine distance is undefined for them."""
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector (cosine metric)")
    return X / norms


def get_metric(metric: str):
    """Return (batch_fn, pairwise_fn) for a metric name."""
    if metric == "l2":
        return l2_sq_batch, l2_sq_pairwise
    if metric == "cosine":
        return cosine_dist_batch, cosine_dist_pairwise
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
