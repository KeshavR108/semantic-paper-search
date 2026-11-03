import numpy as np

METRICS = ("l2", "cosine")

def _quiet_matmul():
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def l2_sq_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X - q
    return np.einsum("ij,ij->i", diff, diff)


def cosine_dist_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    with _quiet_matmul():
        return 1.0 - X @ q


def l2_sq_pairwise(X: np.ndarray) -> np.ndarray:
    sq = np.einsum("ij,ij->i", X, X)
    with _quiet_matmul():
        D = sq[:, None] - 2.0 * (X @ X.T) + sq[None, :]
    return np.maximum(D, 0.0)


def cosine_dist_pairwise(X: np.ndarray) -> np.ndarray:
    with _quiet_matmul():
        return 1.0 - X @ X.T


def normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector (cosine metric)")
    return X / norms


def get_metric(metric: str):
    if metric == "l2":
        return l2_sq_batch, l2_sq_pairwise
    if metric == "cosine":
        return cosine_dist_batch, cosine_dist_pairwise
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
