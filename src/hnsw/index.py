import heapq
import math

import numpy as np

from .distance import METRICS, get_metric, normalize


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        seed: int = 0,
        neighbor_selection: str = "heuristic",
        initial_capacity: int = 1024,
    ):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        if M < 2:
            raise ValueError("M must be >= 2 (mL = 1/ln(M) is undefined for M=1)")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        if neighbor_selection not in ("heuristic", "simple"):
            raise ValueError("neighbor_selection must be 'heuristic' or 'simple'")

        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.neighbor_selection = neighbor_selection
        self.rng = np.random.default_rng(seed)
        self._dist_batch, self._dist_pairwise = get_metric(metric)

        self.vectors = np.empty((max(1, initial_capacity), dim), dtype=np.float32)
        self.count = 0
        self.levels: list[int] = []
        self.graph: list[list[list[int]]] = []
        self.entry_point: int | None = None
        self.max_level = -1

    def __len__(self) -> int:
        return self.count

    def _prepare(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {x.shape}")
        if self.metric == "cosine":
            x = normalize(x).astype(np.float32)
        return x

    def _dists(self, q: np.ndarray, ids: list[int]) -> list[float]:
        return self._dist_batch(q, self.vectors[ids]).tolist()

    def random_level(self) -> int:
        return int(math.floor(-math.log(1.0 - self.rng.random()) * self.mL))

    def search_layer(self, q: np.ndarray, entry_ids: list[int], ef: int, layer: int):
        visited = set(entry_ids)
        entry_d = self._dists(q, entry_ids)
        C = [(d, e) for d, e in zip(entry_d, entry_ids)]
        W = [(-d, e) for d, e in zip(entry_d, entry_ids)]
        heapq.heapify(C)
        heapq.heapify(W)
        while len(W) > ef:
            heapq.heappop(W)

        while C:
            d_c, c = heapq.heappop(C)
            if d_c > -W[0][0] and len(W) >= ef:
                break

            unvisited = [e for e in self.graph[c][layer] if e not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)

            for d_e, e in zip(self._dists(q, unvisited), unvisited):
                if len(W) < ef or d_e < -W[0][0]:
                    heapq.heappush(C, (d_e, e))
                    heapq.heappush(W, (-d_e, e))
                    if len(W) > ef:
                        heapq.heappop(W)

        return sorted((-neg_d, e) for neg_d, e in W)

    def select_neighbors_simple(self, candidates, M: int):
        return candidates[:M]

    def select_neighbors_heuristic(self, candidates, M: int, keep_pruned: bool = True):
        ids =[e for _, e in candidates]
        pair = self._dist_pairwise(self.vectors[ids]).tolist()

        selected: list[int] = []
        discarded: list[int] = []
        for i, (d, _) in enumerate(candidates):
            if len(selected) >= M:
                break
            row = pair[i]
            if all(row[j] > d for j in selected):
                selected.append(i)
            else:
                discarded.append(i)

        if keep_pruned:
            for i in discarded:
                if len(selected) >= M:
                    break
                selected.append(i)

        return [candidates[i] for i in selected]

    def _select(self, candidates, M: int):
        if self.neighbor_selection == "heuristic":
            return self.select_neighbors_heuristic(candidates, M)
        return self.select_neighbors_simple(candidates, M)

    def add(self, vec) -> int:
        x = self._prepare(vec)

        if self.count == len(self.vectors):
            grown = np.empty((2 * len(self.vectors), self.dim), dtype=np.float32)
            grown[: self.count] = self.vectors[: self.count]
            self.vectors = grown

        new_id = self.count
        self.vectors[new_id] = x
        self.count += 1

        level = self.random_level()
        self.levels.append(level)
        self.graph.append([[] for _ in range(level + 1)])

        if self.entry_point is None:
            self.entry_point = new_id
            self.max_level = level
            return new_id

        ep = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            ep = [self.search_layer(x, ep, 1, layer)[0][1]]

        for layer in range(min(level, self.max_level), -1, -1):
            W = self.search_layer(x, ep, self.ef_construction, layer)
            Mmax = self.M0 if layer == 0 else self.M
            neighbors = self._select(W, self.M)
            self.graph[new_id][layer] = [e for _, e in neighbors]

            for _, e in neighbors:
                links = self.graph[e][layer]
                links.append(new_id)
                if len(links) > Mmax:
                    d = self._dists(self.vectors[e], links)
                    cand = sorted(zip(d, links))
                    self.graph[e][layer] = [n for _, n in self._select(cand, Mmax)]

            ep = [e for _, e in W]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = new_id
        return new_id

    def add_batch(self, X) -> list[int]:
        return [self.add(x) for x in np.asarray(X)]

    def search(self, q, k: int, ef: int | None = None):
        if k < 1:
            raise ValueError("k must be >= 1")
        if ef is None:
            ef = max(k, 10)
        if ef < k:
            raise ValueError(f"ef ({ef}) must be >= k ({k}): W never holds more than ef results")
        if self.entry_point is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        x = self._prepare(q)
        ep = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            ep = [self.search_layer(x, ep, 1, layer)[0][1]]
        W = self.search_layer(x, ep, ef, 0)[:k]

        ids = np.array([e for _, e in W], dtype=np.int64)
        dists = np.array([d for d, _ in W], dtype=np.float32)
        return ids, dists

    def batch_search(self, Q, k: int, ef: int | None = None):
        Q = np.atleast_2d(np.asarray(Q))
        ids = np.full((len(Q), k), -1, dtype=np.int64)
        dists = np.full((len(Q), k), np.inf, dtype=np.float32)
        for i, q in enumerate(Q):
            r_ids, r_dists = self.search(q, k, ef)
            ids[i, : len(r_ids)] = r_ids
            dists[i, : len(r_dists)] = r_dists
        return ids, dists
