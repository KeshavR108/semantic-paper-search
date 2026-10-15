"""HNSW index (Malkov & Yashunin, arXiv:1603.09320), Python + NumPy.

Algorithm numbers in comments refer to the paper:
  Alg. 1 = insert (add), Alg. 2 = search_layer, Alg. 3 = simple neighbor selection,
  Alg. 4 = heuristic neighbor selection, Alg. 5 = k-NN search.
"""

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
        self.M = M                      # max links per node on layers >= 1
        self.M0 = 2 * M                 # max links per node on layer 0
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)     # level normalization: P(level >= L) = M^-L
        self.neighbor_selection = neighbor_selection
        self.rng = np.random.default_rng(seed)
        self._dist_batch, self._dist_pairwise = get_metric(metric)

        # All vectors live in one contiguous float32 array that doubles when full.
        # Rows [0, count) are in use; node id == row number.
        self.vectors = np.empty((max(1, initial_capacity), dim), dtype=np.float32)
        self.count = 0
        self.levels: list[int] = []                 # levels[n] = top layer of node n
        self.graph: list[list[list[int]]] = []      # graph[n][layer] = neighbor ids
        self.entry_point: int | None = None
        self.max_level = -1

    def __len__(self) -> int:
        return self.count

    # ------------------------------------------------------------------ helpers

    def _prepare(self, x) -> np.ndarray:
        """Validate a vector and convert it to the index's float32 representation."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {x.shape}")
        if self.metric == "cosine":
            x = normalize(x).astype(np.float32)
        return x

    def _dists(self, q: np.ndarray, ids: list[int]) -> list[float]:
        """Distances from q to the given nodes, in one NumPy call."""
        return self._dist_batch(q, self.vectors[ids]).tolist()

    def random_level(self) -> int:
        """Draw a node's top layer: floor(-ln(U) * mL) with U in (0, 1].

        rng.random() is in [0, 1), so 1 - rng.random() is in (0, 1] and ln never
        sees 0. This gives P(level >= L) = exp(-L / mL) = M^-L.
        """
        return int(math.floor(-math.log(1.0 - self.rng.random()) * self.mL))

    # ------------------------------------------------------- Alg. 2: search_layer

    def search_layer(self, q: np.ndarray, entry_ids: list[int], ef: int, layer: int):
        """Best-first beam search on one layer.

        Returns up to `ef` (distance, id) pairs, nearest first.

        C (candidates) is a min-heap: pops the NEAREST unexpanded node.
        W (results)    is a max-heap: pops the FARTHEST kept node. heapq is a
                       min-heap, so W stores negated distances; -W[0][0] is the
                       distance of the worst result we are currently keeping.
        """
        visited = set(entry_ids)
        entry_d = self._dists(q, entry_ids)
        C = [(d, e) for d, e in zip(entry_d, entry_ids)]
        W = [(-d, e) for d, e in zip(entry_d, entry_ids)]
        heapq.heapify(C)
        heapq.heapify(W)
        while len(W) > ef:              # more entry points than ef: keep the best ef
            heapq.heappop(W)

        while C:
            d_c, c = heapq.heappop(C)
            # The nearest unexpanded candidate is already worse than our worst
            # result, and W is full, so expanding further cannot help: stop.
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
                        heapq.heappop(W)    # evict the farthest result

        return sorted((-neg_d, e) for neg_d, e in W)

    # ------------------------------------------- Alg. 3 and 4: neighbor selection

    def select_neighbors_simple(self, candidates, M: int):
        """Alg. 3: keep the M nearest candidates. Baseline for benchmarks only."""
        return candidates[:M]

    def select_neighbors_heuristic(self, candidates, M: int, keep_pruned: bool = True):
        """Alg. 4: pick up to M neighbors that point in diverse directions.

        `candidates` is a list of (distance_to_base, id), sorted nearest first.
        The distances to the base are already in the list, so the base vector
        itself isn't needed here.

        A candidate e is kept only if it is closer to the base than to every
        neighbor already kept. If some kept neighbor r is closer to e than the
        base is, the path base -> r -> e already reaches e, so a direct link
        would be redundant; skipping it leaves room for a link in another
        direction (e.g. toward a different cluster).
        """
        ids =[e for _, e in candidates]
        # All candidate-to-candidate distances in one matrix: pair[i][j] = dist(i, j).
        pair = self._dist_pairwise(self.vectors[ids]).tolist()

        selected: list[int] = []        # positions into `candidates`
        discarded: list[int] = []
        for i, (d, _) in enumerate(candidates):
            if len(selected) >= M:
                break
            row = pair[i]
            if all(row[j] > d for j in selected):
                selected.append(i)
            else:
                discarded.append(i)

        if keep_pruned:                 # fill remaining slots with the nearest rejects
            for i in discarded:
                if len(selected) >= M:
                    break
                selected.append(i)

        return [candidates[i] for i in selected]

    def _select(self, candidates, M: int):
        if self.neighbor_selection == "heuristic":
            return self.select_neighbors_heuristic(candidates, M)
        return self.select_neighbors_simple(candidates, M)

    # ------------------------------------------------------------- Alg. 1: insert

    def add(self, vec) -> int:
        """Insert one vector and return its id."""
        x = self._prepare(vec)

        if self.count == len(self.vectors):          # grow storage by doubling
            grown = np.empty((2 * len(self.vectors), self.dim), dtype=np.float32)
            grown[: self.count] = self.vectors[: self.count]
            self.vectors = grown

        new_id = self.count
        self.vectors[new_id] = x
        self.count += 1

        level = self.random_level()
        self.levels.append(level)
        self.graph.append([[] for _ in range(level + 1)])

        if self.entry_point is None:                 # first node: it is the whole graph
            self.entry_point = new_id
            self.max_level = level
            return new_id

        # Phase 1: above the new node's top layer, only find a good starting
        # point for the layer below (greedy, ef=1).
        ep = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            ep = [self.search_layer(x, ep, 1, layer)[0][1]]

        # Phase 2: on each layer the new node lives on, find candidates, pick
        # neighbors, and link both ways.
        for layer in range(min(level, self.max_level), -1, -1):
            W = self.search_layer(x, ep, self.ef_construction, layer)
            Mmax = self.M0 if layer == 0 else self.M
            neighbors = self._select(W, self.M)
            self.graph[new_id][layer] = [e for _, e in neighbors]

            for _, e in neighbors:
                links = self.graph[e][layer]
                links.append(new_id)                 # reverse link e -> new node
                if len(links) > Mmax:                # e is over capacity: re-select
                    d = self._dists(self.vectors[e], links)
                    cand = sorted(zip(d, links))
                    self.graph[e][layer] = [n for _, n in self._select(cand, Mmax)]

            ep = [e for _, e in W]                   # all candidates seed the next layer

        if level > self.max_level:                   # new top: future searches start here
            self.max_level = level
            self.entry_point = new_id
        return new_id

    def add_batch(self, X) -> list[int]:
        return [self.add(x) for x in np.asarray(X)]

    # ------------------------------------------------------------- Alg. 5: search

    def search(self, q, k: int, ef: int | None = None):
        """Return (ids, dists) of the approximate k nearest neighbors, nearest first.

        Fewer than k results come back only when the index has fewer than k nodes.
        """
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
        for layer in range(self.max_level, 0, -1):   # greedy descent to layer 1
            ep = [self.search_layer(x, ep, 1, layer)[0][1]]
        W = self.search_layer(x, ep, ef, 0)[:k]      # wide beam search on layer 0

        ids = np.array([e for _, e in W], dtype=np.int64)
        dists = np.array([d for d, _ in W], dtype=np.float32)
        return ids, dists

    def batch_search(self, Q, k: int, ef: int | None = None):
        """Search each row of Q. Returns (ids, dists) shaped (n_queries, k).
        If the index holds fewer than k nodes, missing slots are -1 / inf."""
        Q = np.atleast_2d(np.asarray(Q))
        ids = np.full((len(Q), k), -1, dtype=np.int64)
        dists = np.full((len(Q), k), np.inf, dtype=np.float32)
        for i, q in enumerate(Q):
            r_ids, r_dists = self.search(q, k, ef)
            ids[i, : len(r_ids)] = r_ids
            dists[i, : len(r_dists)] = r_dists
        return ids, dists
