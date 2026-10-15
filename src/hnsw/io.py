"""Save and load an HNSWIndex with np.savez (no pickle).

The graph is a ragged list-of-lists-of-lists, which NumPy can't store directly,
so it is flattened in a fixed order: node 0 layer 0, node 0 layer 1, ...,
node 1 layer 0, ... For each (node, layer) slot we store its degree, and all
neighbor ids are concatenated into one array. Since levels[n] tells us how many
layers node n has, the degrees are enough to cut the array back apart.
"""

import json

import numpy as np

from .index import HNSWIndex

FORMAT_VERSION = 1


def save(index: HNSWIndex, path) -> None:
    degrees = [len(nbrs) for node in index.graph for nbrs in node]
    neighbors = [n for node in index.graph for nbrs in node for n in nbrs]
    meta = {
        "format_version": FORMAT_VERSION,
        "dim": index.dim,
        "metric": index.metric,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "neighbor_selection": index.neighbor_selection,
        "entry_point": index.entry_point,
        "max_level": index.max_level,
        # Saving the RNG state means adding more vectors after load gives
        # exactly the same graph as if the index had never been saved.
        "rng_state": index.rng.bit_generator.state,
    }
    np.savez(
        path,
        vectors=index.vectors[: index.count],
        levels=np.array(index.levels, dtype=np.int32),
        degrees=np.array(degrees, dtype=np.int32),
        neighbors=np.array(neighbors, dtype=np.int64),
        meta=np.array(json.dumps(meta)),
    )


def load(path) -> HNSWIndex:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {meta['format_version']}")
        vectors = f["vectors"]
        levels = f["levels"].tolist()
        degrees = f["degrees"].tolist()
        neighbors = f["neighbors"].tolist()

    index = HNSWIndex(
        dim=meta["dim"],
        metric=meta["metric"],
        M=meta["M"],
        ef_construction=meta["ef_construction"],
        neighbor_selection=meta["neighbor_selection"],
        initial_capacity=max(1, len(vectors)),
    )
    index.rng.bit_generator.state = meta["rng_state"]
    index.vectors[: len(vectors)] = vectors
    index.count = len(vectors)
    index.levels = levels
    index.entry_point = meta["entry_point"]
    index.max_level = meta["max_level"]

    graph, slot, pos = [], 0, 0
    for level in levels:
        node = []
        for _ in range(level + 1):
            node.append(neighbors[pos: pos + degrees[slot]])
            pos += degrees[slot]
            slot += 1
        graph.append(node)
    index.graph = graph
    return index
