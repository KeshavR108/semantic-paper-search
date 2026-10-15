import numpy as np
import pytest

from hnsw import HNSWIndex, load, save


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_save_load_round_trip(tmp_path, metric):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 16)).astype(np.float32)
    Q = rng.standard_normal((50, 16)).astype(np.float32)
    idx = HNSWIndex(16, metric=metric, M=8, ef_construction=64, seed=7)
    idx.add_batch(X)

    path = tmp_path / "index.npz"
    save(idx, path)
    loaded = load(path)

    assert loaded.graph == idx.graph
    assert loaded.levels == idx.levels
    assert (loaded.entry_point, loaded.max_level) == (idx.entry_point, idx.max_level)

    a_ids, a_d = idx.batch_search(Q, k=10, ef=50)
    b_ids, b_d = loaded.batch_search(Q, k=10, ef=50)
    np.testing.assert_array_equal(a_ids, b_ids)
    np.testing.assert_array_equal(a_d, b_d)

    # The RNG state was saved too, so adding more vectors gives identical graphs.
    more = rng.standard_normal((100, 16)).astype(np.float32)
    idx.add_batch(more)
    loaded.add_batch(more)
    assert loaded.graph == idx.graph


def test_save_load_empty_and_single(tmp_path):
    for n in (0, 1):
        idx = HNSWIndex(4)
        if n:
            idx.add(np.ones(4))
        save(idx, tmp_path / f"i{n}.npz")
        loaded = load(tmp_path / f"i{n}.npz")
        assert len(loaded) == n
        assert loaded.entry_point == idx.entry_point
