import math

import numpy as np
import pytest

from hnsw import HNSWIndex


def test_mL_is_one_over_ln_M():
    assert HNSWIndex(dim=4, M=16).mL == pytest.approx(1 / math.log(16))


def test_level_distribution_matches_M_to_the_minus_L():
    # P(level >= L) = M^-L. With n = 100k draws and M = 16:
    #   level >= 1: expected 6250, std ~ 77  -> 10% tolerance is ~8 std
    #   level >= 2: expected  391, std ~ 20  -> 25% tolerance is ~5 std
    idx = HNSWIndex(dim=4, M=16, seed=123)
    levels = np.array([idx.random_level() for _ in range(100_000)])
    assert levels.min() >= 0
    frac1 = np.mean(levels >= 1)
    frac2 = np.mean(levels >= 2)
    assert frac1 == pytest.approx(1 / 16, rel=0.10)
    assert frac2 == pytest.approx(1 / 256, rel=0.25)


def test_random_level_never_takes_log_of_zero():
    # rng.random() can return exactly 0.0; we use 1 - U, so ln sees 1.0 -> level 0.
    class ZeroRNG:
        def random(self):
            return 0.0

    idx = HNSWIndex(dim=4, M=16)
    idx.rng = ZeroRNG()
    assert idx.random_level() == 0
