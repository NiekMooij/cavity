from __future__ import annotations

import numpy as np
import pytest

from cavity import DegreeDistribution


def test_random_regular_excess_degree_is_c_minus_one() -> None:
    degree = DegreeDistribution.random_regular(4)
    rng = np.random.default_rng(123)
    draws = [degree.sample_excess_k(rng, k_max=8) for _ in range(32)]
    assert draws == [3] * 32


def test_poisson_excess_degree_matches_poisson_mean() -> None:
    degree = DegreeDistribution.poisson(3.0)
    rng = np.random.default_rng(123)
    draws = np.array([degree.sample_excess_k(rng, k_max=20) for _ in range(20_000)], dtype=float)
    assert draws.mean() == pytest.approx(3.0, abs=0.1)


def test_custom_excess_degree_matches_size_biased_shift() -> None:
    pmf = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    degree = DegreeDistribution.custom(pmf)
    expected = np.array([0.1, 0.3, 0.6], dtype=float)
    observed = degree.excess_pmf_array(k_max=3)
    assert np.allclose(observed[: expected.size], expected)
