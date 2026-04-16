"""
Tests for the cavity.rsb (1RSB) extensions.

Covers:
- Population1RSB initialisation and shape invariants
- population.run() at β=0 (uniform resampling) and β>0
- estimate_phi: output keys, finite values, Bethe identity at β=0
- estimate_site_observables: range constraints
- compute_complexity: shape, Legendre-transform consistency
- rs_energy: analytic value for a known case
- End-to-end pipeline: population → estimate_phi → compute_complexity
"""
import numpy as np
import pytest

from cavity import (
    DegreeDistribution,
    JDistribution,
    PopulationConfig,
    Population1RSB,
    estimate_phi,
    estimate_site_observables,
    compute_complexity,
    rs_energy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
    """Small configuration for fast tests."""
    return PopulationConfig(G=51, M=100, T=0, seed=7, verbose=False, recenter=True, dtype=float)


@pytest.fixture(scope="module")
def rs_population(small_cfg):
    """A population in the RS phase (mu < mu_c = 1/(2*sqrt(3)) ≈ 0.289)."""
    degree = DegreeDistribution.random_regular(4)
    couplings = JDistribution(kind="uniform_pos", uniform_low=0.20, uniform_high=0.20 + 1e-9)
    return Population1RSB(degree, couplings, small_cfg, burn_in=100, resample_batch=16)


@pytest.fixture(scope="module")
def rsb_population(small_cfg):
    """A population in the RSB phase (mu > mu_c ≈ 0.289)."""
    degree = DegreeDistribution.random_regular(4)
    couplings = JDistribution(kind="uniform_pos", uniform_low=0.60, uniform_high=0.60 + 1e-9)
    return Population1RSB(degree, couplings, small_cfg, burn_in=100, resample_batch=16)


# ---------------------------------------------------------------------------
# Population1RSB initialisation
# ---------------------------------------------------------------------------

class TestPopulation1RSBInit:
    def test_h0_shape(self, rs_population, small_cfg):
        assert rs_population.h0.shape == (small_cfg.M, small_cfg.G)

    def test_h0_dtype(self, rs_population):
        assert rs_population.h0.dtype == np.float64

    def test_grid_shape(self, rs_population, small_cfg):
        assert rs_population.grid.shape == (small_cfg.G,)
        assert rs_population.grid[0] == pytest.approx(0.0)
        assert rs_population.grid[-1] == pytest.approx(1.0)

    def test_degree_stored(self, rs_population):
        assert rs_population.c == 4
        assert rs_population.degree.kind == "random_regular"

    def test_phi_is_onsite_potential(self, rs_population):
        # phi(n) = 0.5*n^2 - n
        n = rs_population.grid
        expected = 0.5 * n ** 2 - n
        np.testing.assert_allclose(rs_population.phi, expected)

    def test_recenter_nonneg(self, rs_population):
        # After recentering, every field should have min >= 0 (up to float64 eps)
        mins = rs_population.h0.min(axis=1)
        assert np.all(mins >= -1e-10)

    def test_c_minus_one_js(self, rs_population):
        # _js has exactly c-1 entries
        assert rs_population._js.shape == (rs_population.c - 1,)
        assert rs_population._js.dtype == np.float64
        # Couplings are negative by convention
        assert np.all(rs_population._js <= 0.0)


# ---------------------------------------------------------------------------
# population.run()
# ---------------------------------------------------------------------------

class TestPopulationRun:
    def test_run_beta0_changes_pool(self, small_cfg):
        """At β=0 resampling is uniform; pool should change but shape is preserved."""
        degree = DegreeDistribution.random_regular(4)
        couplings = JDistribution(kind="uniform_pos", uniform_low=0.3, uniform_high=0.3 + 1e-9)
        pop = Population1RSB(degree, couplings, small_cfg, burn_in=50, resample_batch=16)
        h0_before = pop.h0.copy()
        pop.run(beta=0.0, n_steps=20)
        assert pop.h0.shape == h0_before.shape
        # Pool should have been modified
        assert not np.array_equal(pop.h0, h0_before)

    def test_run_high_beta(self, small_cfg):
        """At high β population dynamics should run without error."""
        degree = DegreeDistribution.random_regular(4)
        couplings = JDistribution(kind="uniform_pos", uniform_low=0.6, uniform_high=0.6 + 1e-9)
        pop = Population1RSB(degree, couplings, small_cfg, burn_in=50, resample_batch=16)
        pop.run(beta=5.0, n_steps=30)
        assert pop.h0.shape == (small_cfg.M, small_cfg.G)
        assert np.all(np.isfinite(pop.h0))

    def test_run_preserves_dtype(self, rsb_population):
        rsb_population.run(beta=2.0, n_steps=5)
        assert rsb_population.h0.dtype == np.float64

    def test_run_zero_steps_is_noop(self, small_cfg):
        degree = DegreeDistribution.random_regular(4)
        couplings = JDistribution(kind="uniform_pos", uniform_low=0.3, uniform_high=0.3 + 1e-9)
        pop = Population1RSB(degree, couplings, small_cfg, burn_in=50, resample_batch=16)
        h0_before = pop.h0.copy()
        pop.run(beta=2.0, n_steps=0)
        np.testing.assert_array_equal(pop.h0, h0_before)


# ---------------------------------------------------------------------------
# estimate_phi
# ---------------------------------------------------------------------------

class TestEstimatePhi:
    def test_output_keys(self, rs_population):
        result = estimate_phi(rs_population, beta=1.0, n_mc=50, seed=0)
        assert set(result.keys()) == {
            "phi", "delta_phi_site", "delta_phi_edge", "f_site_mean", "f_edge_mean"
        }

    def test_all_finite(self, rs_population):
        result = estimate_phi(rs_population, beta=2.0, n_mc=50, seed=1)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_bethe_identity_at_beta0(self, rs_population):
        """At β=0: Φ = <f_site> - (c/2)*<f_edge>, i.e. the Lyapunov density."""
        result = estimate_phi(rs_population, beta=0.0, n_mc=200, seed=42)
        c = rs_population.c
        expected_phi = result["f_site_mean"] - (c / 2.0) * result["f_edge_mean"]
        assert result["phi"] == pytest.approx(expected_phi, rel=1e-6)

    def test_deterministic_with_seed(self, rs_population):
        r1 = estimate_phi(rs_population, beta=1.5, n_mc=50, seed=99)
        r2 = estimate_phi(rs_population, beta=1.5, n_mc=50, seed=99)
        assert r1["phi"] == pytest.approx(r2["phi"])

    def test_different_seeds_differ(self, rs_population):
        r1 = estimate_phi(rs_population, beta=2.0, n_mc=100, seed=0)
        r2 = estimate_phi(rs_population, beta=2.0, n_mc=100, seed=1)
        # With only 100 samples there will be MC noise
        assert r1["phi"] != pytest.approx(r2["phi"], abs=0.0)

    def test_bethe_decomposition(self, rs_population):
        """Φ = δΦ_site - (c/2) * δΦ_edge must hold exactly by construction."""
        result = estimate_phi(rs_population, beta=3.0, n_mc=80, seed=5)
        c = rs_population.c
        reconstructed = result["delta_phi_site"] - (c / 2.0) * result["delta_phi_edge"]
        assert result["phi"] == pytest.approx(reconstructed, rel=1e-10)


# ---------------------------------------------------------------------------
# estimate_site_observables
# ---------------------------------------------------------------------------

class TestEstimateSiteObservables:
    def test_output_keys(self, rs_population):
        obs = estimate_site_observables(rs_population, n_mc=50, seed=0)
        assert set(obs.keys()) == {"diversity", "mean_abundance"}

    def test_diversity_in_range(self, rs_population):
        obs = estimate_site_observables(rs_population, n_mc=200, seed=0)
        assert 0.0 <= obs["diversity"] <= 1.0

    def test_mean_abundance_in_range(self, rs_population):
        obs = estimate_site_observables(rs_population, n_mc=200, seed=0)
        # Grid is [0,1] so mean abundance must be in [0, 1]
        assert 0.0 <= obs["mean_abundance"] <= 1.0

    def test_rsb_phase_has_survivors(self, rsb_population):
        """In the RSB phase (mu=0.6 > mu_c≈0.289) there should be survivors."""
        rsb_population.run(beta=2.0, n_steps=10)
        obs = estimate_site_observables(rsb_population, n_mc=200, seed=0)
        assert obs["diversity"] > 0.0

    def test_deterministic_with_seed(self, rs_population):
        obs1 = estimate_site_observables(rs_population, n_mc=50, seed=7)
        obs2 = estimate_site_observables(rs_population, n_mc=50, seed=7)
        assert obs1["diversity"] == pytest.approx(obs2["diversity"])
        assert obs1["mean_abundance"] == pytest.approx(obs2["mean_abundance"])


# ---------------------------------------------------------------------------
# compute_complexity
# ---------------------------------------------------------------------------

class TestComputeComplexity:
    def test_output_keys(self):
        beta = np.array([1.0, 2.0, 3.0])
        phi = np.array([-0.10, -0.09, -0.085])
        result = compute_complexity(beta, phi)
        assert set(result.keys()) == {
            "beta", "phi", "phi_smooth", "epsilon", "sigma", "dphi_dbeta"
        }

    def test_output_shapes(self):
        n = 8
        beta = np.linspace(0.5, 4.0, n)
        phi = -0.1 + 0.005 * beta
        result = compute_complexity(beta, phi)
        for key in ("beta", "phi", "phi_smooth", "epsilon", "sigma", "dphi_dbeta"):
            assert result[key].shape == (n,), f"{key} has wrong shape"

    def test_phi_smooth_is_nondecreasing(self):
        """Isotonic smoothing must produce a non-decreasing Φ."""
        beta = np.linspace(0.5, 5.0, 10)
        # Add some noise
        rng = np.random.default_rng(0)
        phi = 0.01 * beta + 0.002 * rng.standard_normal(10)
        result = compute_complexity(beta, phi)
        diffs = np.diff(result["phi_smooth"])
        assert np.all(diffs >= -1e-12)

    def test_legendre_consistency(self):
        """Legendre transform: ε = Φ_smooth - β * dΦ/dβ must hold pointwise."""
        beta = np.linspace(1.0, 5.0, 6)
        phi = 0.005 * beta ** 2
        result = compute_complexity(beta, phi)
        reconstructed = result["phi_smooth"] - beta * result["dphi_dbeta"]
        np.testing.assert_allclose(result["epsilon"], reconstructed, rtol=1e-10)

    def test_sigma_equals_beta_sq_dphi(self):
        """Σ = -β² · dΦ/dβ must hold pointwise."""
        beta = np.linspace(0.5, 3.0, 5)
        phi = 0.01 * beta
        result = compute_complexity(beta, phi)
        expected_sigma = -(beta ** 2) * result["dphi_dbeta"]
        np.testing.assert_allclose(result["sigma"], expected_sigma, rtol=1e-10)

    def test_single_point(self):
        result = compute_complexity(np.array([2.0]), np.array([-0.05]))
        assert result["sigma"][0] == pytest.approx(0.0)
        assert result["epsilon"][0] == pytest.approx(-0.05)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            compute_complexity(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# rs_energy
# ---------------------------------------------------------------------------

class TestRsEnergy:
    def test_rs_phase_finite(self):
        """In the RS phase mu < mu_c = 1/(2*sqrt(c-1)), rs_energy returns a finite value."""
        val = rs_energy(c=4, mu=0.2)
        assert np.isfinite(val)

    def test_returns_float(self):
        assert isinstance(rs_energy(c=4, mu=0.1), float)

    def test_increases_with_mu(self):
        """A larger coupling raises the energy landscape (less negative minimum)."""
        e1 = rs_energy(c=4, mu=0.1)
        e2 = rs_energy(c=4, mu=0.2)
        assert e2 > e1

    def test_known_value_c4_mu0(self):
        """At mu=0, delta=1 so the formula gives a=0.5, b=1, d=3; minimum at n=1 is 2.5."""
        # E(n) = 0.5*n^2 - n + 3  →  min at n=1: 0.5 - 1 + 3 = 2.5
        val = rs_energy(c=4, mu=0.0)
        assert val == pytest.approx(2.5, abs=1e-4)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, small_cfg):
        """Smoke test: warm-up → beta sweep → complexity curve."""
        degree = DegreeDistribution.random_regular(4)
        couplings = JDistribution(kind="uniform_pos", uniform_low=0.6, uniform_high=0.6 + 1e-9)
        pop = Population1RSB(degree, couplings, small_cfg, burn_in=80, resample_batch=16)

        beta_values = np.array([1.0, 2.0, 3.0])
        phi_values = np.empty(len(beta_values))

        for i, beta in enumerate(beta_values):
            pop.run(beta, n_steps=20)
            result = estimate_phi(pop, beta=beta, n_mc=80, seed=i)
            phi_values[i] = result["phi"]

        assert np.all(np.isfinite(phi_values))

        cx = compute_complexity(beta_values, phi_values)
        assert np.all(np.isfinite(cx["epsilon"]))
        assert np.all(np.isfinite(cx["sigma"]))

    def test_rsb_accessible_from_top_level(self):
        """All 1RSB symbols must be importable directly from cavity."""
        import cavity
        for name in ("Population1RSB", "estimate_phi", "estimate_site_observables",
                     "compute_complexity", "rs_energy"):
            assert hasattr(cavity, name), f"cavity.{name} not found"

    def test_rsb_accessible_from_subpackage(self):
        """All 1RSB symbols must also be importable from cavity.rsb."""
        import cavity.rsb
        for name in ("Population1RSB", "estimate_phi", "estimate_site_observables",
                     "compute_complexity", "rs_energy"):
            assert hasattr(cavity.rsb, name), f"cavity.rsb.{name} not found"
