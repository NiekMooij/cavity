"""
1RSB population dynamics.

Population1RSB is the 1RSB counterpart of CavityPopulation.  It manages a
pool of cavity fields h_0(n) and advances them at a given inverse temperature
β via beta-weighted importance resampling — the core of the 1RSB population
dynamics algorithm.

Initialisation
--------------
The pool is seeded from a replica-symmetric warm-up (via cavity.simulation.
warm_up_pool) so that the initial fields already reflect the RS fixed point.
The 1RSB dynamics then deforms this distribution toward the β-weighted
stationary measure.

Usage
-----
    from cavity import DegreeDistribution, JDistribution, PopulationConfig
    from cavity import Population1RSB, estimate_phi, compute_complexity

    degree = DegreeDistribution.random_regular(4)
    couplings = JDistribution(kind="uniform_pos", uniform_low=0.6, uniform_high=0.6 + 1e-9)
    cfg = PopulationConfig(G=801, M=8000, T=0, seed=42, verbose=True, recenter=True,
                           dtype=float)

    pop = Population1RSB(degree, couplings, cfg, burn_in=10_000)
    pop.run(beta=2.0, n_steps=40_000)
    result = estimate_phi(pop, beta=2.0, n_mc=12_000)
"""
from __future__ import annotations

import numpy as np

from ..distributions import DegreeDistribution, JDistribution
from ..config import PopulationConfig
from ..kernels import onsite_phi, _accum_min_convs
from .kernels import (
    NUMBA_AVAILABLE,
    build_candidates_numba,
    resample_indices_numba,
)


class Population1RSB:
    """
    Pool of 1RSB cavity fields h_0(n) for a sparse random graph.

    Manages a population of M fields on a grid of G abundance values.
    Each field represents a cavity message: the effective energy landscape
    seen by a node when one of its neighbours is removed.

    The population is evolved at inverse temperature β via a batch update:
    for each replacement step, a set of candidate fields is constructed from
    randomly chosen parents, and a candidate is selected with probability
    proportional to exp(-β · f_cav), where f_cav = min_n h_new(n) is the
    cavity free energy of the candidate.

    Parameters
    ----------
    degree : DegreeDistribution
        *Full* degree distribution (e.g. random_regular(4) for c=4).
        The cavity degree (c-1 for regular graphs) is derived internally.
    couplings : JDistribution
        Coupling strength distribution.  Returns negative values by
        convention (competitive interactions).
    cfg : PopulationConfig
        Grid and population configuration.  cfg.T is ignored; use run().
    burn_in : int
        Number of RS warm-up steps (passed to warm_up_pool).
    resample_batch : int
        Number of candidates generated per replacement step.  Larger values
        give better importance-resampling accuracy at the cost of memory.
    """

    def __init__(
        self,
        degree: DegreeDistribution,
        couplings: JDistribution,
        cfg: PopulationConfig,
        *,
        burn_in: int = 5000,
        resample_batch: int = 128,
    ):
        # Lazy import to break the population ↔ simulation circular dependency.
        from ..simulation import warm_up_pool

        self.degree = degree
        self.couplings = couplings
        self.cfg = cfg
        self.c = int(degree.c)  # full degree

        self.grid = np.linspace(0.0, 1.0, cfg.G, dtype=np.float64)
        self.phi = onsite_phi(self.grid).astype(np.float64)

        # Cavity degree: k → k-1 for regular graphs; unchanged for Poisson.
        if degree.kind == "random_regular":
            degree_cav = DegreeDistribution.random_regular(self.c - 1)
        else:
            degree_cav = degree

        # RS warm-up initialises the pool at the replica-symmetric fixed point.
        cfg_warmup = PopulationConfig(
            G=cfg.G,
            M=cfg.M,
            T=0,
            seed=cfg.seed,
            verbose=cfg.verbose,
            recenter=True,
            dtype=cfg.dtype,
        )
        rs_pool = warm_up_pool(degree_cav, couplings, cfg_warmup, burn_in=burn_in)

        self.h0: np.ndarray = rs_pool.astype(np.float64)
        self.m: int = cfg.M
        self.g: int = cfg.G
        self._rng = np.random.default_rng(cfg.seed + 100)
        self.resample_batch: int = max(1, min(cfg.M, int(resample_batch)))

        # Pre-sample cavity couplings (c-1 values).
        # For narrow distributions (e.g. uniform_pos with tiny width) this is
        # equivalent to using the mean coupling for every step.
        self._js: np.ndarray = couplings.sample(
            np.random.default_rng(cfg.seed + 200), size=self.c - 1
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Internal update
    # ------------------------------------------------------------------

    def _build_candidates_python(
        self,
        parent_idx: np.ndarray,
        out_candidates: np.ndarray,
        out_shifts: np.ndarray,
    ) -> None:
        """Python fallback for build_candidates_numba (no numba required)."""
        for b in range(parent_idx.shape[0]):
            accum = self.phi.copy()
            _accum_min_convs(
                self.h0[parent_idx[b]],
                self._js,
                self.grid,
                self.grid,
                accum,
            )
            f_cav = float(np.min(accum))
            out_candidates[b] = accum - f_cav
            out_shifts[b] = f_cav

    def _replace_batch(self, beta: float, batch_size: int) -> None:
        """
        Generate *batch_size* candidate fields, importance-resample by
        exp(-β · f_cav), and write the selected candidates back into the pool.
        """
        rng = self._rng
        parent_idx = rng.integers(
            0, self.m, size=(batch_size, self.c - 1), dtype=np.int64
        )
        candidates = np.empty((batch_size, self.g), dtype=np.float64)
        shifts = np.empty(batch_size, dtype=np.float64)

        if NUMBA_AVAILABLE:
            build_candidates_numba(
                self.h0,
                self.phi,
                self.grid,
                self._js,
                parent_idx,
                candidates,
                shifts,
            )
        else:
            self._build_candidates_python(parent_idx, candidates, shifts)

        # Slots to overwrite in the pool (sampled independently of candidates).
        slots = rng.integers(0, self.m, size=batch_size, dtype=np.int64)

        if beta == 0.0:
            picked = rng.integers(0, batch_size, size=batch_size, dtype=np.int64)
        elif NUMBA_AVAILABLE:
            picked = np.empty(batch_size, dtype=np.int64)
            resample_indices_numba(
                shifts,
                float(beta),
                rng.random(batch_size, dtype=np.float64),
                picked,
            )
        else:
            log_w = -beta * shifts
            log_w -= float(np.max(log_w))
            weights = np.exp(log_w)
            weights /= float(np.sum(weights))
            picked = rng.choice(batch_size, size=batch_size, replace=True, p=weights)

        self.h0[slots] = candidates[picked]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, beta: float, n_steps: int) -> None:
        """
        Advance the population for *n_steps* replacement steps at inverse
        temperature *beta*.

        Each step replaces resample_batch entries, so the effective number of
        individual field updates is n_steps * resample_batch.

        Parameters
        ----------
        beta : float
            Parisi inverse temperature.
        n_steps : int
            Number of batch replacement steps.
        """
        remaining = int(n_steps)
        while remaining > 0:
            batch = min(self.resample_batch, remaining)
            self._replace_batch(beta, batch)
            remaining -= batch
