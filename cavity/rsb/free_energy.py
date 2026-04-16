"""
Free-energy estimation for the 1RSB cavity method.

Provides MC estimators for the Bethe free entropy Φ(β) and site observables
(diversity, mean abundance) from a converged Population1RSB.

Physics
-------
For a c-regular graph with coupling μ, the generalised free energy per site is:

    Φ(β) = δΦ_site(β) - (c/2) · δΦ_edge(β)

where

    δΦ_site = -(1/β) log E[exp(-β f_site)]
    δΦ_edge = -(1/β) log E[exp(-β f_edge)]

and the expectations are MC averages over randomly drawn cavity fields and
couplings from the current 1RSB population pool.
"""
from __future__ import annotations

import numpy as np

from ..kernels import onsite_phi, _accum_min_convs
from .kernels import (
    NUMBA_AVAILABLE,
    batch_f_site_numba,
    batch_site_argmin_numba,
    batch_f_edge_numba,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _minus_inv_beta_logmeanexp_neg_beta(beta: float, values: np.ndarray) -> float:
    """Compute -(1/β) log mean exp(-β v) stably."""
    v = np.asarray(values, dtype=np.float64)
    max_lw = float(np.max(-beta * v))
    log_sum = max_lw + float(np.log(np.mean(np.exp(-beta * v - max_lw))))
    return -log_sum / beta


def _batch_f_site(population, rng: np.random.Generator, n_mc: int) -> np.ndarray:
    """
    Draw n_mc site free-energy samples from the population pool.

    For each sample: draw c cavity fields uniformly from the pool and c
    couplings from population.couplings, then minimise the site Hamiltonian.
    """
    h0 = population.h0
    phi = population.phi
    grid = population.grid
    c = int(population.degree.c)
    m = h0.shape[0]
    js = population.couplings.sample(rng, size=c).astype(grid.dtype)

    if NUMBA_AVAILABLE:
        idx = rng.integers(0, m, size=(n_mc, c), dtype=np.int64)
        return batch_f_site_numba(h0, phi, grid, js, idx)

    # Python fallback
    out = np.empty(n_mc, dtype=np.float64)
    for s in range(n_mc):
        idx = rng.integers(0, m, size=c)
        accum = phi.copy()
        _accum_min_convs(h0[idx], js, grid, grid, accum)
        out[s] = float(np.min(accum))
    return out


def _batch_f_edge(population, rng: np.random.Generator, n_mc: int) -> np.ndarray:
    """
    Draw n_mc edge free-energy samples from the population pool.

    For each sample: draw two cavity fields and one coupling, then minimise
    the two-site edge Hamiltonian.
    """
    h0 = population.h0
    grid = population.grid
    m = h0.shape[0]
    js = population.couplings.sample(rng, size=1).astype(grid.dtype)

    if NUMBA_AVAILABLE:
        idx = rng.integers(0, m, size=(n_mc, 2), dtype=np.int64)
        return batch_f_edge_numba(h0, grid, js, idx)

    # Python fallback
    out = np.empty(n_mc, dtype=np.float64)
    for s in range(n_mc):
        i0, i1 = rng.integers(0, m, size=2)
        accum = h0[i0].copy()
        _accum_min_convs(
            h0[i1][np.newaxis, :],
            js,
            grid,
            grid,
            accum,
        )
        out[s] = float(np.min(accum))
    return out


def _batch_site_argmin(population, rng: np.random.Generator, n_mc: int) -> np.ndarray:
    """
    Like _batch_f_site but returns the argmin (abundance index) for each
    sample, used to compute the equilibrium abundance distribution.
    """
    h0 = population.h0
    phi = population.phi
    grid = population.grid
    c = int(population.degree.c)
    m = h0.shape[0]
    js = population.couplings.sample(rng, size=c).astype(grid.dtype)

    if NUMBA_AVAILABLE:
        idx = rng.integers(0, m, size=(n_mc, c), dtype=np.int64)
        return batch_site_argmin_numba(h0, phi, grid, js, idx)

    # Python fallback
    out = np.empty(n_mc, dtype=np.int64)
    for s in range(n_mc):
        idx = rng.integers(0, m, size=c)
        accum = phi.copy()
        _accum_min_convs(h0[idx], js, grid, grid, accum)
        out[s] = int(np.argmin(accum))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_phi(
    population,
    beta: float,
    *,
    n_mc: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Estimate the 1RSB free entropy Φ(β) via Monte Carlo.

    Uses the Bethe decomposition:

        Φ(β) = δΦ_site(β) − (c/2) · δΦ_edge(β)

    where each term is estimated by drawing cavity fields and couplings
    from the current Population1RSB pool.

    Parameters
    ----------
    population : Population1RSB
        A converged 1RSB population (after sufficient population.run() steps).
    beta : float
        Inverse temperature (Parisi parameter).
    n_mc : int
        Number of MC samples per term.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        phi, delta_phi_site, delta_phi_edge, f_site_mean, f_edge_mean
    """
    rng = np.random.default_rng(seed)
    c = int(population.degree.c)

    f_site = _batch_f_site(population, rng, n_mc)
    if beta == 0.0:
        delta_phi_site = float(np.mean(f_site))
    else:
        delta_phi_site = _minus_inv_beta_logmeanexp_neg_beta(beta, f_site)

    f_edge = _batch_f_edge(population, rng, n_mc)
    if beta == 0.0:
        delta_phi_edge = float(np.mean(f_edge))
    else:
        delta_phi_edge = _minus_inv_beta_logmeanexp_neg_beta(beta, f_edge)

    phi = delta_phi_site - (c / 2.0) * delta_phi_edge
    return {
        "phi": phi,
        "delta_phi_site": delta_phi_site,
        "delta_phi_edge": delta_phi_edge,
        "f_site_mean": float(np.mean(f_site)),
        "f_edge_mean": float(np.mean(f_edge)),
    }


def estimate_site_observables(
    population,
    *,
    n_mc: int = 5000,
    seed: int = 0,
    survival_eps: float = 1e-12,
) -> dict[str, float]:
    """
    Estimate site observables (diversity, mean abundance) from the 1RSB pool.

    For each MC sample the equilibrium abundance n* = argmin h_site is found;
    diversity is the fraction of samples with n* > survival_eps.

    Parameters
    ----------
    population : Population1RSB
    n_mc : int
    seed : int
    survival_eps : float
        Threshold below which a species is considered extinct.

    Returns
    -------
    dict with keys: diversity, mean_abundance
    """
    rng = np.random.default_rng(seed)
    argmin_idx = _batch_site_argmin(population, rng, n_mc)
    n_star = population.grid[argmin_idx]
    return {
        "diversity": float(np.mean(n_star > survival_eps)),
        "mean_abundance": float(np.mean(n_star)),
    }
