from __future__ import annotations

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .distributions import DegreeDistribution, JDistribution
from .config import PopulationConfig
from .population import CavityPopulation
from .kernels import onsite_phi, _update_one_nb


def cavity_simulation(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    burn_in: int = 5000,
) -> np.ndarray:
    """
    Run a single-population cavity simulation (no replicas, no trace sampling).

    Returns
    -------
    population : (M, G) array
        Final population after burn-in + cfg.T steps.
    """
    # --- config snapshot
    M        = int(cfg.M)
    k_max    = int(cfg.k_max)
    recenter = bool(cfg.recenter)
    dtype    = cfg.dtype
    seed     = int(cfg.seed)

    # progress policy
    def _want_bars() -> bool:
        if not cfg.verbose:
            return False
        else:
            return True

    show_bars = _want_bars()
    def _pbar(total: int, desc: str):
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False, disable=not show_bars)

    # --- helpers (duplicated locally to keep function self-contained)
    def _step(rng, pop, phi, x, y, scratch, hbuf, recenter_flag):
        deg_exc = degree.sample_k(rng, size_bias=False, k_max=k_max)
        if deg_exc > 0:
            idx = rng.integers(0, M, size=deg_exc, dtype=np.int64)
            Js = couplings.sample(rng, size=deg_exc).astype(dtype, copy=False)
        else:
            idx = np.empty(0, dtype=np.int64)
            Js = np.empty(0, dtype=dtype)
        _update_one_nb(pop.population, phi, x, y, idx, Js, scratch, hbuf, recenter_flag)
        repl = int(rng.integers(0, M))
        pop.population[repl] = hbuf
        return hbuf

    # --- initialize
    pop = CavityPopulation(degree, couplings, cfg)
    rng = np.random.default_rng(seed)
    x = pop.n.astype(dtype, copy=False)
    y = pop.grid.astype(dtype, copy=False)
    phi = onsite_phi(x).astype(dtype, copy=False)
    scratch = np.empty_like(phi)
    hbuf = np.empty_like(phi)
    # initialize pool to onsite potential as in discrete method
    pop.population = np.tile(phi[None, :], (M, 1)).astype(dtype, copy=False)
    if recenter:
        pop.population -= pop.population.min(axis=1, keepdims=True).astype(dtype, copy=False)

    # --- burn-in
    burn_in = max(0, int(burn_in))
    bar = _pbar(burn_in, "warm-up")
    for _ in range(burn_in):
        _step(rng, pop, phi, x, y, scratch, hbuf, recenter)
        bar.update(1)
    bar.close()

    # --- main run
    T = int(cfg.T)
    bar = _pbar(T, "run")
    for _ in range(T):
        _step(rng, pop, phi, x, y, scratch, hbuf, recenter)
        bar.update(1)
    bar.close()

    return pop.population.astype(dtype, copy=False)


def warm_up_pool(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    burn_in: int = 5000,
) -> np.ndarray:
    """
    Create a warmed-up population pool using the discrete update method.

    Returns
    -------
    population : (M, G) array
        Warmed-up population after burn-in updates.
    """
    M = int(cfg.M)
    k_max = int(cfg.k_max)
    recenter = bool(cfg.recenter)
    dtype = cfg.dtype
    seed = int(cfg.seed)

    # progress policy
    def _want_bars() -> bool:
        if not cfg.verbose:
            return False
        else:
            return True

    show_bars = _want_bars()
    def _pbar(total: int, desc: str):
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False, disable=not show_bars)

    pop = CavityPopulation(degree, couplings, cfg)
    rng = np.random.default_rng(seed)
    x = pop.n.astype(dtype, copy=False)
    y = pop.grid.astype(dtype, copy=False)
    phi = onsite_phi(x).astype(dtype, copy=False)
    scratch = np.empty_like(phi)
    hbuf = np.empty_like(phi)
    pop.population = np.tile(phi[None, :], (M, 1)).astype(dtype, copy=False)
    if recenter:
        pop.population -= pop.population.min(axis=1, keepdims=True).astype(dtype, copy=False)

    burn_in = max(0, int(burn_in))
    bar = _pbar(burn_in, "warm-up")
    for _ in range(burn_in):
        d = degree.sample_k(rng, size_bias=False, k_max=k_max)
        if d > 0:
            idx = rng.integers(0, M, size=d, dtype=np.int64)
            Js = couplings.sample(rng, size=d).astype(dtype, copy=False)
        else:
            idx = np.empty(0, dtype=np.int64)
            Js = np.empty(0, dtype=dtype)
        _update_one_nb(pop.population, phi, x, y, idx, Js, scratch, hbuf, recenter)
        repl = int(rng.integers(0, M))
        pop.population[repl] = hbuf
        bar.update(1)
    bar.close()
    return pop.population
