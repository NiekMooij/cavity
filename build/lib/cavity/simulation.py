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


def lv_integration(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    dt: float = 0.01,
    steps: int = 20000,
    seed_offset: int = 7,
    return_trajectory: bool = False,
) -> np.ndarray:
    """
    Integrate a Lotka-Volterra system on an Erdos-Renyi interaction graph.

    Graph model
    -----------
    - Undirected simple graph G(M, p) with p = c_mean / (M - 1)
    - No self-loops or multiedges
    - Symmetric edge couplings sampled from `couplings`
    - Fixed self-interaction A_ii = -1

    Dynamics
    --------
    Forward Euler integration of:
        dN_i/dt = N_i * (1 + sum_j A_ij N_j)

    Returns
    -------
    N : (M,) float64 array
        Final abundance vector after integration when `return_trajectory=False`.
    traj : (steps + 1, M) float64 array
        Full trajectory (including initial condition) when `return_trajectory=True`.
    """
    rng = np.random.default_rng(int(cfg.seed) + int(seed_offset))
    M = int(cfg.M)
    c_mean = float(degree.mean(int(cfg.k_max)))
    if M <= 1:
        p_edge = 0.0
    else:
        p_edge = min(max(c_mean / float(M - 1), 0.0), 1.0)

    A = np.zeros((M, M), dtype=np.float64)
    tri_i, tri_j = np.triu_indices(M, k=1)
    edge_mask = rng.random(size=tri_i.size) < p_edge
    n_edges = int(edge_mask.sum())
    if n_edges > 0:
        w = couplings.sample(rng, size=n_edges).astype(np.float64, copy=False)
        ii = tri_i[edge_mask]
        jj = tri_j[edge_mask]
        A[ii, jj] = w
        A[jj, ii] = w
    np.fill_diagonal(A, -1.0)

    r = np.ones(M, dtype=np.float64)
    N = rng.uniform(0.1, 1.0, size=M).astype(np.float64)
    if return_trajectory:
        traj = np.empty((int(steps) + 1, M), dtype=np.float64)
        traj[0] = N

    for t in tqdm(range(int(steps)), desc="LV integration", disable=not bool(cfg.verbose)):
        dN = N * (r + A @ N)
        N = N + float(dt) * dN
        np.clip(N, 0.0, np.inf, out=N)
        if return_trajectory:
            traj[t + 1] = N

    if return_trajectory:
        return traj
    return N
