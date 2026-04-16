from __future__ import annotations

from typing import Tuple
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
from .simulation import warm_up_pool


def replica_h_traces(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    burn_in: int = 5000,
    init_jitter: float = 1e-6,
    n_trace_samples: int = 64,
    sample_window: float = 0.10,
    sync: str | set[str] = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic tail sampling.
    Run the simulation for cfg.iterations steps. Define the tail as the last
    `sample_window` fraction of the run (e.g., sample_window=0.1 => last 10%).
    Collect `n_trace_samples` samples at deterministically and evenly spaced
    timesteps within that tail, with the FIRST sample exactly at the **start**
    of the tail (e.g., 0.9 * iterations), and the LAST sample at the final step.

    Returns
    -------
    t_series : (B,) int32
        The selected iteration indices (1-based).
    H1, H2 : (B, G)
        Energy traces at those iterations for the two replicas.
    """
    # --- guards
    if not (0.0 < sample_window <= 1.0):
        raise ValueError("sample_window must be in (0, 1].")
    if n_trace_samples <= 0:
        raise ValueError("n_trace_samples must be positive.")

    # --- parse sync flags
    if isinstance(sync, str):
        if sync == "none":
            sync_set: set[str] = set()
        elif sync == "disorder":
            sync_set = {"k", "parents", "js"}
        elif sync == "all":
            sync_set = {"k", "parents", "js", "replace"}
        else:
            sync_set = {sync}
    else:
        sync_set = set(sync)

    # --- config snapshot
    M        = int(cfg.M)
    G        = int(cfg.G)
    T        = int(cfg.T)
    k_max    = int(cfg.k_max)
    recenter = bool(cfg.recenter)
    dtype    = cfg.dtype
    seed     = int(cfg.seed)

    # RNG for shared randomness when requested
    rng_disorder = np.random.default_rng(seed + 99_999)

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
    def _shared_step(rng1, rng2, pop1, pop2, phi, x, y, scratch1, scratch2, hbuf1, hbuf2, recenter_flag):
        # choose how to draw degree (no size bias, matches discrete method)
        if "k" in sync_set:
            d1 = degree.sample_k(rng_disorder, size_bias=False, k_max=k_max)
            d2 = d1
        else:
            d1 = degree.sample_k(rng1, size_bias=False, k_max=k_max)
            d2 = degree.sample_k(rng2, size_bias=False, k_max=k_max)

        # replica 1
        if d1 > 0:
            if "js" in sync_set:
                js1 = couplings.sample(rng_disorder, size=d1).astype(phi.dtype, copy=False)
            else:
                js1 = couplings.sample(rng1, size=d1).astype(phi.dtype, copy=False)
            if "parents" in sync_set:
                idx1 = rng_disorder.integers(0, M, size=d1, dtype=np.int64)
            else:
                idx1 = rng1.integers(0, M, size=d1, dtype=np.int64)
        else:
            js1 = np.empty(0, dtype=phi.dtype)
            idx1 = np.empty(0, dtype=np.int64)
        _update_one_nb(pop1.population, phi, x, y, idx1, js1, scratch1, hbuf1, recenter_flag)
        h1 = hbuf1

        # replica 2
        if d2 > 0:
            if "js" in sync_set:
                if "k" in sync_set:
                    js2 = js1
                else:
                    js2 = couplings.sample(rng_disorder, size=d2).astype(phi.dtype, copy=False)
            else:
                js2 = couplings.sample(rng2, size=d2).astype(phi.dtype, copy=False)

            if "parents" in sync_set:
                if "k" in sync_set:
                    idx2 = idx1
                else:
                    idx2 = rng_disorder.integers(0, M, size=d2, dtype=np.int64)
            else:
                idx2 = rng2.integers(0, M, size=d2, dtype=np.int64)
        else:
            js2 = np.empty(0, dtype=phi.dtype)
            idx2 = np.empty(0, dtype=np.int64)
        _update_one_nb(pop2.population, phi, x, y, idx2, js2, scratch2, hbuf2, recenter_flag)
        h2 = hbuf2

        # replacement indices
        if "replace" in sync_set:
            repl1 = int(rng_disorder.integers(0, M))
            repl2 = repl1
        else:
            repl1 = int(rng1.integers(0, M))
            repl2 = int(rng2.integers(0, M))

        pop1.population[repl1] = h1
        pop2.population[repl2] = h2
        return h1, h2

    # --- warm-up (use cfg.burn_in as a sensible warmup length)
    pool = warm_up_pool(degree, couplings, cfg, burn_in=burn_in)

    # --- replicas
    pop1 = CavityPopulation(degree, couplings, cfg)
    pop2 = CavityPopulation(degree, couplings, cfg)
    rng1 = np.random.default_rng(seed + 10_001)
    rng2 = np.random.default_rng(seed + 20_002)

    # initialize from pool (optionally shared)
    idx1 = rng1.integers(0, pool.shape[0], size=M)
    if "init_indices" in sync_set:
        idx2 = idx1.copy()
    else:
        idx2 = rng2.integers(0, pool.shape[0], size=M)

    pop1.population = pool[idx1].astype(dtype, copy=True)
    pop2.population = pool[idx2].astype(dtype, copy=True)
    if recenter:
        pop1.population -= pop1.population.min(axis=1, keepdims=True).astype(dtype, copy=False)
        pop2.population -= pop2.population.min(axis=1, keepdims=True).astype(dtype, copy=False)
    if init_jitter > 0:
        rngj = np.random.default_rng(seed + 33_003)
        jitter = (init_jitter * rngj.normal(size=pop2.population.shape)).astype(dtype, copy=False)
        pop2.population += jitter
        # ensure no negative values introduced by jitter
        np.clip(pop2.population, 0.0, np.inf, out=pop2.population)
        if recenter:
            pop2.population -= pop2.population.min(axis=1, keepdims=True).astype(dtype, copy=False)

    # geometry & buffers
    nvec = pop1.n.astype(dtype, copy=False)
    grid = pop1.grid.astype(dtype, copy=False)
    phi = onsite_phi(nvec).astype(dtype, copy=False)
    scratch1 = np.empty_like(phi)
    scratch2 = np.empty_like(phi)
    hbuf1 = np.empty_like(phi)
    hbuf2 = np.empty_like(phi)

    # --- deterministic target times
    # 1-based t in [1, T]
    start_t = int(np.ceil((1.0 - sample_window) * T))
    start_t = max(1, min(start_t, T))
    if n_trace_samples == 1:
        t_targets = np.array([start_t], dtype=np.int32)
    else:
        # evenly spaced inclusive of start_t and T
        grid_f = np.linspace(start_t, T, n_trace_samples, endpoint=True)
        t_targets = np.rint(grid_f).astype(np.int32)
        # ensure bounds and monotonicity, and explicitly enforce endpoints
        t_targets[0] = start_t
        t_targets[-1] = T
        # de-duplicate if rounding caused collisions
        t_targets = np.unique(t_targets)
    # If uniqueness reduced count below desired, pad by walking forward
    if t_targets.size < min(n_trace_samples, T - start_t + 1):
        needed = min(n_trace_samples, T - start_t + 1) - t_targets.size
        # fill from the tail backward ensuring we get exact count
        candidates = [t for t in range(T, start_t - 1, -1) if t not in set(t_targets)]
        if candidates:
            add = np.array(sorted(candidates[:needed]))
            t_targets = np.sort(np.concatenate([t_targets, add]))
    # Final clamp to available steps
    max_count = max(1, T - start_t + 1)
    if t_targets.size > max_count:
        t_targets = t_targets[-max_count:]
    # Prepare for O(1) membership checks
    target_set = set(int(t) for t in t_targets.tolist())

    # --- run the full simulation & capture at target times
    t_series: list[int] = []
    H1_list: list[np.ndarray] = []
    H2_list: list[np.ndarray] = []

    bar = _pbar(T, "run")
    for t in range(1, T + 1):
        h1, h2 = _shared_step(rng1, rng2, pop1, pop2, phi, nvec, grid, scratch1, scratch2, hbuf1, hbuf2, recenter)
        if t in target_set:
            t_series.append(t)
            H1_list.append(h1.astype(dtype, copy=True))
            H2_list.append(h2.astype(dtype, copy=True))
        bar.update(1)
    bar.close()

    # Order outputs by time (already increasing)
    t_series_arr = np.asarray(t_series, dtype=np.int32)
    H1 = np.vstack(H1_list) if H1_list else np.empty((0, G), dtype=dtype)
    H2 = np.vstack(H2_list) if H2_list else np.empty((0, G), dtype=dtype)
    return t_series_arr, H1, H2
