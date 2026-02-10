"""
Compare population abundance histograms from:
1) cavity_simulation
2) warm-up pool (same procedure as replica_h_traces._warm_up_pool)
3) integrating a corresponding Lotka-Volterra (LV) system

Run this file to generate a plot with the three histograms together.
"""

import numpy as np

from cavity import (
    DegreeDistribution,
    JDistribution,
    PopulationConfig,
    cavity_simulation,
    warm_up_pool,
)


def _population_abundances(population: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return abundance at argmin of each h(n) curve."""
    idx = np.argmin(population, axis=1)
    return grid[idx]



def _sample_lv_system(degree, couplings, cfg, *, dt=0.01, steps=20000, seed_offset=7):
    """
    Integrate a corresponding LV system with random interactions.
    Assumptions:
    - Self-interaction = -1 for stability
    - Growth rates r_i = 1
    - Interactions follow degree distribution and coupling magnitudes
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    M = int(cfg.M)
    k_max = int(cfg.k_max)

    # Build sparse interaction matrix with negative couplings
    A = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        k_i = degree.sample_k(rng, size_bias=False, k_max=k_max)
        if k_i > 0:
            idx = rng.choice(M, size=min(k_i, M), replace=False)
            js = couplings.sample(rng, size=idx.size)
            A[i, idx] = js
        A[i, i] = -1.0

    r = np.ones(M, dtype=np.float64)
    N = rng.uniform(0.1, 1.0, size=M).astype(np.float64)

    for _ in range(int(steps)):
        dN = N * (r + A @ N)
        N = N + dt * dN
        np.clip(N, 0.0, np.inf, out=N)
    return N


def compare_abundance_histograms(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    burn_in: int = 5000,
    lv_steps: int = 20000,
    lv_dt: float = 0.01,
    bins: int = 30,
    out_path: str = "abundance_histograms.png",
):
    # cavity_simulation
    pop = cavity_simulation(degree, couplings, cfg, burn_in=burn_in)
    abund_cav = _population_abundances(pop, np.linspace(0.0, 1.0, cfg.G, dtype=cfg.dtype))

    # warm-up pool with same total updates as cavity_simulation (burn_in + T)
    pool = warm_up_pool(degree, couplings, cfg, burn_in=burn_in + int(cfg.T))
    abund_pool = _population_abundances(pool, np.linspace(0.0, 1.0, cfg.G, dtype=cfg.dtype))

    # LV system integration
    abund_lv = _sample_lv_system(degree, couplings, cfg, dt=lv_dt, steps=lv_steps)

    # plot
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to plot histograms") from exc

    plt.figure(figsize=(8, 5))
    plt.hist(abund_cav, bins=bins, histtype="step", linewidth=2.0, label="cavity_simulation")
    plt.hist(abund_pool, bins=bins, histtype="step", linewidth=2.0, label="warm_up_pool")
    plt.hist(abund_lv, bins=bins, histtype="step", linewidth=2.0, label="LV integration")
    plt.xlabel("Abundance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    cfg = PopulationConfig(G=401, M=3000, T=20_000, seed=123, verbose=False)
    degree = DegreeDistribution.poisson(3.0)
    couplings = JDistribution(kind="abs_gaussian", normal_sigma=0.1)
    path = compare_abundance_histograms(
        degree,
        couplings,
        cfg,
        burn_in=2000,
        lv_steps=20000,
        lv_dt=0.01,
        bins=30,
        out_path="abundance_histograms.png",
    )
    print(f"wrote {path}")
