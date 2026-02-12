"""
Run the same abundance histogram comparison as tests/test_cavity.py,
using the installed `cavity` package.
"""

import numpy as np

from cavity import (
    CavityPopulation,
    DegreeDistribution,
    JDistribution,
    PopulationConfig,
    cavity_simulation,
    lv_integration,
    warm_up_pool,
)


def compare_abundance_histograms(
    degree: DegreeDistribution,
    couplings: JDistribution,
    cfg: PopulationConfig,
    *,
    burn_in: int = 5000,
    lv_steps: int = 20000,
    lv_dt: float = 0.01,
    bins: int = 30,
    marginal_samples: int = 20_000,
    out_path: str = "abundance_histograms.png",
):
    # cavity_simulation -> sample marginals from generated fields
    pop = cavity_simulation(degree, couplings, cfg, burn_in=burn_in)
    pop_from_cavity = CavityPopulation(degree, couplings, cfg, population=pop)
    abund_cav = pop_from_cavity.abundances(marginal_samples, seed=cfg.seed + 122)

    # warm-up pool with same total updates as cavity_simulation (burn_in + T)
    pool = warm_up_pool(degree, couplings, cfg, burn_in=burn_in + int(cfg.T))
    pop_from_fields = CavityPopulation(degree, couplings, cfg, population=pool)
    abund_pool = pop_from_fields.abundances(marginal_samples, seed=cfg.seed + 123)

    # LV system integration
    abund_lv = lv_integration(degree, couplings, cfg, dt=lv_dt, steps=lv_steps)

    # plot
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to plot histograms") from exc

    # Use shared bin edges and normalized density for fair comparison.
    all_vals = np.concatenate([abund_cav, abund_pool, abund_lv]).astype(np.float64, copy=False)
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("abundance arrays contain non-finite values")
    if hi <= lo:
        hi = lo + 1e-12
    bin_edges = np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.hist(abund_cav, bins=bin_edges, density=True, histtype="step", linewidth=2.0, label="cavity_simulation")
    plt.hist(abund_pool, bins=bin_edges, density=True, histtype="step", linewidth=2.0, label="warm_up_pool")
    plt.hist(abund_lv, bins=bin_edges, density=True, histtype="step", linewidth=2.0, label="LV integration")
    plt.xlabel("Abundance")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    cfg = PopulationConfig(G=401, M=8000, T=20_000, seed=123, verbose=True)
    degree = DegreeDistribution.poisson(4.0)
    couplings = JDistribution(kind="abs_gaussian", normal_sigma=0.1)
    path = compare_abundance_histograms(
        degree,
        couplings,
        cfg,
        burn_in=20_000,
        lv_steps=10000,
        lv_dt=0.05,
        bins=40,
        marginal_samples=20_000,
        out_path="abundance_histograms.png",
    )
    print(f"wrote {path}")
