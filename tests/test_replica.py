"""
Sweep mu for sigma=0.15, run replica_h_traces, and plot mean L-infinity norm
between replicas as a function of mu. Saves compact data to disk.
"""

import numpy as np
from tqdm import tqdm

from cavity import DegreeDistribution, JDistribution, PopulationConfig, replica_h_traces


def mean_linf_norm(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mean over samples of L-infinity norm between replica traces."""
    if h1.size == 0 or h2.size == 0:
        return float("nan")
    linf = np.max(np.abs(h1 - h2), axis=1)
    return float(linf.mean())

def main():
    # settings
    sigma = 0.0
    mu_values = np.linspace(0.0, 0.5, 2)
    cfg = PopulationConfig(G=601, M=2500, T=20_000, seed=123, verbose=True)
    degree = DegreeDistribution.poisson(3.0)

    mean_linf = np.empty(mu_values.shape[0], dtype=np.float64)

    for i, mu in enumerate(tqdm(mu_values, desc="mu sweep")):
        couplings = JDistribution(kind="abs_gaussian", normal_mu=float(mu), normal_sigma=sigma)
        _, h1, h2 = replica_h_traces(
            degree,
            couplings,
            cfg,
            burn_in=80_000,
            n_trace_samples=32,
            sample_window=0.1,
            init_jitter=1e-6,
            sync={"k", "parents", "js", "replace"},
        )
        mean_linf[i] = mean_linf_norm(h1, h2)

    # save compact results
    np.savez_compressed(
        "replica_linf_vs_mu.npz",
        mu=mu_values,
        mean_linf=mean_linf,
        sigma=sigma,
        G=int(cfg.G),
        M=int(cfg.M),
        T=int(cfg.T),
    )

    # plot
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to plot results") from exc

    plt.figure(figsize=(7, 4.5))
    plt.plot(mu_values, mean_linf, marker="o", linewidth=2.0)
    plt.xlabel("mu")
    plt.ylabel("mean L∞ norm |H1 - H2|")
    plt.title(f"Replica divergence vs mu (sigma={sigma})")
    plt.tight_layout()
    plt.savefig("replica_linf_vs_mu.png", dpi=150)
    print("wrote replica_linf_vs_mu.npz and replica_linf_vs_mu.png")


if __name__ == "__main__":
    main()
