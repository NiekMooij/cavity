"""
Run a full 1RSB cavity simulation and save the data.

Sweeps the Parisi inverse temperature β over a configured range, estimating
the free entropy Φ(β) and site observables at each point, then computes
the complexity curve Σ(ε) via the Legendre transform.

Results are saved to an NPZ file. Use `plot_1rsb_simulation.py` to generate
the figure and diagnostics from the saved data.

Usage
-----
Quick smoke-run (small parameters, ~seconds):

    python run_1rsb_simulation.py --quick

Full production run (matches numerical/generate_cavity.py defaults):

    python run_1rsb_simulation.py

Custom parameters:

    python run_1rsb_simulation.py --c 4 --mu 0.6 --m 4000 --g 601 \\
        --beta-min 0.5 --beta-max 9.0 --n-beta 18 \\
        --n-steps 60000 --n-mc 20000 --burn-in 15000 \\
        --save my_results

Plot an existing dataset:

    python plot_1rsb_simulation.py --input my_results.npz
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

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
# Configuration
# ---------------------------------------------------------------------------

def build_beta_grid(
    beta_min: float,
    beta_max: float,
    n_beta: int,
    *,
    tail_beta_from: float | None = None,
    tail_n_beta: int = 0,
) -> tuple[float, ...]:
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_max <= beta_min:
        raise ValueError("beta_max must be larger than beta_min")
    if n_beta < 2:
        raise ValueError("n_beta must be at least 2")

    grids = [np.linspace(beta_min, beta_max, int(n_beta), dtype=np.float64)]
    if tail_beta_from is not None and tail_n_beta >= 2:
        tail_lo = max(beta_min, float(tail_beta_from))
        if tail_lo < beta_max:
            grids.append(np.linspace(tail_lo, beta_max, int(tail_n_beta), dtype=np.float64))

    beta_values = np.unique(np.round(np.concatenate(grids), decimals=12))
    beta_values.sort()
    return tuple(float(v) for v in beta_values)


@dataclass
class SimParams:
    # Graph / coupling
    c: int = 4
    mu: float = 0.6
    # Population
    m: int = 4000
    g: int = 601
    seed: int = 42
    # Beta sweep
    beta_values: tuple[float, ...] = tuple(float(v) for v in np.linspace(0.5, 9.0, 18))
    # Steps / MC
    n_steps: int = 60_000
    n_mc: int = 20_000
    n_estimate_repeats: int = 10
    # High-beta refinement
    high_beta_from: float = 6.0
    high_beta_steps: int = 180_000
    high_beta_n_mc: int = 50_000
    high_beta_repeats: int = 16
    # Warm-up
    burn_in: int = 15_000
    resample_batch: int = 256
    # Output
    save: str | None = None


QUICK_PARAMS = SimParams(
    m=500,
    g=101,
    beta_values=(0.5, 1.0, 2.0, 3.0),
    n_steps=500,
    n_mc=300,
    n_estimate_repeats=2,
    high_beta_from=99.0,   # disable high-beta branch
    high_beta_steps=500,
    high_beta_n_mc=300,
    high_beta_repeats=2,
    burn_in=300,
    resample_batch=32,
)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_1rsb_simulation(params: SimParams) -> dict:
    """
    Run the full 1RSB beta sweep and return a results dict.

    Keys: beta, phi, phi_std, epsilon, sigma, dphi_dbeta,
          diversity, diversity_std, mean_abundance, mean_abundance_std,
          c, mu
    """
    c, mu = params.c, params.mu
    beta_values = np.asarray(params.beta_values, dtype=np.float64)

    mu_c = 1.0 / (2.0 * np.sqrt(c - 1))
    print(f"\nc={c}  mu={mu:.4f}  mu_c={mu_c:.4f}  RSB phase: {mu > mu_c}")
    if mu < mu_c:
        print(f"RS analytic energy: {rs_energy(c, mu):.6f}")

    degree = DegreeDistribution.random_regular(c)
    couplings = JDistribution(
        kind="uniform_pos",
        uniform_low=mu,
        uniform_high=mu + 1e-9,
    )
    cfg = PopulationConfig(
        G=params.g,
        M=params.m,
        T=0,
        seed=params.seed,
        verbose=True,
        recenter=True,
        dtype=float,
    )

    print(f"\nRS warm-up  (burn_in={params.burn_in}, M={params.m}, G={params.g})...")
    t0 = time.time()
    population = Population1RSB(
        degree, couplings, cfg,
        burn_in=params.burn_in,
        resample_batch=params.resample_batch,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    phi_arr = np.empty(len(beta_values))
    phi_std_arr = np.empty(len(beta_values))
    diversity_arr = np.empty(len(beta_values))
    diversity_std_arr = np.empty(len(beta_values))
    abundance_arr = np.empty(len(beta_values))
    abundance_std_arr = np.empty(len(beta_values))

    print(
        f"\nBeta sweep: {beta_values[0]:.2f} → {beta_values[-1]:.2f}"
        f"  ({len(beta_values)} points)"
    )

    for i, beta in enumerate(beta_values):
        t1 = time.time()
        high = beta >= params.high_beta_from
        n_steps      = params.high_beta_steps  if high else params.n_steps
        n_mc         = params.high_beta_n_mc   if high else params.n_mc
        n_repeats    = params.high_beta_repeats if high else params.n_estimate_repeats

        population.run(beta, n_steps)

        phi_samples:       list[float] = []
        diversity_samples: list[float] = []
        abundance_samples: list[float] = []

        for rep in range(n_repeats):
            base_seed = params.seed + 1_000_000 * (i + 1) + rep
            res = estimate_phi(population, beta=beta, n_mc=n_mc, seed=base_seed)
            obs = estimate_site_observables(
                population, n_mc=n_mc, seed=base_seed + 500_000
            )
            phi_samples.append(res["phi"])
            diversity_samples.append(obs["diversity"])
            abundance_samples.append(obs["mean_abundance"])

        phi_arr[i]           = float(np.mean(phi_samples))
        phi_std_arr[i]       = float(np.std(phi_samples))
        diversity_arr[i]     = float(np.mean(diversity_samples))
        diversity_std_arr[i] = float(np.std(diversity_samples))
        abundance_arr[i]     = float(np.mean(abundance_samples))
        abundance_std_arr[i] = float(np.std(abundance_samples))

        print(
            f"  beta={beta:.2f}  Phi={phi_arr[i]:.5f} ± {phi_std_arr[i]:.5f}"
            f"  div={diversity_arr[i]:.3f}  [{time.time()-t1:.1f}s]"
        )

    cx = compute_complexity(beta_values, phi_arr)

    return dict(
        beta=beta_values,
        phi=phi_arr,
        Phi=phi_arr,
        phi_std=phi_std_arr,
        Phi_std=phi_std_arr,
        phi_smooth=cx["phi_smooth"],
        Phi_smooth=cx["phi_smooth"],
        epsilon=cx["epsilon"],
        sigma=cx["sigma"],
        Sigma=cx["sigma"],
        dphi_dbeta=cx["dphi_dbeta"],
        dPhi_dbeta=cx["dphi_dbeta"],
        diversity=diversity_arr,
        diversity_std=diversity_std_arr,
        mean_abundance=abundance_arr,
        mean_abundance_std=abundance_std_arr,
        c=np.int64(c),
        mu=np.float64(mu),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a 1RSB cavity simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Use small parameters for a fast smoke-run.")
    p.add_argument("--c",         type=int,   default=4)
    p.add_argument("--mu",        type=float, default=0.6)
    p.add_argument("--m",         type=int,   default=4000,   help="Population size")
    p.add_argument("--g",         type=int,   default=601,    help="Grid points")
    p.add_argument("--beta-min",  type=float, default=0.5)
    p.add_argument("--beta-max",  type=float, default=9.0)
    p.add_argument("--n-beta",    type=int,   default=18,     help="Number of beta points")
    p.add_argument("--n-steps",   type=int,   default=60_000)
    p.add_argument("--n-mc",      type=int,   default=20_000)
    p.add_argument("--n-repeats", type=int,   default=10,
                   help="Number of independent Phi/observable estimates per beta.")
    p.add_argument("--high-beta-from", type=float, default=6.0)
    p.add_argument("--high-beta-steps", type=int, default=180_000)
    p.add_argument("--high-beta-n-mc", type=int, default=50_000)
    p.add_argument("--high-beta-repeats", type=int, default=16)
    p.add_argument("--burn-in",   type=int,   default=15_000)
    p.add_argument("--resample-batch", type=int, default=256)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--save",      type=str,   default=None,
                   help="Output prefix (without .npz). Defaults to rsb_c<c>_mu<mu>.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.quick:
        params = QUICK_PARAMS
    else:
        beta_values = tuple(
            float(v) for v in np.linspace(args.beta_min, args.beta_max, args.n_beta)
        )
        params = SimParams(
            c=args.c,
            mu=args.mu,
            m=args.m,
            g=args.g,
            seed=args.seed,
            beta_values=beta_values,
            n_steps=args.n_steps,
            n_mc=args.n_mc,
            n_estimate_repeats=args.n_repeats,
            high_beta_from=args.high_beta_from,
            high_beta_steps=args.high_beta_steps,
            high_beta_n_mc=args.high_beta_n_mc,
            high_beta_repeats=args.high_beta_repeats,
            burn_in=args.burn_in,
            resample_batch=args.resample_batch,
            save=args.save,
        )

    results = run_1rsb_simulation(params)

    # Save NPZ
    save_prefix = params.save or f"rsb_c{params.c}_mu{params.mu:.4f}"
    npz_path = save_prefix + ".npz"
    np.savez(npz_path, **results)
    print(f"\nResults saved to {npz_path}")
    print("Use plot_1rsb_simulation.py to create the figure.")


if __name__ == "__main__":
    main()
