"""
Validate the 1RSB cavity prediction against explicit small-system LV dynamics.

For one small random c-regular graph, this script:
1. Integrates the Lotka-Volterra dynamics from many random initial conditions.
2. Deduplicates the converged fixed points.
3. Computes the manuscript Lyapunov density H/S for each discovered fixed point.
4. Compares the resulting histogram to the cavity-predicted Sigma(H/S) curve.

The comparison is qualitative. The cavity curve is a large-S complexity, while
the finite-system histogram depends on how many unique states are discovered.

Usage
-----
Generate validation data and a figure against an existing cavity NPZ:

    python tests/validate_1rsb_small_system.py \
        --cavity-npz tests/rsb_c4_mu0.6000.npz

Quick smoke run:

    python tests/validate_1rsb_small_system.py \
        --cavity-npz tests/rsb_c4_mu0.6000.npz \
        --n-nodes 16 --n-inits 80 --steps 3000
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from cavity import compute_complexity


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)


def set_prl_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3, width=0.6)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.5, width=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


@dataclass
class ValidationParams:
    cavity_npz: str
    c: int = 4
    mu: float = 0.6
    n_nodes: int = 24
    n_inits: int = 400
    dt: float = 0.03
    steps: int = 8000
    convergence_tol: float = 1e-10
    residual_tol: float = 1e-6
    round_decimals: int = 6
    seed: int = 123
    bins: int = 18
    cavity_h_sign: float = -1.0
    save_prefix: str | None = None


def build_random_regular_graph(
    n_nodes: int,
    c: int,
    rng: np.random.Generator,
    *,
    max_tries: int = 10_000,
) -> np.ndarray:
    """
    Draw a simple c-regular graph by rejection on random stub pairings.

    This is intended for small systems only.
    """
    n_nodes = int(n_nodes)
    c = int(c)
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if c < 0 or c >= n_nodes:
        raise ValueError("need 0 <= c < n_nodes for a simple c-regular graph")
    if (n_nodes * c) % 2 != 0:
        raise ValueError("n_nodes * c must be even for a c-regular graph")

    stubs = np.repeat(np.arange(n_nodes, dtype=np.int32), c)
    for _ in range(max_tries):
        shuffled = stubs.copy()
        rng.shuffle(shuffled)
        pairs = shuffled.reshape(-1, 2)
        a = np.minimum(pairs[:, 0], pairs[:, 1])
        b = np.maximum(pairs[:, 0], pairs[:, 1])
        if np.any(a == b):
            continue
        edge_code = a.astype(np.int64) * n_nodes + b.astype(np.int64)
        if np.unique(edge_code).size != edge_code.size:
            continue

        adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        adj[a, b] = 1
        adj[b, a] = 1
        return adj

    raise RuntimeError("failed to draw a simple random regular graph; increase max_tries")


def interaction_matrix(adj: np.ndarray, mu: float) -> np.ndarray:
    a = -float(mu) * adj.astype(np.float64, copy=False)
    np.fill_diagonal(a, -1.0)
    return a


def lv_flow(n: np.ndarray, a: np.ndarray) -> np.ndarray:
    return n * (1.0 + a @ n)


def integrate_lv(
    a: np.ndarray,
    rng: np.random.Generator,
    *,
    dt: float,
    steps: int,
    convergence_tol: float,
    residual_tol: float,
) -> tuple[np.ndarray, float, int]:
    """
    Integrate the LV flow with a Heun step and stop early on convergence.
    """
    n = rng.uniform(1e-3, 1.0, size=a.shape[0]).astype(np.float64)
    dt = float(dt)
    steps = int(steps)
    check_every = 25

    for step in range(steps):
        f0 = lv_flow(n, a)
        trial = n + dt * f0
        np.clip(trial, 0.0, np.inf, out=trial)
        f1 = lv_flow(trial, a)
        n_next = n + 0.5 * dt * (f0 + f1)
        np.clip(n_next, 0.0, np.inf, out=n_next)

        if (step + 1) % check_every == 0:
            delta = float(np.max(np.abs(n_next - n)))
            resid = float(np.max(np.abs(lv_flow(n_next, a))))
            n = n_next
            if delta < convergence_tol and resid < residual_tol:
                return n, resid, step + 1
        else:
            n = n_next

    resid = float(np.max(np.abs(lv_flow(n, a))))
    return n, resid, steps


def lyapunov_density(state: np.ndarray, adj: np.ndarray, mu: float) -> float:
    state = np.asarray(state, dtype=np.float64)
    onsite = float(np.sum(state * (0.5 * state - 1.0)))
    edge = 0.5 * float(mu) * float(state @ (adj @ state))
    return (onsite + edge) / float(state.size)


def deduplicate_states(
    states: list[np.ndarray],
    residuals: list[float],
    n_steps: list[int],
    *,
    round_decimals: int,
) -> dict[str, np.ndarray]:
    buckets: dict[bytes, dict[str, object]] = {}

    for state, resid, steps_taken in zip(states, residuals, n_steps):
        rounded = np.round(np.asarray(state, dtype=np.float64), decimals=round_decimals)
        key = rounded.tobytes()
        if key not in buckets:
            buckets[key] = {
                "state": rounded,
                "count": 1,
                "residual": float(resid),
                "steps": int(steps_taken),
            }
        else:
            buckets[key]["count"] = int(buckets[key]["count"]) + 1
            buckets[key]["residual"] = min(float(buckets[key]["residual"]), float(resid))
            buckets[key]["steps"] = min(int(buckets[key]["steps"]), int(steps_taken))

    if not buckets:
        return {
            "states": np.empty((0, 0), dtype=np.float64),
            "multiplicity": np.empty(0, dtype=np.int64),
            "residual": np.empty(0, dtype=np.float64),
            "steps": np.empty(0, dtype=np.int64),
        }

    ordered = sorted(
        buckets.values(),
        key=lambda item: (-int(item["count"]), float(item["residual"])),
    )
    states_arr = np.stack([np.asarray(item["state"], dtype=np.float64) for item in ordered], axis=0)
    multiplicity = np.array([int(item["count"]) for item in ordered], dtype=np.int64)
    residual = np.array([float(item["residual"]) for item in ordered], dtype=np.float64)
    steps_arr = np.array([int(item["steps"]) for item in ordered], dtype=np.int64)
    return {
        "states": states_arr,
        "multiplicity": multiplicity,
        "residual": residual,
        "steps": steps_arr,
    }


def load_cavity_curve(npz_path: str | Path, *, cavity_h_sign: float) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        results = {key: data[key] for key in data.files}

    phi = np.asarray(results.get("Phi", results.get("phi")), dtype=float)
    beta = np.asarray(results["beta"], dtype=float)
    if "epsilon" not in results or ("Sigma" not in results and "sigma" not in results):
        cx = compute_complexity(beta, phi)
        results["epsilon"] = cx["epsilon"]
        results["Sigma"] = cx["sigma"]

    epsilon = np.asarray(results["epsilon"], dtype=float)
    sigma = np.asarray(results.get("Sigma", results["sigma"]), dtype=float)
    h_over_s = float(cavity_h_sign) * epsilon
    order = np.argsort(h_over_s)
    return {
        "beta": beta[order],
        "h_over_s": h_over_s[order],
        "sigma": sigma[order],
    }


def plot_validation(
    *,
    h_over_s_unique: np.ndarray,
    cavity_h_over_s: np.ndarray,
    cavity_sigma: np.ndarray,
    out_path: str | Path,
    bins: int,
) -> None:
    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    set_prl_axes(ax1)

    if h_over_s_unique.size:
        hist_counts, hist_edges, _ = ax1.hist(
            h_over_s_unique,
            bins=int(bins),
            density=True,
            color="0.75",
            edgecolor="0.25",
            linewidth=0.6,
            alpha=0.8,
        )
        rug_y = -0.02 * max(hist_counts.max(initial=1.0), 1.0)
        ax1.plot(h_over_s_unique, np.full_like(h_over_s_unique, rug_y), "|", color="0.2", markersize=6)
        ax1.set_ylim(bottom=1.5 * rug_y)
        if hist_edges.size >= 2:
            xmin = min(float(hist_edges[0]), float(np.min(cavity_h_over_s)))
            xmax = max(float(hist_edges[-1]), float(np.max(cavity_h_over_s)))
            if xmax > xmin:
                ax1.set_xlim(xmin, xmax)
    else:
        ax1.text(0.5, 0.55, "No converged unique\nfixed points found", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_xlabel(r"$H/S$")
    ax1.set_ylabel("Unique fixed-point density")

    ax2 = ax1.twinx()
    set_prl_axes(ax2)
    ax2.plot(
        cavity_h_over_s,
        cavity_sigma,
        "s-",
        linewidth=1.4,
        markersize=3.5,
        color="tab:blue",
    )
    ax2.axhline(0.0, color="tab:blue", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.set_ylabel(r"Cavity $\Sigma(H/S)$", color="tab:blue")
    ax2.tick_params(axis="y", colors="tab:blue")

    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def run_validation(params: ValidationParams) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(params.seed))
    cavity = load_cavity_curve(params.cavity_npz, cavity_h_sign=params.cavity_h_sign)

    adj = build_random_regular_graph(params.n_nodes, params.c, rng)
    a = interaction_matrix(adj, params.mu)

    converged_states: list[np.ndarray] = []
    converged_residuals: list[float] = []
    converged_steps: list[int] = []
    failed_residuals: list[float] = []

    for _ in range(int(params.n_inits)):
        state, resid, steps_taken = integrate_lv(
            a,
            rng,
            dt=params.dt,
            steps=params.steps,
            convergence_tol=params.convergence_tol,
            residual_tol=params.residual_tol,
        )
        if np.isfinite(state).all() and resid <= params.residual_tol:
            converged_states.append(state)
            converged_residuals.append(resid)
            converged_steps.append(steps_taken)
        else:
            failed_residuals.append(resid)

    dedup = deduplicate_states(
        converged_states,
        converged_residuals,
        converged_steps,
        round_decimals=params.round_decimals,
    )

    states = dedup["states"]
    multiplicity = dedup["multiplicity"]
    h_over_s_unique = np.array(
        [lyapunov_density(state, adj, params.mu) for state in states],
        dtype=np.float64,
    ) if states.size else np.empty(0, dtype=np.float64)

    prefix = params.save_prefix
    if prefix is None:
        prefix = f"validation_c{params.c}_mu{params.mu:.4f}_n{params.n_nodes}"
    out_npz = Path(f"{prefix}.npz")
    out_pdf = Path(f"{prefix}.pdf")

    np.savez(
        out_npz,
        c=np.int64(params.c),
        mu=np.float64(params.mu),
        n_nodes=np.int64(params.n_nodes),
        n_inits=np.int64(params.n_inits),
        dt=np.float64(params.dt),
        steps=np.int64(params.steps),
        round_decimals=np.int64(params.round_decimals),
        residual_tol=np.float64(params.residual_tol),
        adjacency=adj,
        states=states,
        multiplicity=multiplicity,
        residual=dedup["residual"],
        integration_steps=dedup["steps"],
        h_over_s_unique=h_over_s_unique,
        n_converged=np.int64(len(converged_states)),
        n_failed=np.int64(len(failed_residuals)),
        failed_residuals=np.asarray(failed_residuals, dtype=np.float64),
        cavity_h_over_s=cavity["h_over_s"],
        cavity_sigma=cavity["sigma"],
        cavity_beta=cavity["beta"],
    )

    plot_validation(
        h_over_s_unique=h_over_s_unique,
        cavity_h_over_s=cavity["h_over_s"],
        cavity_sigma=cavity["sigma"],
        out_path=out_pdf,
        bins=params.bins,
    )

    total_unique = int(h_over_s_unique.size)
    print(f"graph: c={params.c}, mu={params.mu:.4f}, S={params.n_nodes}")
    print(f"initial conditions: {params.n_inits}")
    print(f"converged trajectories: {len(converged_states)}")
    print(f"unique fixed points: {total_unique}")
    if total_unique:
        print(
            "H/S range of discovered unique states: "
            f"[{np.min(h_over_s_unique):.6f}, {np.max(h_over_s_unique):.6f}]"
        )
        print(
            "largest multiplicities among unique states: "
            + ", ".join(str(int(v)) for v in multiplicity[: min(5, multiplicity.size)])
        )
    if failed_residuals:
        print(
            "non-converged trajectories: "
            f"{len(failed_residuals)} (max residual {np.max(failed_residuals):.3e})"
        )
    print(f"wrote {out_npz}")
    print(f"wrote {out_pdf}")

    return {
        "h_over_s_unique": h_over_s_unique,
        "multiplicity": multiplicity,
        "cavity_h_over_s": cavity["h_over_s"],
        "cavity_sigma": cavity["sigma"],
    }


def parse_args(argv: list[str] | None = None) -> ValidationParams:
    p = argparse.ArgumentParser(
        description="Compare small-system LV fixed points to the 1RSB cavity Sigma(H/S) curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cavity-npz", required=True, help="Saved NPZ from run_1rsb_simulation.py")
    p.add_argument("--c", type=int, default=4, help="Regular degree")
    p.add_argument("--mu", type=float, default=0.6, help="Constant positive interaction strength")
    p.add_argument("--n-nodes", type=int, default=24, help="Small finite system size S")
    p.add_argument("--n-inits", type=int, default=400, help="Number of random initial conditions")
    p.add_argument("--dt", type=float, default=0.03, help="LV integration time step")
    p.add_argument("--steps", type=int, default=8000, help="Maximum integration steps per initial condition")
    p.add_argument("--convergence-tol", type=float, default=1e-10, help="Max coordinate update threshold for early stopping")
    p.add_argument("--residual-tol", type=float, default=1e-6, help="Max LV residual to accept a converged fixed point")
    p.add_argument("--round-decimals", type=int, default=6, help="Rounding used to deduplicate fixed points")
    p.add_argument("--bins", type=int, default=18, help="Histogram bins for the discovered H/S values")
    p.add_argument(
        "--cavity-h-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help=(
            "Sign used to convert the saved cavity epsilon into manuscript H/S. "
            "Use -1 for the current code-generated NPZ files, 1 if epsilon already equals H/S."
        ),
    )
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--save-prefix", default=None, help="Output prefix for the validation NPZ/PDF")
    args = p.parse_args(argv)
    return ValidationParams(
        cavity_npz=args.cavity_npz,
        c=args.c,
        mu=args.mu,
        n_nodes=args.n_nodes,
        n_inits=args.n_inits,
        dt=args.dt,
        steps=args.steps,
        convergence_tol=args.convergence_tol,
        residual_tol=args.residual_tol,
        round_decimals=args.round_decimals,
        bins=args.bins,
        cavity_h_sign=args.cavity_h_sign,
        seed=args.seed,
        save_prefix=args.save_prefix,
    )


def main(argv: list[str] | None = None) -> None:
    params = parse_args(argv)
    run_validation(params)


if __name__ == "__main__":
    main()
