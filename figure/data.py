from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

from cavity import DegreeDistribution, JDistribution, PopulationConfig, replica_h_traces


MU_C_C4 = 1.0 / (2.0 * np.sqrt(3.0))


@dataclass(frozen=True)
class SweepParams:
    c: int = 4
    mu_values: tuple[float, ...] = ()
    mu_min: float = 0.0
    mu_max: float = 0.4
    n_mu_points: int = 60
    coupling_eps: float = 1e-12
    g: int = 401
    m: int = 2500
    t: int = 30_000
    burn_in: int = 60_000
    seed: int = 123
    verbose: bool = True
    n_trace_samples: int = 48
    sample_window: float = 0.15
    init_jitter: float = 1e-6
    output: str = "data.npz"


QUICK_PARAMS = SweepParams(
    n_mu_points=4,
    g=151,
    m=600,
    t=3_000,
    burn_in=4_000,
    n_trace_samples=24,
    sample_window=0.20,
    output="data.npz",
)


def _constant_mu_distribution(mu: float, eps: float) -> JDistribution:
    return JDistribution(
        kind="uniform_pos",
        uniform_low=float(mu),
        uniform_high=float(mu) + float(eps),
    )


def _compute_pair_stats(h1: np.ndarray, h2: np.ndarray) -> dict[str, np.ndarray | float]:
    diff = h1 - h2
    abs_diff = np.abs(diff)

    sample_l1 = abs_diff.mean(axis=1)
    sample_l2 = np.sqrt(np.mean(diff * diff, axis=1))
    sample_linf = abs_diff.max(axis=1)

    return {
        "mean_replica_1": h1.mean(axis=0),
        "mean_replica_2": h2.mean(axis=0),
        "mean_abs_diff_profile": abs_diff.mean(axis=0),
        "rms_diff_profile": np.sqrt(np.mean(diff * diff, axis=0)),
        "sample_l1": sample_l1,
        "sample_l2": sample_l2,
        "sample_linf": sample_linf,
        "mean_l1": float(sample_l1.mean()),
        "mean_l2": float(sample_l2.mean()),
        "mean_linf": float(sample_linf.mean()),
    }


def run_sweep(params: SweepParams) -> dict[str, np.ndarray]:
    mu_values = np.asarray(params.mu_values, dtype=np.float64)
    if mu_values.ndim != 1 or mu_values.size == 0:
        raise ValueError("mu_values must be a non-empty 1D list")
    if not np.all(np.diff(mu_values) >= 0.0):
        raise ValueError("mu_values must be sorted in non-decreasing order")
    if int(params.c) != 4:
        raise ValueError("this figure workflow is fixed to c=4")

    degree = DegreeDistribution.random_regular(params.c)
    cfg = PopulationConfig(
        G=params.g,
        M=params.m,
        T=params.t,
        seed=params.seed,
        verbose=params.verbose,
        recenter=True,
        dtype=float,
    )
    grid = np.linspace(0.0, 1.0, cfg.G, dtype=np.float64)

    n_mu = mu_values.size
    replica_1_samples = np.empty((n_mu, params.n_trace_samples, cfg.G), dtype=np.float64)
    replica_2_samples = np.empty((n_mu, params.n_trace_samples, cfg.G), dtype=np.float64)
    mean_replica_1 = np.empty((n_mu, cfg.G), dtype=np.float64)
    mean_replica_2 = np.empty((n_mu, cfg.G), dtype=np.float64)
    mean_abs_diff_profile = np.empty((n_mu, cfg.G), dtype=np.float64)
    rms_diff_profile = np.empty((n_mu, cfg.G), dtype=np.float64)
    sample_l1 = np.empty((n_mu, params.n_trace_samples), dtype=np.float64)
    sample_l2 = np.empty((n_mu, params.n_trace_samples), dtype=np.float64)
    sample_linf = np.empty((n_mu, params.n_trace_samples), dtype=np.float64)
    mean_l1 = np.empty(n_mu, dtype=np.float64)
    mean_l2 = np.empty(n_mu, dtype=np.float64)
    mean_linf = np.empty(n_mu, dtype=np.float64)
    t_series_all = np.empty((n_mu, params.n_trace_samples), dtype=np.int32)

    print(f"Random regular two-replica sweep for c={params.c}")
    print(f"mu_c ~= {MU_C_C4:.6f}")
    print(
        f"config: G={cfg.G}, M={cfg.M}, T={cfg.T}, burn_in={params.burn_in}, "
        f"n_trace_samples={params.n_trace_samples}"
    )

    for i, mu in enumerate(mu_values):
        couplings = _constant_mu_distribution(float(mu), params.coupling_eps)
        print(f"\n[{i + 1}/{n_mu}] mu={mu:.6f}")
        t0 = time.time()
        t_series, h1, h2 = replica_h_traces(
            degree,
            couplings,
            cfg,
            burn_in=params.burn_in,
            init_jitter=params.init_jitter,
            n_trace_samples=params.n_trace_samples,
            sample_window=params.sample_window,
            sync={"k", "parents", "js", "replace"},
        )
        if h1.shape != h2.shape:
            raise RuntimeError("replica_h_traces returned mismatched shapes")
        if h1.shape[0] != params.n_trace_samples:
            raise RuntimeError(
                f"expected {params.n_trace_samples} trace samples, got {h1.shape[0]}"
            )

        stats = _compute_pair_stats(h1, h2)
        replica_1_samples[i] = h1
        replica_2_samples[i] = h2
        mean_replica_1[i] = np.asarray(stats["mean_replica_1"], dtype=np.float64)
        mean_replica_2[i] = np.asarray(stats["mean_replica_2"], dtype=np.float64)
        mean_abs_diff_profile[i] = np.asarray(stats["mean_abs_diff_profile"], dtype=np.float64)
        rms_diff_profile[i] = np.asarray(stats["rms_diff_profile"], dtype=np.float64)
        sample_l1[i] = np.asarray(stats["sample_l1"], dtype=np.float64)
        sample_l2[i] = np.asarray(stats["sample_l2"], dtype=np.float64)
        sample_linf[i] = np.asarray(stats["sample_linf"], dtype=np.float64)
        mean_l1[i] = float(stats["mean_l1"])
        mean_l2[i] = float(stats["mean_l2"])
        mean_linf[i] = float(stats["mean_linf"])
        t_series_all[i] = t_series

        print(
            f"  mean L1={mean_l1[i]:.6e}  mean L2={mean_l2[i]:.6e}  "
            f"mean Linf={mean_linf[i]:.6e}  [{time.time() - t0:.1f}s]"
        )

    return {
        "mu_values": mu_values,
        "mu_c": np.float64(MU_C_C4),
        "c": np.int64(params.c),
        "sigma": np.float64(0.0),
        "coupling_eps": np.float64(params.coupling_eps),
        "grid": grid,
        "t_series": t_series_all,
        "replica_1_samples": replica_1_samples,
        "replica_2_samples": replica_2_samples,
        "mean_replica_1": mean_replica_1,
        "mean_replica_2": mean_replica_2,
        "mean_abs_diff_profile": mean_abs_diff_profile,
        "rms_diff_profile": rms_diff_profile,
        "sample_l1": sample_l1,
        "sample_l2": sample_l2,
        "sample_linf": sample_linf,
        "mean_l1": mean_l1,
        "mean_l2": mean_l2,
        "mean_linf": mean_linf,
        "G": np.int64(cfg.G),
        "M": np.int64(cfg.M),
        "T": np.int64(cfg.T),
        "burn_in": np.int64(params.burn_in),
        "n_trace_samples": np.int64(params.n_trace_samples),
        "sample_window": np.float64(params.sample_window),
        "seed": np.int64(params.seed),
        "init_jitter": np.float64(params.init_jitter),
        "sync_flags": np.array(("k", "parents", "js", "replace")),
    }


def _parse_mu_values(args: argparse.Namespace, default_values: tuple[float, ...]) -> tuple[float, ...]:
    if args.mu_values:
        values = tuple(float(v) for v in args.mu_values)
    else:
        values = tuple(float(v) for v in default_values)
    if len(values) == 0:
        raise ValueError("need at least one mu value")
    return tuple(sorted(values))


def _default_mu_values(params: SweepParams) -> tuple[float, ...]:
    n_mu_points = int(params.n_mu_points)
    if n_mu_points <= 0:
        raise ValueError("n_mu_points must be positive")
    values = np.linspace(
        float(params.mu_min),
        float(params.mu_max),
        n_mu_points,
        dtype=np.float64,
    )
    return tuple(float(v) for v in values)


def _resolve_output_path(output: str | Path) -> Path:
    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    return BASE_DIR / output_path


def _build_metadata(params: SweepParams) -> np.ndarray:
    metadata = {
        "schema_version": 1,
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "c": int(params.c),
        "mu_min": float(params.mu_min),
        "mu_max": float(params.mu_max),
        "n_mu_points": int(params.n_mu_points),
        "mu_values": [float(v) for v in params.mu_values],
        "coupling_eps": float(params.coupling_eps),
        "g": int(params.g),
        "m": int(params.m),
        "t": int(params.t),
        "burn_in": int(params.burn_in),
        "seed": int(params.seed),
        "n_trace_samples": int(params.n_trace_samples),
        "sample_window": float(params.sample_window),
        "init_jitter": float(params.init_jitter),
        "sync_flags": ["k", "parents", "js", "replace"],
    }
    return np.array(json.dumps(metadata, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-replica cavity sweep on random regular graphs with c=4 and constant mu."
    )
    parser.add_argument("--quick", action="store_true", help="Use a lightweight smoke-test configuration.")
    parser.add_argument("--mu-values", nargs="*", default=None, help="Explicit mu values to sweep.")
    parser.add_argument("--g", type=int, default=None, help="Grid size.")
    parser.add_argument("--m", type=int, default=None, help="Population size.")
    parser.add_argument("--t", type=int, default=None, help="Number of run steps.")
    parser.add_argument("--burn-in", type=int, default=None, help="Warm-up steps.")
    parser.add_argument("--n-trace-samples", type=int, default=None, help="Tail samples to record.")
    parser.add_argument("--sample-window", type=float, default=None, help="Tail fraction used for sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument("--output", default=None, help="Output .npz path.")
    parser.add_argument("--skip-plots", action="store_true", help="Only save the data, do not make figures.")
    args = parser.parse_args()

    base = QUICK_PARAMS if args.quick else SweepParams()
    default_mu_values = _default_mu_values(base)
    params = SweepParams(
        c=base.c,
        mu_values=_parse_mu_values(args, default_mu_values),
        mu_min=base.mu_min,
        mu_max=base.mu_max,
        n_mu_points=base.n_mu_points,
        coupling_eps=base.coupling_eps,
        g=base.g if args.g is None else int(args.g),
        m=base.m if args.m is None else int(args.m),
        t=base.t if args.t is None else int(args.t),
        burn_in=base.burn_in if args.burn_in is None else int(args.burn_in),
        seed=base.seed if args.seed is None else int(args.seed),
        verbose=True,
        n_trace_samples=base.n_trace_samples if args.n_trace_samples is None else int(args.n_trace_samples),
        sample_window=base.sample_window if args.sample_window is None else float(args.sample_window),
        init_jitter=base.init_jitter,
        output=base.output if args.output is None else str(args.output),
    )

    results = run_sweep(params)
    output_path = _resolve_output_path(params.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata_json=_build_metadata(params), **results)
    print(f"\nwrote data: {output_path}")

    if not args.skip_plots:
        from plot import plot_results

        plot_path = plot_results(output_path)
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
