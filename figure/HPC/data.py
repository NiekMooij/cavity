from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


def _find_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "cavity").is_dir():
            return candidate
    return None


ROOT = _find_repo_root(Path(__file__).resolve().parent)
if ROOT is not None and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SHARD_DIR = OUTPUT_DIR / "shards"
FINAL_DATA_PATH = OUTPUT_DIR / "data.npz"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

from cavity import DegreeDistribution, JDistribution, PopulationConfig, replica_h_traces


MU_C_C4 = 1.0 / (2.0 * np.sqrt(3.0))
SYNC_FLAGS = ("k", "parents", "js", "replace")


@dataclass(frozen=True)
class SweepParams:
    c: int = 4
    mu_min: float = 0.0
    mu_max: float = 0.4
    n_mu_points: int = 80
    coupling_eps: float = 1e-12
    g: int = 801
    m: int = 5000
    t: int = 700_000
    burn_in: int = 100_000
    seed: int = 123
    verbose: bool = True
    n_trace_samples: int = 48
    sample_window: float = 0.15
    init_jitter: float = 1e-6
    shard_dir: str = str(SHARD_DIR)
    output: str = str(FINAL_DATA_PATH)


QUICK_PARAMS = SweepParams(
    n_mu_points=4,
    g=151,
    m=600,
    t=3_000,
    burn_in=4_000,
    n_trace_samples=24,
    sample_window=0.20,
)


def _resolve_local_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path.cwd() / resolved


def _default_mu_values(params: SweepParams) -> np.ndarray:
    n_mu_points = int(params.n_mu_points)
    if n_mu_points <= 0:
        raise ValueError("n_mu_points must be positive")
    return np.linspace(
        float(params.mu_min),
        float(params.mu_max),
        n_mu_points,
        dtype=np.float64,
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


def _shared_metadata(params: SweepParams) -> dict[str, object]:
    return {
        "schema_version": 1,
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "c": int(params.c),
        "mu_min": float(params.mu_min),
        "mu_max": float(params.mu_max),
        "n_mu_points": int(params.n_mu_points),
        "coupling_eps": float(params.coupling_eps),
        "g": int(params.g),
        "m": int(params.m),
        "t": int(params.t),
        "burn_in": int(params.burn_in),
        "seed": int(params.seed),
        "n_trace_samples": int(params.n_trace_samples),
        "sample_window": float(params.sample_window),
        "init_jitter": float(params.init_jitter),
        "sync_flags": list(SYNC_FLAGS),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _metadata_array(metadata: dict[str, object]) -> np.ndarray:
    return np.array(json.dumps(metadata, sort_keys=True))


def _build_params(args: argparse.Namespace) -> SweepParams:
    base = QUICK_PARAMS if args.quick else SweepParams()
    return SweepParams(
        c=base.c if args.c is None else int(args.c),
        mu_min=base.mu_min if args.mu_min is None else float(args.mu_min),
        mu_max=base.mu_max if args.mu_max is None else float(args.mu_max),
        n_mu_points=base.n_mu_points if args.n_mu_points is None else int(args.n_mu_points),
        coupling_eps=base.coupling_eps if args.coupling_eps is None else float(args.coupling_eps),
        g=base.g if args.g is None else int(args.g),
        m=base.m if args.m is None else int(args.m),
        t=base.t if args.t is None else int(args.t),
        burn_in=base.burn_in if args.burn_in is None else int(args.burn_in),
        seed=base.seed if args.seed is None else int(args.seed),
        verbose=base.verbose if args.verbose is None else bool(args.verbose),
        n_trace_samples=base.n_trace_samples if args.n_trace_samples is None else int(args.n_trace_samples),
        sample_window=base.sample_window if args.sample_window is None else float(args.sample_window),
        init_jitter=base.init_jitter if args.init_jitter is None else float(args.init_jitter),
        shard_dir=base.shard_dir if args.shard_dir is None else str(args.shard_dir),
        output=base.output if args.output is None else str(args.output),
    )


def _mu_task_id(args: argparse.Namespace) -> int:
    if args.task_id is not None:
        return int(args.task_id)
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        raise ValueError("task id not provided and SLURM_ARRAY_TASK_ID is not set")
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def _run_single_mu(params: SweepParams, mu_index: int) -> dict[str, np.ndarray]:
    mu_values = _default_mu_values(params)
    if mu_index < 0 or mu_index >= mu_values.size:
        raise IndexError(f"mu index {mu_index} out of range for {mu_values.size} points")
    if int(params.c) != 4:
        raise ValueError("this workflow is fixed to c=4")

    mu = float(mu_values[mu_index])
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
    couplings = _constant_mu_distribution(mu, params.coupling_eps)

    print(f"task {mu_index + 1}/{mu_values.size}  mu={mu:.6f}")
    t0 = time.time()
    t_series, h1, h2 = replica_h_traces(
        degree,
        couplings,
        cfg,
        burn_in=params.burn_in,
        init_jitter=params.init_jitter,
        n_trace_samples=params.n_trace_samples,
        sample_window=params.sample_window,
        sync=set(SYNC_FLAGS),
    )
    if h1.shape != h2.shape:
        raise RuntimeError("replica_h_traces returned mismatched shapes")
    if h1.shape[0] != params.n_trace_samples:
        raise RuntimeError(f"expected {params.n_trace_samples} trace samples, got {h1.shape[0]}")

    stats = _compute_pair_stats(h1, h2)
    elapsed = time.time() - t0
    print(
        f"  mean L1={float(stats['mean_l1']):.6e}  "
        f"mean L2={float(stats['mean_l2']):.6e}  "
        f"mean Linf={float(stats['mean_linf']):.6e}  [{elapsed:.1f}s]"
    )

    metadata = _shared_metadata(params)
    metadata.update(
        {
            "mode": "shard",
            "mu_index": int(mu_index),
            "mu_value": mu,
            "elapsed_seconds": elapsed,
        }
    )

    return {
        "metadata_json": _metadata_array(metadata),
        "mu_index": np.int64(mu_index),
        "mu_value": np.float64(mu),
        "mu_c": np.float64(MU_C_C4),
        "c": np.int64(params.c),
        "sigma": np.float64(0.0),
        "coupling_eps": np.float64(params.coupling_eps),
        "grid": grid,
        "t_series": np.asarray(t_series, dtype=np.int32),
        "replica_1_samples": np.asarray(h1, dtype=np.float64),
        "replica_2_samples": np.asarray(h2, dtype=np.float64),
        "mean_replica_1": np.asarray(stats["mean_replica_1"], dtype=np.float64),
        "mean_replica_2": np.asarray(stats["mean_replica_2"], dtype=np.float64),
        "mean_abs_diff_profile": np.asarray(stats["mean_abs_diff_profile"], dtype=np.float64),
        "rms_diff_profile": np.asarray(stats["rms_diff_profile"], dtype=np.float64),
        "sample_l1": np.asarray(stats["sample_l1"], dtype=np.float64),
        "sample_l2": np.asarray(stats["sample_l2"], dtype=np.float64),
        "sample_linf": np.asarray(stats["sample_linf"], dtype=np.float64),
        "mean_l1": np.float64(stats["mean_l1"]),
        "mean_l2": np.float64(stats["mean_l2"]),
        "mean_linf": np.float64(stats["mean_linf"]),
        "G": np.int64(cfg.G),
        "M": np.int64(cfg.M),
        "T": np.int64(cfg.T),
        "burn_in": np.int64(params.burn_in),
        "n_trace_samples": np.int64(params.n_trace_samples),
        "sample_window": np.float64(params.sample_window),
        "seed": np.int64(params.seed),
        "init_jitter": np.float64(params.init_jitter),
        "sync_flags": np.array(SYNC_FLAGS),
    }


def _shard_path(shard_dir: Path, mu_index: int) -> Path:
    return shard_dir / f"mu_{mu_index:04d}.npz"


def run_shard_mode(params: SweepParams, *, task_id: int) -> Path:
    shard_dir = _resolve_local_path(params.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = _shard_path(shard_dir, task_id)
    result = _run_single_mu(params, task_id)
    np.savez_compressed(shard_path, **result)
    print(f"wrote shard: {shard_path}")
    return shard_path


def _load_shard(shard_path: Path) -> dict[str, np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def combine_shards(params: SweepParams) -> Path:
    shard_dir = _resolve_local_path(params.shard_dir)
    output_path = _resolve_local_path(params.output)
    mu_values = _default_mu_values(params)
    shard_paths = [_shard_path(shard_dir, idx) for idx in range(mu_values.size)]
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(path.name for path in missing[:8])
        if len(missing) > 8:
            missing_text += ", ..."
        raise FileNotFoundError(f"missing {len(missing)} shard(s): {missing_text}")

    shard_data = [_load_shard(path) for path in shard_paths]
    shard_data.sort(key=lambda item: int(item["mu_index"]))

    grid = np.asarray(shard_data[0]["grid"], dtype=np.float64)
    n_mu = len(shard_data)
    n_trace_samples = int(shard_data[0]["n_trace_samples"])
    g = int(shard_data[0]["G"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _shared_metadata(params)
    metadata.update(
        {
            "mode": "combine",
            "mu_values": [float(v) for v in mu_values],
            "shard_dir": str(shard_dir),
            "n_shards": n_mu,
        }
    )

    results = {
        "metadata_json": _metadata_array(metadata),
        "mu_values": np.asarray([float(item["mu_value"]) for item in shard_data], dtype=np.float64),
        "mu_c": np.float64(MU_C_C4),
        "c": np.int64(int(shard_data[0]["c"])),
        "sigma": np.float64(0.0),
        "coupling_eps": np.float64(float(shard_data[0]["coupling_eps"])),
        "grid": grid,
        "t_series": np.empty((n_mu, n_trace_samples), dtype=np.int32),
        "replica_1_samples": np.empty((n_mu, n_trace_samples, g), dtype=np.float64),
        "replica_2_samples": np.empty((n_mu, n_trace_samples, g), dtype=np.float64),
        "mean_replica_1": np.empty((n_mu, g), dtype=np.float64),
        "mean_replica_2": np.empty((n_mu, g), dtype=np.float64),
        "mean_abs_diff_profile": np.empty((n_mu, g), dtype=np.float64),
        "rms_diff_profile": np.empty((n_mu, g), dtype=np.float64),
        "sample_l1": np.empty((n_mu, n_trace_samples), dtype=np.float64),
        "sample_l2": np.empty((n_mu, n_trace_samples), dtype=np.float64),
        "sample_linf": np.empty((n_mu, n_trace_samples), dtype=np.float64),
        "mean_l1": np.empty(n_mu, dtype=np.float64),
        "mean_l2": np.empty(n_mu, dtype=np.float64),
        "mean_linf": np.empty(n_mu, dtype=np.float64),
        "G": np.int64(g),
        "M": np.int64(int(shard_data[0]["M"])),
        "T": np.int64(int(shard_data[0]["T"])),
        "burn_in": np.int64(int(shard_data[0]["burn_in"])),
        "n_trace_samples": np.int64(n_trace_samples),
        "sample_window": np.float64(float(shard_data[0]["sample_window"])),
        "seed": np.int64(int(shard_data[0]["seed"])),
        "init_jitter": np.float64(float(shard_data[0]["init_jitter"])),
        "sync_flags": np.array(SYNC_FLAGS),
    }

    for idx, item in enumerate(shard_data):
        if int(item["mu_index"]) != idx:
            raise RuntimeError(f"expected shard index {idx}, got {int(item['mu_index'])}")
        if np.asarray(item["grid"], dtype=np.float64).shape != grid.shape or not np.allclose(item["grid"], grid):
            raise RuntimeError(f"grid mismatch in shard {idx}")
        results["t_series"][idx] = np.asarray(item["t_series"], dtype=np.int32)
        results["replica_1_samples"][idx] = np.asarray(item["replica_1_samples"], dtype=np.float64)
        results["replica_2_samples"][idx] = np.asarray(item["replica_2_samples"], dtype=np.float64)
        results["mean_replica_1"][idx] = np.asarray(item["mean_replica_1"], dtype=np.float64)
        results["mean_replica_2"][idx] = np.asarray(item["mean_replica_2"], dtype=np.float64)
        results["mean_abs_diff_profile"][idx] = np.asarray(item["mean_abs_diff_profile"], dtype=np.float64)
        results["rms_diff_profile"][idx] = np.asarray(item["rms_diff_profile"], dtype=np.float64)
        results["sample_l1"][idx] = np.asarray(item["sample_l1"], dtype=np.float64)
        results["sample_l2"][idx] = np.asarray(item["sample_l2"], dtype=np.float64)
        results["sample_linf"][idx] = np.asarray(item["sample_linf"], dtype=np.float64)
        results["mean_l1"][idx] = float(item["mean_l1"])
        results["mean_l2"][idx] = float(item["mean_l2"])
        results["mean_linf"][idx] = float(item["mean_linf"])

    np.savez_compressed(output_path, **results)
    print(f"wrote combined data: {output_path}")
    return output_path


def print_array_spec(params: SweepParams) -> None:
    n_mu = _default_mu_values(params).size
    print(f"0-{n_mu - 1}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="Use a lightweight smoke-test configuration.")
    parser.add_argument("--c", type=int, default=None, help="Graph degree.")
    parser.add_argument("--mu-min", type=float, default=None, help="Minimum mu value.")
    parser.add_argument("--mu-max", type=float, default=None, help="Maximum mu value.")
    parser.add_argument("--n-mu-points", type=int, default=None, help="Number of mu values.")
    parser.add_argument("--coupling-eps", type=float, default=None, help="Tiny width for constant couplings.")
    parser.add_argument("--g", type=int, default=None, help="Grid size.")
    parser.add_argument("--m", type=int, default=None, help="Population size.")
    parser.add_argument("--t", type=int, default=None, help="Number of run steps.")
    parser.add_argument("--burn-in", type=int, default=None, help="Warm-up steps.")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument("--n-trace-samples", type=int, default=None, help="Tail samples to record.")
    parser.add_argument("--sample-window", type=float, default=None, help="Tail fraction used for sampling.")
    parser.add_argument("--init-jitter", type=float, default=None, help="Initial replica perturbation.")
    parser.add_argument("--shard-dir", default=None, help="Directory for per-mu shard files.")
    parser.add_argument("--output", default=None, help="Combined output .npz path.")
    parser.add_argument("--verbose", default=None, action=argparse.BooleanOptionalAction, help="Enable solver verbosity.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel random-regular two-replica sweep for SLURM array execution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard_parser = subparsers.add_parser("shard", help="Run one mu point and save a shard.")
    _add_common_args(shard_parser)
    shard_parser.add_argument("--task-id", type=int, default=None, help="Mu index. Defaults to SLURM_ARRAY_TASK_ID.")

    combine_parser = subparsers.add_parser("combine", help="Combine all shard files into one .npz.")
    _add_common_args(combine_parser)

    array_parser = subparsers.add_parser("array-spec", help="Print the SLURM array range for the current config.")
    _add_common_args(array_parser)

    args = parser.parse_args()
    params = _build_params(args)

    if args.command == "shard":
        run_shard_mode(params, task_id=_mu_task_id(args))
        return
    if args.command == "combine":
        combine_shards(params)
        return
    if args.command == "array-spec":
        print_array_spec(params)
        return
    raise RuntimeError(f"unsupported command {args.command}")


if __name__ == "__main__":
    main()
