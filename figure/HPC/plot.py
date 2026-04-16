from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "output" / "data.npz"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "REGULAR_AT.pdf"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        mu_values = np.asarray(data["mu_values"], dtype=np.float64)
        mu_c = float(data["mu_c"])
        sample_l1 = np.asarray(data["sample_l1"], dtype=np.float64)
        sample_l2 = np.asarray(data["sample_l2"], dtype=np.float64)
        sample_linf = np.asarray(data["sample_linf"], dtype=np.float64)
        mean_l1 = np.asarray(data["mean_l1"], dtype=np.float64)
        mean_l2 = np.asarray(data["mean_l2"], dtype=np.float64)
        mean_linf = np.asarray(data["mean_linf"], dtype=np.float64)
        c = int(data["c"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )

    sample_l1_std = sample_l1.std(axis=1)
    sample_l2_std = sample_l2.std(axis=1)
    sample_linf_std = sample_linf.std(axis=1)

    fig, ax = plt.subplots(figsize=(6.0, 4.25), constrained_layout=True)
    ax.plot(mu_values, mean_l1, marker="o", linewidth=1.8, label="mean sample L1")
    ax.plot(mu_values, mean_l2, marker="s", linewidth=1.8, label="mean sample L2")
    ax.plot(mu_values, mean_linf, marker="^", linewidth=1.8, label="mean sample Linf")
    ax.fill_between(mu_values, mean_l1 - sample_l1_std, mean_l1 + sample_l1_std, alpha=0.15)
    ax.fill_between(mu_values, mean_l2 - sample_l2_std, mean_l2 + sample_l2_std, alpha=0.15)
    ax.fill_between(mu_values, mean_linf - sample_linf_std, mean_linf + sample_linf_std, alpha=0.12)
    ax.axvline(mu_c, color="black", linestyle="--", linewidth=1.0, label=rf"$\mu_c \approx {mu_c:.4f}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Replica separation")
    ax.set_title(f"Random regular c={c}: replica-difference metrics")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = plot_results()
    print(f"wrote plot: {output_path}")


if __name__ == "__main__":
    main()
