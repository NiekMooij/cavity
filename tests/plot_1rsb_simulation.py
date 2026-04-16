"""
Load a saved 1RSB NPZ file, print diagnostics, and generate the figure.

Usage
-----
    python plot_1rsb_simulation.py --input rsb_c4_mu0.6000.npz
"""
from __future__ import annotations

import argparse
import os
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
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6,
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


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = max(1, int(window))
    if window <= 1 or y.size <= 2:
        return y.copy()
    if window % 2 == 0:
        window += 1

    half = window // 2
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, i + half + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def load_results(npz_path: str | Path) -> dict:
    with np.load(npz_path) as data:
        results = {key: data[key] for key in data.files}

    if "phi_smooth" not in results and "Phi_smooth" not in results:
        cx = compute_complexity(results["beta"], results.get("Phi", results["phi"]))
        results["phi_smooth"] = cx["phi_smooth"]
        results["Phi_smooth"] = cx["phi_smooth"]
        results["epsilon"] = cx["epsilon"]
        results["sigma"] = cx["sigma"]
        results["Sigma"] = cx["sigma"]
        results["dphi_dbeta"] = cx["dphi_dbeta"]
        results["dPhi_dbeta"] = cx["dphi_dbeta"]
    if "Phi" not in results and "phi" in results:
        results["Phi"] = results["phi"]
    if "Phi_std" not in results and "phi_std" in results:
        results["Phi_std"] = results["phi_std"]
    if "Sigma" not in results and "sigma" in results:
        results["Sigma"] = results["sigma"]
    return results


def analyze_results(results: dict) -> list[str]:
    beta = np.asarray(results["beta"], dtype=float)
    phi = np.asarray(results.get("Phi", results["phi"]), dtype=float)
    phi_std = np.asarray(results.get("Phi_std", results["phi_std"]), dtype=float)
    epsilon = np.asarray(results["epsilon"], dtype=float)
    sigma = np.asarray(results.get("Sigma", results["sigma"]), dtype=float)
    phi_smooth = np.asarray(results.get("Phi_smooth", results.get("phi_smooth", phi)), dtype=float)
    diversity = np.asarray(results.get("diversity", []), dtype=float)
    abundance = np.asarray(results.get("mean_abundance", []), dtype=float)

    messages: list[str] = []

    if not np.isfinite(phi).all() or not np.isfinite(sigma).all() or not np.isfinite(epsilon).all():
        messages.append("FAIL: non-finite values detected in Phi/Sigma/epsilon.")
        return messages

    rel_phi_std = phi_std / np.maximum(np.abs(phi), 1e-12)
    messages.append(
        "OK: all primary arrays are finite; "
        f"median relative Phi uncertainty is {np.median(rel_phi_std):.2%} "
        f"(max {np.max(rel_phi_std):.2%})."
    )

    if phi_smooth.shape == phi.shape and np.all(np.diff(phi_smooth) >= -1e-10):
        messages.append("OK: isotonic-smoothed Phi(beta) is non-decreasing, as required by the complexity transform.")
    else:
        messages.append("WARN: smoothed Phi(beta) is not non-decreasing; complexity estimates are unreliable.")

    n_sigma_pos = int(np.sum(sigma > 0.0))
    if n_sigma_pos == 0:
        messages.append("WARN: Sigma(H/S) is non-positive everywhere; this looks RS-like rather than a clear RSB branch.")
    else:
        beta_peak = float(beta[np.argmax(sigma)])
        sigma_peak = float(np.max(sigma))
        messages.append(
            f"OK: Sigma(H/S) is positive on {n_sigma_pos}/{sigma.size} beta points, "
            f"with peak {sigma_peak:.3f} at beta={beta_peak:.2f}."
        )

    eps_diffs = np.diff(epsilon)
    if np.all(eps_diffs >= -1e-10):
        messages.append("OK: H/S(beta) is non-decreasing.")
    else:
        first_bad = int(np.where(eps_diffs < -1e-10)[0][0])
        messages.append(
            "WARN: H/S(beta) is not monotone; the Legendre branch turns back near "
            f"beta={beta[first_bad + 1]:.2f}. This usually means the high-beta tail is noisy "
            "and may need more MC samples."
        )

    if diversity.size:
        dmin = float(np.min(diversity))
        dmax = float(np.max(diversity))
        if dmin < -1e-8 or dmax > 1.0 + 1e-8:
            messages.append(f"WARN: diversity left the physical range [0, 1]: [{dmin:.3f}, {dmax:.3f}].")

    if abundance.size:
        amin = float(np.min(abundance))
        if amin < -1e-8:
            messages.append(f"WARN: mean abundance became negative ({amin:.4f}), which is unphysical.")

    return messages


def plot_results(results: dict, out_path: str, *, smooth_window: int = 3) -> None:
    beta = np.asarray(results["beta"], dtype=float)
    phi = np.asarray(results.get("Phi", results["phi"]), dtype=float)
    phi_std = np.asarray(results.get("Phi_std", results["phi_std"]), dtype=float)
    sigma = np.asarray(results.get("Sigma", results["sigma"]), dtype=float)
    eps = np.asarray(results["epsilon"], dtype=float)
    phi_smooth = np.asarray(results.get("Phi_smooth", results.get("phi_smooth", phi)), dtype=float)
    c = int(results["c"])
    mu = float(results["mu"])

    # Omit the last point from the figure; keep the saved data unchanged.
    if beta.size > 1:
        keep = slice(None, -1)
        beta = beta[keep]
        phi = phi[keep]
        phi_std = phi_std[keep]
        sigma = sigma[keep]
        eps = eps[keep]
        phi_smooth = phi_smooth[keep]

    phi_ma = moving_average(phi, smooth_window)
    phi_smooth_ma = moving_average(phi_smooth, smooth_window)
    eps_ma = moving_average(eps, smooth_window)
    sigma_ma = moving_average(sigma, smooth_window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    ax1.errorbar(
        beta,
        phi,
        yerr=phi_std,
        fmt="o",
        linestyle="none",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.1,
        markersize=4,
        capsize=3,
    )
    if phi_smooth.shape == phi.shape:
        ax1.plot(
            beta,
            phi_smooth_ma,
            linewidth=2.4,
            linestyle="--",
            color="black",
            zorder=3,
        )
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$\Phi(\beta)$")
    set_prl_axes(ax1)

    mask = sigma_ma >= -1e-3
    if mask.any():
        ax2.plot(
            eps_ma[mask],
            sigma_ma[mask],
            "s-",
            linewidth=1.8,
            markersize=4,
        )
        ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax2.set_xlabel(r"$H/S$")
        ax2.set_ylabel(r"$\Sigma(H/S)$")
        set_prl_axes(ax2)
    else:
        ax2.text(0.5, 0.5, "No physical Σ > 0\n(RS phase?)",
                 ha="center", va="center", transform=ax2.transAxes)
        set_prl_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Plot saved 1RSB simulation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input NPZ file produced by run_1rsb_simulation.py")
    p.add_argument("--output", default=None, help="Output PDF path. Defaults to the NPZ stem plus .pdf")
    p.add_argument("--smooth-window", type=int, default=3,
                   help="Centered moving-average window used for the plotted curves.")
    args = p.parse_args(argv)

    results = load_results(args.input)
    for line in analyze_results(results):
        print(line)

    out_path = args.output or str(Path(args.input).with_suffix(".pdf"))
    plot_results(results, out_path, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
