"""
Complexity analysis for the 1RSB cavity method.

Provides the Legendre transform that converts the free entropy Φ(β) into the
complexity–energy curve Σ(ε), plus helper numerics and the analytic RS energy
formula for regular graphs.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(values))
    if np.isneginf(max_value):
        return max_value
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """
    Pool-adjacent-violators isotonic regression (non-decreasing).

    Returns a non-decreasing array with the same shape as y, obtained by
    merging adjacent blocks that violate monotonicity into their weighted
    average.
    """
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y.copy()

    values: list[float] = []
    counts: list[int] = []
    for val in y:
        values.append(float(val))
        counts.append(1)
        while len(values) >= 2 and values[-2] > values[-1]:
            total = counts[-2] + counts[-1]
            avg = (values[-2] * counts[-2] + values[-1] * counts[-1]) / total
            values[-2] = avg
            counts[-2] = total
            values.pop()
            counts.pop()

    out = np.empty(y.size, dtype=float)
    pos = 0
    for val, count in zip(values, counts):
        out[pos : pos + count] = val
        pos += count
    return out


def _local_poly_derivative(x: np.ndarray, y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Estimate dy/dx at each point using a local polynomial fit of degree ≤ 2
    over a sliding window of *window* points (odd, clamped to [3, n]).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n <= 1:
        return np.zeros_like(y)
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([slope, slope], dtype=float)

    win = min(window, n)
    if win % 2 == 0:
        win -= 1
    win = max(3, win)

    d = np.empty(n, dtype=float)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + win)
        lo = max(0, hi - win)
        xw = x[lo:hi]
        yw = y[lo:hi]
        deg = min(2, xw.size - 1)
        coef = np.polyfit(xw, yw, deg=deg)
        dcoef = np.polyder(coef)
        d[i] = float(np.polyval(dcoef, x[i]))
    return d


# ---------------------------------------------------------------------------
# Complexity curve
# ---------------------------------------------------------------------------

def compute_complexity(
    beta_values: np.ndarray,
    phi_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute the complexity–energy curve Σ(ε) from the free-entropy curve Φ(β).

    The Legendre transform is:

        ε(β)  = Φ_smooth(β) + β · dΦ/dβ   (Lyapunov energy density)
        Σ(ε)  = β² · dΦ/dβ                 (configurational entropy)

    A monotone-smoothing step (isotonic regression on Φ) is applied before
    differentiation to suppress MC noise.

    Parameters
    ----------
    beta_values : (N,) float
        Parisi inverse-temperature parameter, strictly increasing.
    phi_values : (N,) float
        Estimated free entropy Φ(β) at each β point.

    Returns
    -------
    dict with keys:
        beta, phi, phi_smooth, epsilon, sigma, dphi_dbeta
    """
    beta = np.asarray(beta_values, dtype=float)
    phi = np.asarray(phi_values, dtype=float)

    if beta.size == 0:
        raise ValueError("beta_values must contain at least one point")
    if beta.size == 1:
        dphi = np.zeros(1, dtype=float)
        return {
            "beta": beta,
            "phi": phi,
            "phi_smooth": phi.copy(),
            "epsilon": phi.copy(),
            "sigma": np.zeros(1, dtype=float),
            "dphi_dbeta": dphi,
        }

    phi_smooth = _isotonic_increasing(phi)
    dphi = _local_poly_derivative(beta, phi_smooth, window=5)
    epsilon = phi_smooth + beta * dphi
    sigma = beta ** 2 * dphi
    return {
        "beta": beta,
        "phi": phi,
        "phi_smooth": phi_smooth,
        "epsilon": epsilon,
        "sigma": sigma,
        "dphi_dbeta": dphi,
    }


# ---------------------------------------------------------------------------
# Analytic RS reference
# ---------------------------------------------------------------------------

def rs_energy(c: int, mu: float, g: int = 1001) -> float:
    """
    Analytic replica-symmetric energy density for a c-regular graph with
    uniform coupling strength mu.

    Valid for mu < mu_c = 1 / (2 * sqrt(c - 1)).  Returns the minimum of the
    RS energy landscape over the abundance grid.

    Parameters
    ----------
    c : int
        Node degree.
    mu : float
        Coupling strength.
    g : int
        Grid resolution for the minimisation.
    """
    delta = np.sqrt(max(0.0, 1.0 - 4.0 * (c - 1) * mu ** 2))
    denom = 1.0 - 2.0 * (c - 1) * mu + delta
    a = 0.5 * (1.0 - 2.0 * c * mu ** 2 / (1.0 + delta))
    b = (1.0 + 2.0 * mu + delta) / denom
    d = c * (c - 1) * (1.0 + delta) / ((c - 2) * denom ** 2)
    n_grid = np.linspace(0.0, 1.0, g)
    return float(np.min(a * n_grid ** 2 - b * n_grid + d))
