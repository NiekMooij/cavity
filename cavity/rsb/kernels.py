"""
1RSB-specific numba kernels.

These complement cavity.kernels (which handles the RS cavity update) with the
batch operations needed for the 1RSB population dynamics and free-energy
estimation:

  - Convex-hull trick variant that writes into caller-supplied buffers
    (avoids allocation inside tight numba loops).
  - Indexed-parent accumulation (_accum_min_convs_indexed): like the RS
    _accum_min_convs but selects a subset of rows via an index array.
  - Batch candidate generation and beta-weighted importance resampling.
  - Batch site / edge free-energy kernels for MC estimation.
"""
from __future__ import annotations

import numpy as np

NUMBA_IMPORT_ERROR = None

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception as exc:
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco


# ---------------------------------------------------------------------------
# Convex-hull trick (pre-allocated variant)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _cht_build_mono_into(
    slopes: np.ndarray,
    inter: np.ndarray,
    mh: np.ndarray,
    ch: np.ndarray,
    xs: np.ndarray,
) -> int:
    """
    Build the lower envelope of lines y = slopes[i]*x + inter[i] for
    monotone-decreasing slopes, writing results into the caller-supplied
    arrays mh/ch/xs.  Returns the hull size k.

    Differs from cavity.kernels._cht_build_mono in that it avoids any heap
    allocation, making it safe to call inside parallel numba loops.
    """
    k = 0
    for i in range(slopes.shape[0]):
        mi = slopes[i]
        ci = inter[i]
        if k > 0 and mi == mh[k - 1]:
            if ci >= ch[k - 1]:
                continue
            k -= 1

        s = 0.0
        while k > 0:
            denom = mh[k - 1] - mi
            if denom == 0.0:
                s = np.inf
            else:
                s = (ci - ch[k - 1]) / denom
            if s <= xs[k - 1]:
                k -= 1
            else:
                break

        xs[k] = -np.inf if k == 0 else s
        mh[k] = mi
        ch[k] = ci
        k += 1
    return k


@njit(cache=True, fastmath=True)
def _accum_min_convs_indexed(
    pool: np.ndarray,
    idx: np.ndarray,
    js: np.ndarray,
    nvec: np.ndarray,
    out_accum: np.ndarray,
    slopes: np.ndarray,
    mh: np.ndarray,
    ch: np.ndarray,
    xs: np.ndarray,
) -> None:
    """
    For each parent p in idx:
        out_accum[g] += min_x ( pool[idx[p], x] - js[p] * nvec[g] * x )

    Uses the CHT for an O(G) scan (after O(G) hull build) per parent.
    All scratch arrays (slopes, mh, ch, xs) must be pre-allocated with
    length >= pool.shape[1].

    Unlike _accum_min_convs (which operates on all rows of P), this kernel
    selects rows via the index array idx, enabling batch operations where
    each candidate uses a different subset of parents.
    """
    for p in range(idx.shape[0]):
        hull_size = _cht_build_mono_into(slopes, pool[idx[p]], mh, ch, xs)
        ptr = 0
        jp = js[p]
        for g in range(nvec.shape[0]):
            sval = nvec[g] * jp
            while ptr + 1 < hull_size and xs[ptr + 1] <= sval:
                ptr += 1
            out_accum[g] += mh[ptr] * sval + ch[ptr]


@njit(cache=True, fastmath=True)
def _min_value(arr: np.ndarray) -> float:
    """Return the minimum value of a 1-D array (no-allocation helper)."""
    best = arr[0]
    for i in range(1, arr.shape[0]):
        if arr[i] < best:
            best = arr[i]
    return best


# ---------------------------------------------------------------------------
# 1RSB candidate generation and beta-weighted resampling
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def build_candidates_numba(
    h0: np.ndarray,
    phi: np.ndarray,
    grid: np.ndarray,
    js: np.ndarray,
    parent_idx: np.ndarray,
    out_candidates: np.ndarray,
    out_shifts: np.ndarray,
) -> None:
    """
    For each of the *batch* candidates b:
        out_candidates[b] = phi + sum_{t} min_{x'} (h0[parent_idx[b,t], x'] - js[t]*grid*x')
        out_shifts[b]     = min(out_candidates[b])   (before subtraction)

    The candidates are recentred (shifted so their minimum is 0) to keep
    field values numerically bounded.

    Parameters
    ----------
    h0 : (M, G) float64
        Current 1RSB population pool.
    phi : (G,) float64
        On-site potential.
    grid : (G,) float64
        Abundance grid.
    js : (c-1,) float64
        Cavity couplings (negative by convention).
    parent_idx : (batch, c-1) int64
        Parent indices for each candidate.
    out_candidates : (batch, G) float64  — output
    out_shifts : (batch,) float64        — output (cavity free energies)
    """
    batch = parent_idx.shape[0]
    g = h0.shape[1]
    slopes = -grid.copy()
    mh = np.empty(g, dtype=h0.dtype)
    ch = np.empty(g, dtype=h0.dtype)
    xs = np.empty(g, dtype=h0.dtype)

    for b in range(batch):
        accum = out_candidates[b]
        for i in range(g):
            accum[i] = phi[i]
        _accum_min_convs_indexed(h0, parent_idx[b], js, grid, accum, slopes, mh, ch, xs)
        shift = _min_value(accum)
        out_shifts[b] = shift
        for i in range(g):
            accum[i] -= shift


@njit(cache=True, fastmath=True)
def _sample_cdf(cdf: np.ndarray, u: float) -> int:
    """Binary-search CDF lookup: return smallest i s.t. cdf[i] >= u."""
    lo = 0
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True, fastmath=True)
def resample_indices_numba(
    shifts: np.ndarray,
    beta: float,
    uniforms: np.ndarray,
    out_idx: np.ndarray,
) -> None:
    """
    Importance-resample *batch* candidates using weights w_b ∝ exp(-β * shifts[b]).

    Parameters
    ----------
    shifts : (batch,) float64
        Cavity free energies (minimum of each candidate field).
    beta : float
        Inverse temperature.
    uniforms : (n_replace,) float64
        Pre-drawn uniform random numbers in [0, 1).
    out_idx : (n_replace,) int64  — output
        Indices into [0, batch) drawn proportional to the weights.
    """
    n = shifts.shape[0]
    cdf = np.empty(n, dtype=np.float64)

    if beta == 0.0:
        for i in range(n):
            cdf[i] = (i + 1) / n
    else:
        max_log_w = -beta * shifts[0]
        for i in range(1, n):
            log_w = -beta * shifts[i]
            if log_w > max_log_w:
                max_log_w = log_w

        total = 0.0
        for i in range(n):
            total += np.exp(-beta * shifts[i] - max_log_w)
            cdf[i] = total

        inv_total = 1.0 / total
        for i in range(n):
            cdf[i] *= inv_total
        cdf[n - 1] = 1.0

    for i in range(uniforms.shape[0]):
        out_idx[i] = _sample_cdf(cdf, uniforms[i])


# ---------------------------------------------------------------------------
# Batch MC kernels for free-energy estimation
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def batch_f_site_numba(
    h0: np.ndarray,
    phi: np.ndarray,
    grid: np.ndarray,
    js: np.ndarray,
    site_idx: np.ndarray,
) -> np.ndarray:
    """
    For each of n_mc samples s:
        f_site[s] = min_n [ phi(n) + sum_{t} min_{n'} (h0[site_idx[s,t], n'] - js[t]*n*n') ]

    Parameters
    ----------
    h0 : (M, G)
    phi : (G,)
    grid : (G,)
    js : (c,) float64   — couplings (negative by convention)
    site_idx : (n_mc, c) int64

    Returns
    -------
    f_site : (n_mc,) float64
    """
    n_mc = site_idx.shape[0]
    g = h0.shape[1]
    out = np.empty(n_mc, dtype=np.float64)
    slopes = -grid.copy()
    mh = np.empty(g, dtype=h0.dtype)
    ch = np.empty(g, dtype=h0.dtype)
    xs = np.empty(g, dtype=h0.dtype)
    accum = np.empty(g, dtype=h0.dtype)

    for s in range(n_mc):
        for i in range(g):
            accum[i] = phi[i]
        _accum_min_convs_indexed(h0, site_idx[s], js, grid, accum, slopes, mh, ch, xs)
        out[s] = _min_value(accum)

    return out


@njit(cache=True, fastmath=True)
def batch_site_argmin_numba(
    h0: np.ndarray,
    phi: np.ndarray,
    grid: np.ndarray,
    js: np.ndarray,
    site_idx: np.ndarray,
) -> np.ndarray:
    """
    Like batch_f_site_numba but returns the argmin index (abundance index)
    instead of the minimum value.  Used to sample the equilibrium abundance.

    Returns
    -------
    argmin_idx : (n_mc,) int64
    """
    n_mc = site_idx.shape[0]
    g = h0.shape[1]
    out = np.empty(n_mc, dtype=np.int64)
    slopes = -grid.copy()
    mh = np.empty(g, dtype=h0.dtype)
    ch = np.empty(g, dtype=h0.dtype)
    xs = np.empty(g, dtype=h0.dtype)
    accum = np.empty(g, dtype=h0.dtype)

    for s in range(n_mc):
        for i in range(g):
            accum[i] = phi[i]
        _accum_min_convs_indexed(h0, site_idx[s], js, grid, accum, slopes, mh, ch, xs)
        best_i = 0
        best_v = accum[0]
        for i in range(1, g):
            if accum[i] < best_v:
                best_v = accum[i]
                best_i = i
        out[s] = best_i

    return out


@njit(cache=True, fastmath=True)
def batch_f_edge_numba(
    h0: np.ndarray,
    grid: np.ndarray,
    edge_js: np.ndarray,
    edge_idx: np.ndarray,
) -> np.ndarray:
    """
    For each of n_mc edge samples (i0, i1):
        f_edge[s] = min_{n,n'} [ h0[i0,n] + h0[i1,n'] - edge_js[0]*n*n' ]

    This estimates the two-node edge contribution δΦ_edge.

    Parameters
    ----------
    h0 : (M, G)
    grid : (G,)
    edge_js : (1,) float64   — single edge coupling (negative by convention)
    edge_idx : (n_mc, 2) int64

    Returns
    -------
    f_edge : (n_mc,) float64
    """
    n_mc = edge_idx.shape[0]
    g = h0.shape[1]
    out = np.empty(n_mc, dtype=np.float64)
    slopes = -grid.copy()
    mh = np.empty(g, dtype=h0.dtype)
    ch = np.empty(g, dtype=h0.dtype)
    xs = np.empty(g, dtype=h0.dtype)
    accum = np.empty(g, dtype=h0.dtype)
    parent_idx = np.empty(1, dtype=np.int64)

    for s in range(n_mc):
        i0 = edge_idx[s, 0]
        i1 = edge_idx[s, 1]
        for i in range(g):
            accum[i] = h0[i0, i]
        parent_idx[0] = i1
        _accum_min_convs_indexed(h0, parent_idx, edge_js, grid, accum, slopes, mh, ch, xs)
        out[s] = _min_value(accum)

    return out
