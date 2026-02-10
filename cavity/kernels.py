from __future__ import annotations

import os
import numpy as np

# ------------------ optional numba JIT ------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # no numba -> define a no-op decorator
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

# Numba extras if available
try:
    from numba import prange as _prange, set_num_threads as _set_num_threads
except Exception:
    def _prange(x): return range(x)
    def _set_num_threads(n): pass

# Safe default: 1 thread unless overridden by env
try:
    _n_threads_env = int(os.getenv("CAVITY_NUMBA_THREADS", "1"))
except Exception:
    _n_threads_env = 1
try:
    _set_num_threads(_n_threads_env)
except Exception:
    pass


@njit(cache=True, fastmath=True)
def _cht_build_mono(slopes, inter):
    """
    Build lower envelope for lines y = m*x + c given MONOTONE slopes (e.g., decreasing).
    Returns (mh, ch, xs) trimmed to hull size; xs[k] is the s where line k becomes optimal.
    """
    n = slopes.shape[0]
    mh = np.empty(n, dtype=np.float32)
    ch = np.empty(n, dtype=np.float32)
    xs = np.empty(n, dtype=np.float32)
    k = 0
    for i in range(n):
        mi = slopes[i]
        ci = inter[i]
        if k > 0 and mi == mh[k-1]:
            if ci >= ch[k-1]:
                continue
            else:
                k -= 1
        s = 0.0
        while k > 0:
            denom = (mh[k-1] - mi)
            if denom == 0.0:
                s = np.inf
            else:
                s = (ci - ch[k-1]) / denom
            if s <= xs[k-1]:
                k -= 1
            else:
                break
        xs[k] = -np.inf if k == 0 else s
        mh[k] = mi
        ch[k] = ci
        k += 1
    return mh[:k], ch[:k], xs[:k]


@njit(cache=True, fastmath=True)
def _cht_eval_mono(mh, ch, xs, svals, out):
    """
    Evaluate min_k (mh[k]*s + ch[k]) at non-decreasing svals into out.
    """
    K = mh.shape[0]
    ptr = 0
    for i in range(svals.shape[0]):
        s = svals[i]
        while ptr + 1 < K and xs[ptr+1] <= s:
            ptr += 1
        out[i] = mh[ptr] * s + ch[ptr]


@njit(cache=True, fastmath=True, parallel=True)
def _accum_min_convs_numba(P, js, nvec, grid, out_accum):
    """
    out_accum[g] += min_x ( P[p,x] - (js[p]*nvec[g]) * grid[x] ), for each parent p.
    Exact via convex hull trick and monotone-query scan. Structured with prange over p.
    With threads=1 (default via env CAVITY_NUMBA_THREADS) it's safe under multiprocessing.
    """
    m, G = P.shape
    slopes = -grid  # monotone if grid increases
    tmp = np.empty(G, dtype=np.float32)
    for p in _prange(m):
        mh, ch, xs = _cht_build_mono(slopes, P[p, :])
        j = js[p]
        svals = nvec * j
        _cht_eval_mono(mh, ch, xs, svals, tmp)
        for g in range(G):
            out_accum[g] += tmp[g]


def _accum_min_convs_python_bcast(P, js, nvec, grid, out_accum):
    """
    Vectorized fallback (fast for small/moderate G).
    """
    m, G = P.shape
    for p in range(m):
        j = js[p]
        term = P[p][None, :] - (nvec[:, None] * (j * grid)[None, :])  # (G, G)
        out_accum += term.min(axis=1)


_accum_min_convs = _accum_min_convs_numba if NUMBA_AVAILABLE else _accum_min_convs_python_bcast


def onsite_phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * x - x


@njit(cache=True, fastmath=True)
def _transform_min_y_nb(hy, J, x, y, out):
    """
    out[i] = min_j (hy[j] - J * x[i] * y[j])
    No allocations.
    """
    m = x.shape[0]
    for i in range(m):
        xi = x[i]
        best = 1.0e38
        for j in range(m):
            v = hy[j] - J * xi * y[j]
            if v < best:
                best = v
        out[i] = best


@njit(cache=True, fastmath=True)
def _update_one_nb(pool, phi, x, y, idx, Js, scratch, out_h, recenter_flag):
    """
    out_h = phi + sum_t transform_min_y(pool[idx[t]], Js[t])
    Optionally shift so min(out_h)=0 when recenter_flag is True.
    """
    m = x.shape[0]

    # out_h = phi
    for i in range(m):
        out_h[i] = phi[i]

    # add contributions
    for t in range(idx.shape[0]):
        hy = pool[idx[t]]
        _transform_min_y_nb(hy, Js[t], x, y, scratch)
        for i in range(m):
            out_h[i] += scratch[i]

    if recenter_flag:
        mn = out_h[0]
        for i in range(1, m):
            v = out_h[i]
            if v < mn:
                mn = v
        for i in range(m):
            out_h[i] -= mn
