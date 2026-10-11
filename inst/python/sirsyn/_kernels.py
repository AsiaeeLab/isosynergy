"""Numba kernels: nonincreasing 1D and 2D isotonic fits and per-plate statistics.

Row and column zero are the single agents, the control is one, and doses
increase with index. Flat indices use i * J + j. The 2D projection uses cyclic
pairwise projections (5000 sweeps, tolerance 1e-12) on observed-well order pairs."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

N_SWEEPS = 5000
TOL = 1e-12
MFLOOR = 0.02  # floor of the viability in the proportional SD function


@njit(cache=True)
def pava_dec(y):
    """Nonincreasing isotonic regression with equal weights (pool adjacent violators)."""
    n = y.shape[0]
    val = np.empty(n)
    wt = np.empty(n)
    cnt = np.empty(n, np.int64)
    k = -1
    for i in range(n):
        k += 1
        val[k] = y[i]
        wt[k] = 1.0
        cnt[k] = 1
        while k > 0 and val[k - 1] < val[k]:
            tot = wt[k - 1] + wt[k]
            val[k - 1] = (wt[k - 1] * val[k - 1] + wt[k] * val[k]) / tot
            wt[k - 1] = tot
            cnt[k - 1] += cnt[k]
            k -= 1
    out = np.empty(n)
    p = 0
    for b in range(k + 1):
        for c in range(cnt[b]):
            out[p] = val[b]
            p += 1
    return out


@njit(cache=True)
def dykstra(z, w, pa, pb, n_sweeps, tol):
    """Weighted projection onto {theta : theta[pa[k]] >= theta[pb[k]] for all k} by cyclic pairwise
    Dykstra steps."""
    npair = pa.shape[0]
    theta = z.copy()
    ca = np.zeros(npair)
    cb = np.zeros(npair)
    for _ in range(n_sweeps):
        mm = 0.0
        for k in range(npair):
            a = pa[k]
            b = pb[k]
            ya = theta[a] + ca[k]
            yb = theta[b] + cb[k]
            if ya >= yb:
                na = ya
                nb = yb
            else:
                wa = w[a]
                wb = w[b]
                mval = (wa * ya + wb * yb) / (wa + wb)
                na = mval
                nb = mval
            d = abs(na - theta[a]) + abs(nb - theta[b])
            if d > mm:
                mm = d
            ca[k] = ya - na
            cb[k] = yb - nb
            theta[a] = na
            theta[b] = nb
        if mm < tol:
            break
    return theta


@njit(cache=True)
def sdfun(m, sdk, sdp, skv, sks):
    """Working noise SD at mean viability m: kind 0 constant sdp; 1 proportional sdp * max(m, 0.02);
    2 table interpolation (flat outside the knots)."""
    if sdk == 0:
        return sdp
    if sdk == 1:
        return sdp * max(m, 0.02)
    return np.interp(m, skv, sks)


@njit(cache=True)
def iso_fit(Y, obs, pa, pb):
    """1D fits a_hat, b_hat of the single-agent wells and the 2D restricted-order isotonic fit of the
    plate. Missing combination wells (obs False) get the plug-in Bliss value a_hat_i b_hat_j as a
    placeholder; they are in no covering pair, so the placeholder never influences the fit."""
    I, J = Y.shape
    a = pava_dec(Y[1:, 0].copy())
    b = pava_dec(Y[0, 1:].copy())
    z = Y.ravel().copy()
    for i in range(1, I):
        for j in range(1, J):
            k = i * J + j
            if not obs[k]:
                z[k] = a[i - 1] * b[j - 1]
    th = dykstra(z, np.ones(I * J), pa, pb, N_SWEEPS, TOL)
    return a, b, th


@njit(cache=True)
def plate_stats(Y, obs, pa, pb, sdk, sdp, skv, sks, cells, cptr, out):
    """Statistics of one plate into out (length 2 + nR):
      out[0]      T_std = sum over observed combination cells of ((a_i b_j - P_M(Y)_ij) / s(a_i b_j))_+^2
      out[1]      raw mean Bliss excess, mean over observed combination cells of Y_i0 Y_0j - Y_ij
      out[2 + q]  raw-well region sum of region q, sum over its observed cells of Y_i0 Y_0j - Y_ij
    """
    I, J = Y.shape
    a, b, th = iso_fit(Y, obs, pa, pb)
    t_std = 0.0
    m_raw = 0.0
    nc = 0
    D = np.zeros(I * J)
    for i in range(1, I):
        for j in range(1, J):
            k = i * J + j
            if not obs[k]:
                continue
            r = a[i - 1] * b[j - 1]
            s = sdfun(r, sdk, sdp, skv, sks)
            d = r - th[k]
            if d > 0:
                t_std += (d / s) * (d / s)
            dr = Y[i, 0] * Y[0, j] - Y[i, j]
            D[k] = dr
            m_raw += dr
            nc += 1
    out[0] = t_std
    out[1] = m_raw / max(nc, 1)
    nR = cptr.shape[0] - 1
    for q in range(nR):
        u = 0.0
        for t in range(cptr[q], cptr[q + 1]):
            u += D[cells[t]]
        out[2 + q] = u


@njit(cache=True)
def stats_many(Ys, obs, pa, pb, sdk, sdp, skv, sks, cells, cptr):
    n = Ys.shape[0]
    nR = cptr.shape[0] - 1
    out = np.empty((n, 2 + nR))
    for t in range(n):
        plate_stats(Ys[t], obs, pa, pb, sdk, sdp, skv, sks, cells, cptr, out[t])
    return out


@njit(cache=True)
def tstd_many(Ys, obs, pa, pb, sdk, sdp, skv, sks):
    """T_std alone for a stack of plates (used by the fragility checks)."""
    n = Ys.shape[0]
    out = np.empty(n)
    one = np.empty(2)
    cells = np.zeros(0, np.int64)
    cptr = np.zeros(1, np.int64)
    for t in range(n):
        plate_stats(Ys[t], obs, pa, pb, sdk, sdp, skv, sks, cells, cptr, one)
        out[t] = one[0]
    return out
