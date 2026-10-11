"""Numerical order and coupling calculations using the package projection kernel."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import numpy as np
import pandas as pd
from numba import njit, prange
import numba
import paths
from sirsyn import seed_for, simulate
from sirsyn.geometry import adjacent_pairs
from sirsyn._kernels import dykstra as _dyk
RES = str(paths.SIMS)
I = J = 5
N = I * J
(PA, PB) = adjacent_pairs(I, J)
EA = np.array([5, 10, 15, 20], dtype=np.int64)
EB = np.array([1, 2, 3, 4], dtype=np.int64)
INT = np.array([i * 5 + j for i in range(1, 5) for j in range(1, 5)], dtype=np.int64)
W1 = np.ones(N)
WS1 = np.ones(16)
(TOL, SWEEPS) = (1e-13, 20000)

def set_threads(k=8):
    numba.set_num_threads(1)

@njit(cache=True)
def pava_dec(y, w):
    n = y.shape[0]
    vals = np.empty(n)
    wts = np.empty(n)
    cnt = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        vals[k] = y[i]
        wts[k] = w[i]
        cnt[k] = 1
        while k > 0 and vals[k - 1] < vals[k]:
            tw = wts[k - 1] + wts[k]
            vals[k - 1] = (wts[k - 1] * vals[k - 1] + wts[k] * vals[k]) / tw
            wts[k - 1] = tw
            cnt[k - 1] += cnt[k]
            k -= 1
        k += 1
    out = np.empty(n)
    p = 0
    for b in range(k):
        for c in range(cnt[b]):
            out[p] = vals[b]
            p += 1
    return out

@njit(cache=True)
def pipe(Y, wfit, wstat, pa, pb, tol, sweeps):
    ya = np.empty(4)
    wa = np.empty(4)
    yb = np.empty(4)
    wb = np.empty(4)
    for k in range(4):
        ya[k] = Y[(k + 1) * 5]
        wa[k] = wfit[(k + 1) * 5]
        yb[k] = Y[k + 1]
        wb[k] = wfit[k + 1]
    a = pava_dec(ya, wa)
    b = pava_dec(yb, wb)
    mu = _dyk(Y, wfit, pa, pb, sweeps, tol)
    eta = np.empty(16)
    tp = 0.0
    tm = 0.0
    sw = 0.0
    for i in range(4):
        for j in range(4):
            e = a[i] * b[j] - mu[(i + 1) * 5 + (j + 1)]
            eta[i * 4 + j] = e
            ws = wstat[i * 4 + j]
            if e > 0:
                tp += ws * e * e
            tm += ws * e
            sw += ws
    return (tp, tm / sw, eta, mu, a, b)

@njit(cache=True)
def pipe_pmedge(Y, wfit, wstat, pa, pb, tol, sweeps):
    mu = _dyk(Y, wfit, pa, pb, sweeps, tol)
    tp = 0.0
    tm = 0.0
    eta = np.empty(16)
    for i in range(4):
        for j in range(4):
            e = mu[(i + 1) * 5] * mu[j + 1] - mu[(i + 1) * 5 + (j + 1)]
            eta[i * 4 + j] = e
            if e > 0:
                tp += wstat[i * 4 + j] * e * e
            tm += wstat[i * 4 + j] * e
    return (tp, tm / wstat.sum(), eta)

@njit(parallel=False, cache=True)
def batch(Ym, wfit, wstat, pa, pb, tol, sweeps):
    n = Ym.shape[0]
    tp = np.empty(n)
    tm = np.empty(n)
    for k in prange(n):
        (a, b, _, _, _, _) = pipe(Ym[k], wfit, wstat, pa, pb, tol, sweeps)
        tp[k] = a
        tm[k] = b
    return (tp, tm)

@njit(parallel=False, cache=True)
def batch_eta(Ym, wfit, wstat, pa, pb, tol, sweeps):
    n = Ym.shape[0]
    out = np.empty((n, 16))
    for k in prange(n):
        (_, _, e, _, _, _) = pipe(Ym[k], wfit, wstat, pa, pb, tol, sweeps)
        out[k] = e
    return out

@njit(parallel=False, cache=True)
def _restricted(Ym, w, pa, pb, tol, sweeps):
    n = Ym.shape[0]
    out = np.empty_like(Ym)
    for k in prange(n):
        out[k] = _dyk(Ym[k], w, pa, pb, sweeps, tol)
    return out

def restricted_batch(Ym, pa, pb, w=None, tol=1e-12, sweeps=5000):
    w = W1 if w is None else np.ascontiguousarray(w, float)
    return _restricted(np.ascontiguousarray(Ym, float), w, np.ascontiguousarray(pa), np.ascontiguousarray(pb), tol, sweeps)

def run(Ym, wfit=None, wstat=None, tol=1e-12, sweeps=5000):
    wfit = W1 if wfit is None else np.ascontiguousarray(wfit, float)
    wstat = WS1 if wstat is None else np.ascontiguousarray(wstat, float)
    return batch(np.ascontiguousarray(Ym, float), wfit, wstat, PA, PB, tol, sweeps)

def hill_curve(E, h=1.5, ec50=1.0, span=2.0):
    d = simulate.dose_grid(4, ec50, span)
    return 1 - E * d ** h / (d ** h + 1)
BASES = {'weak': 0.3, 'moderate': 0.7, 'strong': 0.95, 'potent': 0.99}

def bliss_surface(ya, yb):
    return np.outer(ya, yb).ravel()

def base_curves(kind):
    y = hill_curve(BASES[kind])
    return (y.copy(), y.copy())

def with_eta(mu0, eta16):
    mu = mu0.copy()
    mu[INT] = mu0[INT] - np.asarray(eta16, float)
    return mu

def oneil_curve(column='pooled_sd', scale='identity_raw'):
    df = pd.read_csv(paths.INPUTS / 'sd_by_mean.csv')
    d = df[df.scale == scale].sort_values('mean_viability_raw')
    return (d.mean_viability_raw.to_numpy(), d[column].to_numpy())
(KV, KS) = oneil_curve()

def sd_oneil(m, kv=KV, ks=KS):
    return np.interp(m, kv, ks)

def draw_xi(rng, n, t_df=None):
    Z = rng.standard_normal((n, N))
    if t_df is not None:
        Z = rng.standard_t(t_df, (n, N)) / np.sqrt(t_df / (t_df - 2.0))
    return dict(Z=Z, Z2=rng.standard_normal((n, N)), D=rng.standard_normal((n, 1)), T3=rng.standard_t(3, (n, N)) / np.sqrt(3.0))

def gen(model, mu, xi, **kw):
    Z = xi['Z']
    if model == 'add_gauss':
        Y = mu + kw.get('s0', 0.05) * Z
    elif model == 'add_t3':
        Y = mu + kw.get('s0', 0.05) * xi['T3']
    elif model == 'mult_logn':
        s = kw.get('s', 0.15)
        Y = mu * np.exp(s * Z - s * s / 2)
    elif model == 'boxcox_half':
        s = kw.get('s', 0.1)
        Y = np.maximum(np.sqrt(np.maximum(mu, 0)) + 0.5 * s * Z, 0.0) ** 2
    elif model == 'addmult_floor':
        Y = np.maximum(mu + (0.01 + 0.12 * mu) * Z, 0.0)
    elif model == 'mult_plus_add':
        Y = mu * np.exp(0.12 * Z - 0.0072) + 0.02 * xi['Z2']
    elif model == 'add_clip':
        Y = np.clip(mu + 0.08 * Z, 0.0, 1.2)
    elif model == 'plate_shift':
        Y = mu + 0.05 * Z
        Y[:, INT] += kw.get('tau', 0.03) * xi['D']
    elif model == 'mult_plate':
        Y = mu * np.exp(0.1 * Z)
        Y[:, INT] *= np.exp(0.08 * xi['D'])
    elif model == 'hetero_oneil':
        Y = mu + sd_oneil(mu) * Z
    elif model == 'hetero_oneil_floor':
        sc = np.ones(N)
        r = kw.get('edge_rep', 1)
        sc[EA] = sc[EB] = 1.0 / np.sqrt(r)
        Y = np.maximum(mu + sd_oneil(mu) * sc * Z, 0.0)
    elif model == 'hetero_oneil_t3_floor':
        Y = np.maximum(mu + sd_oneil(mu) * xi['T3'], 0.0)
    else:
        raise ValueError(model)
    Y = np.array(Y, dtype=float)
    Y[:, 0] = 1.0
    return np.ascontiguousarray(Y)
ADMISSIBLE = ['add_gauss', 'add_t3', 'mult_logn', 'boxcox_half', 'addmult_floor', 'mult_plus_add', 'add_clip', 'plate_shift', 'mult_plate']
NOT_ADMISSIBLE = ['hetero_oneil', 'hetero_oneil_floor', 'hetero_oneil_t3_floor']

def upper_sets(a=4, b=4):
    out = []

    def rec(i, prev, ts):
        if i == a:
            U = [ii * b + jj for ii in range(a) for jj in range(ts[ii], b)]
            if U:
                out.append(tuple(U))
            return
        for t in range(0, prev + 1):
            rec(i + 1, t, ts + [t])
    rec(0, b, [])
    return out

def seed(*k):
    return seed_for(bytes([114, 53, 116]).decode('ascii'), *k)
