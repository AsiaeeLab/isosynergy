"""Confidence bands and Monte Carlo kernels for the Berger–Boos comparison."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
import paths
from sirsyn import seed_for
from sirsyn import simulate
from sirsyn.geometry import adjacent_pairs
from sirsyn._kernels import dykstra as _DYK
RES = str(paths.INPUTS)
(TOL, SWEEPS) = (1e-10, 5000)
(PLO, PHI) = (0.0, 1.2)
S0_ADD = 0.08
SL_LOGN = 0.12
CV_LOGN = float(np.sqrt(np.exp(SL_LOGN ** 2) - 1.0))
MFLOOR_LOGN = 0.01
KIND = {'add': 0, 'logn': 1, 'oneil': 2}
NOISES = ['add', 'logn', 'oneil']

def _oneil():
    df = pd.read_csv(os.path.join(RES, 'sd_by_mean.csv'))
    d = df[df.scale == 'identity_raw'].sort_values('mean_viability_raw')
    return (np.ascontiguousarray(d.mean_viability_raw.to_numpy(float)), np.ascontiguousarray(d.pooled_sd.to_numpy(float)))
(KV, KS) = _oneil()

def seed(*k):
    return seed_for(bytes([114, 54, 98]).decode('ascii'), *k)

class Geo:

    def __init__(self, I, J):
        (pa, pb) = adjacent_pairs(I, J)
        (self.I, self.J, self.N) = (I, J, I * J)
        self.pa = np.ascontiguousarray(pa)
        self.pb = np.ascontiguousarray(pb)
        self.EA = np.array([i * J for i in range(1, I)], dtype=np.int64)
        self.EB = np.array([j for j in range(1, J)], dtype=np.int64)
        self.INT = np.array([i * J + j for i in range(1, I) for j in range(1, J)], dtype=np.int64)
        self.w = np.ones(I * J)
G5 = Geo(5, 5)

@njit(cache=True)
def interp1(x, xp, fp):
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    k = 0
    while xp[k + 1] < x:
        k += 1
    t = (x - xp[k]) / (xp[k + 1] - xp[k])
    return fp[k] + t * (fp[k + 1] - fp[k])

@njit(cache=True)
def sd_well(m, kind, kv, ks):
    if kind == 0:
        return S0_ADD
    if kind == 1:
        return CV_LOGN * max(m, MFLOOR_LOGN)
    return interp1(min(max(m, 0.0), 1.2), kv, ks)

@njit(cache=True)
def pava_dec(y):
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
def t_std(Y, I, J, pa, pb, w, kind, kv, ks):
    ya = np.empty(I - 1)
    yb = np.empty(J - 1)
    for i in range(1, I):
        ya[i - 1] = Y[i * J]
    for j in range(1, J):
        yb[j - 1] = Y[j]
    a = pava_dec(ya)
    b = pava_dec(yb)
    mu = _DYK(Y, w, pa, pb, SWEEPS, TOL)
    t = 0.0
    for i in range(1, I):
        for j in range(1, J):
            p = a[i - 1] * b[j - 1]
            s = sd_well(min(max(p, 0.0), 1.2), kind, kv, ks)
            e = (p - mu[i * J + j]) / s
            if e > 0:
                t += e * e
    return t

@njit(cache=True)
def edge_value(m, kind, ms, zrow, kv, ks):
    if kind == 1:
        acc = 0.0
        for k in range(ms):
            acc += np.exp(SL_LOGN * zrow[k] - 0.5 * SL_LOGN * SL_LOGN)
        return m * acc / ms
    s = sd_well(m, kind, kv, ks)
    v = m + s * zrow[0] / np.sqrt(ms)
    if kind == 2 and v < 0.0:
        v = 0.0
    return v

@njit(cache=True)
def int_value(m, kind, z, kv, ks):
    if kind == 0:
        return m + S0_ADD * z
    if kind == 1:
        return m * np.exp(SL_LOGN * z - 0.5 * SL_LOGN * SL_LOGN)
    v = m + interp1(min(max(m, 0.0), 1.2), kv, ks) * z
    return v if v > 0.0 else 0.0

@njit(cache=True)
def make_Y(a, b, mint, I, J, ms, kind, ZA, ZB, ZI, kv, ks):
    Y = np.empty(I * J)
    Y[0] = 1.0
    for i in range(1, I):
        Y[i * J] = edge_value(a[i - 1], kind, ms, ZA[i - 1], kv, ks)
    for j in range(1, J):
        Y[j] = edge_value(b[j - 1], kind, ms, ZB[j - 1], kv, ks)
    q = 0
    for i in range(1, I):
        for j in range(1, J):
            Y[i * J + j] = int_value(mint[q], kind, ZI[q], kv, ks)
            q += 1
    return Y

@njit(cache=True)
def gen_T(a, b, mint, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    n = ZI.shape[0]
    out = np.empty(n)
    for k in range(n):
        Y = make_Y(a, b, mint, I, J, ms, kind, ZA[k], ZB[k], ZI[k], kv, ks)
        out[k] = t_std(Y, I, J, pa, pb, w, kind, kv, ks)
    return out

@njit(cache=True)
def bliss_int(a, b):
    (I1, J1) = (a.shape[0], b.shape[0])
    m = np.empty(I1 * J1)
    q = 0
    for i in range(I1):
        for j in range(J1):
            m[q] = a[i] * b[j]
            q += 1
    return m

@njit(cache=True)
def ref_T_bliss(a, b, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    return gen_T(a, b, bliss_int(a, b), I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks)

@njit(cache=True)
def count_pairs(CA, CB, pairs, tobs, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    npair = pairs.shape[0]
    n = ZI.shape[0]
    thr = tobs - 1e-09 * max(1.0, abs(tobs))
    out = np.zeros(npair, np.int64)
    for q in range(npair):
        a = CA[pairs[q, 0]]
        b = CB[pairs[q, 1]]
        mint = bliss_int(a, b)
        c = 0
        for k in range(n):
            Y = make_Y(a, b, mint, I, J, ms, kind, ZA[k], ZB[k], ZI[k], kv, ks)
            if t_std(Y, I, J, pa, pb, w, kind, kv, ks) >= thr:
                c += 1
        out[q] = c
    return out

@njit(cache=True)
def count_one(a, b, tobs, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    n = ZI.shape[0]
    thr = tobs - 1e-09 * max(1.0, abs(tobs))
    mint = bliss_int(a, b)
    c = 0
    for k in range(n):
        Y = make_Y(a, b, mint, I, J, ms, kind, ZA[k], ZB[k], ZI[k], kv, ks)
        if t_std(Y, I, J, pa, pb, w, kind, kv, ks) >= thr:
            c += 1
    return c

@njit(cache=True)
def count_one_es(a, b, tobs, best, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    n = ZI.shape[0]
    thr = tobs - 1e-09 * max(1.0, abs(tobs))
    mint = bliss_int(a, b)
    c = 0
    for k in range(n):
        if c + (n - k) <= best:
            return -1
        Y = make_Y(a, b, mint, I, J, ms, kind, ZA[k], ZB[k], ZI[k], kv, ks)
        if t_std(Y, I, J, pa, pb, w, kind, kv, ks) >= thr:
            c += 1
    return c

@njit(cache=True)
def count_pairs_es(CA, CB, pairs, tobs, best0, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks):
    npair = pairs.shape[0]
    out = np.full(npair, -1, np.int64)
    best = best0
    for q in range(npair):
        c = count_one_es(CA[pairs[q, 0]], CB[pairs[q, 1]], tobs, best, I, J, ms, kind, ZA, ZB, ZI, pa, pb, w, kv, ks)
        out[q] = c
        if c > best:
            best = c
    return out

@njit(cache=True)
def count_conditional(Yobs, ell, tobs, I, J, kind, ZI, pa, pb, w, kv, ks):
    n = ZI.shape[0]
    thr = tobs - 1e-09 * max(1.0, abs(tobs))
    c = 0
    Y = Yobs.copy()
    for k in range(n):
        q = 0
        for i in range(1, I):
            for j in range(1, J):
                Y[i * J + j] = int_value(ell[q], kind, ZI[k, q], kv, ks)
                q += 1
        if t_std(Y, I, J, pa, pb, w, kind, kv, ks) >= thr:
            c += 1
    return c

def draw_noise(rng, n, I, J, ms, noise):
    r = ms if noise == 'logn' else 1
    ZA = rng.standard_normal((n, I - 1, r))
    ZB = rng.standard_normal((n, J - 1, r))
    ZI = rng.standard_normal((n, (I - 1) * (J - 1)))
    return (ZA, ZB, ZI)

def gen_data(rng, n, a, b, mint, I, J, ms, noise):
    (ZA, ZB, ZI) = draw_noise(rng, n, I, J, ms, noise)
    kind = KIND[noise]
    Y = np.empty((n, I * J))
    for k in range(n):
        Y[k] = make_Y(a, b, mint, I, J, ms, kind, ZA[k], ZB[k], ZI[k], KV, KS)
    return Y

def T_batch(Y, g, noise):
    kind = KIND[noise]
    return np.array([t_std(np.ascontiguousarray(y), g.I, g.J, g.pa, g.pb, g.w, kind, KV, KS) for y in Y])

@njit(cache=True)
def T_batch_nb(Ym, I, J, pa, pb, w, kind, kv, ks):
    out = np.empty(Ym.shape[0])
    for k in range(Ym.shape[0]):
        out[k] = t_std(Ym[k].copy(), I, J, pa, pb, w, kind, kv, ks)
    return out
_QCACHE = {}

def logn_mean_quantiles(ms, probs, sl=SL_LOGN):
    probs = np.atleast_1d(np.asarray(probs, float))
    key = (ms, sl, tuple(np.round(probs, 15)))
    if key in _QCACHE:
        return _QCACHE[key]
    if ms == 1:
        q = np.exp(sl * stats.norm.ppf(probs) - 0.5 * sl * sl)
        _QCACHE[key] = q
        return q
    h = 2e-05 if ms <= 16 else 0.0001
    xmax = np.exp(12 * sl) if ms <= 16 else np.exp(8 * sl)
    x = np.arange(0, xmax + h, h)
    f = np.zeros_like(x)
    pos = x > 0
    lx = np.log(x[pos])
    f[pos] = stats.norm.pdf((lx + 0.5 * sl * sl) / sl) / (sl * x[pos])
    f /= f.sum() * h
    L = int(2 ** np.ceil(np.log2(ms * len(x) + 1)))
    F = np.fft.rfft(f * h, L)
    g = np.fft.irfft(F ** ms, L)[:ms * (len(x) - 1) + 1]
    g = np.maximum(g, 0.0)
    g /= g.sum()
    cdf = np.cumsum(g)
    sgrid = np.arange(len(g)) * h
    q = np.interp(probs, cdf - 0.5 * g, sgrid) / ms
    _QCACHE[key] = q
    return q

def sidak_gamma(beta, n_wells):
    return 1.0 - (1.0 - beta) ** (1.0 / n_wells)

@njit(cache=True)
def _seg_interval(y, m0, m1, s0, s1, c, sq):
    (lo, hi) = (m0, m1)
    dm = m1 - m0
    for sign in (-1.0, 1.0):
        h0 = m0 + sign * c * s0 / sq
        h1 = m1 + sign * c * s1 / sq
        sl = (h1 - h0) / dm if dm > 0 else 0.0
        if sign < 0:
            if sl > 0:
                hi = min(hi, m0 + (y - h0) / sl)
            elif sl < 0:
                lo = max(lo, m0 + (y - h0) / sl)
            elif h0 > y:
                return (1.0, 0.0)
        elif sl > 0:
            lo = max(lo, m0 + (y - h0) / sl)
        elif sl < 0:
            hi = min(hi, m0 + (y - h0) / sl)
        elif h0 < y:
            return (1.0, 0.0)
    return (lo, hi)

@njit(cache=True)
def ci_gaussform(y, kind, ms, c, kv, ks, knots):
    sq = np.sqrt(ms)
    lo = 1000000000.0
    hi = -1000000000.0
    for k in range(knots.shape[0] - 1):
        m0 = knots[k]
        m1 = knots[k + 1]
        s0 = sd_well(m0, kind, kv, ks)
        s1 = sd_well(m1, kind, kv, ks)
        (l, h) = _seg_interval(y, m0, m1, s0, s1, c, sq)
        if l <= h:
            if l < lo:
                lo = l
            if h > hi:
                hi = h
    return (lo, hi)

def knots_for(noise):
    if noise == 'oneil':
        k = np.concatenate([[PLO], KV[(KV > PLO) & (KV < PHI)], [PHI]])
    elif noise == 'logn':
        k = np.concatenate([[PLO], [MFLOOR_LOGN], [PHI]])
    else:
        k = np.array([PLO, PHI])
    return np.ascontiguousarray(np.unique(k))

class BandSpec:

    def __init__(self, noise, ms, beta, n_wells, method='exact'):
        (self.noise, self.ms, self.beta, self.n_wells, self.method) = (noise, ms, beta, n_wells, method)
        self.kind = KIND[noise]
        self.gamma = sidak_gamma(beta, n_wells)
        self.knots = knots_for(noise)
        if method == 'plugin_sd':
            self.c = stats.norm.ppf(1 - beta / 16)
        elif method == 'lower':
            self.c = stats.norm.ppf(1 - self.gamma)
        else:
            self.c = stats.norm.ppf(1 - self.gamma / 2)
        if noise == 'logn' and method in ('exact', 'lower'):
            if method == 'exact':
                (self.qlo, self.qhi) = logn_mean_quantiles(ms, [self.gamma / 2, 1 - self.gamma / 2])
            else:
                (self.qlo, self.qhi) = (0.0, logn_mean_quantiles(ms, [1 - self.gamma])[0])

    def intervals(self, y):
        y = np.asarray(y, float)
        lo = np.empty_like(y)
        hi = np.empty_like(y)
        (flat_y, flat_lo, flat_hi) = (y.ravel(), lo.ravel(), hi.ravel())
        if self.method == 'plugin_sd':
            s = _sd_many(flat_y, self.kind, KV, KS)
            flat_lo[:] = flat_y - self.c * s / np.sqrt(self.ms)
            flat_hi[:] = flat_y + self.c * s / np.sqrt(self.ms)
        elif self.noise == 'logn' and self.method == 'exact':
            flat_lo[:] = flat_y / self.qhi
            flat_hi[:] = flat_y / self.qlo
        elif self.noise == 'logn' and self.method == 'lower':
            flat_lo[:] = flat_y / self.qhi
            flat_hi[:] = np.inf
        elif self.method == 'lower':
            _lower_nb(flat_y, flat_lo, self.kind, self.ms, self.c, KV, KS, self.knots)
            flat_hi[:] = np.inf
        else:
            _ci_many(flat_y, flat_lo, flat_hi, self.kind, self.ms, self.c, KV, KS, self.knots)
        lo = flat_lo.reshape(y.shape)
        hi = flat_hi.reshape(y.shape)
        lo = np.maximum(lo, PLO)
        hi = np.minimum(hi, PHI)
        return (lo, hi)

    def band(self, y):
        (lo, hi) = self.intervals(y)
        U = np.minimum.accumulate(hi, axis=-1)
        L = np.maximum.accumulate(lo[..., ::-1], axis=-1)[..., ::-1]
        ok = np.all(L <= U + 1e-15, axis=-1)
        return (L, U, ok)

@njit(cache=True)
def _sd_many(y, kind, kv, ks):
    out = np.empty(y.shape[0])
    for k in range(y.shape[0]):
        out[k] = sd_well(min(max(y[k], 0.0), 1.2), kind, kv, ks)
    return out

@njit(cache=True)
def _ci_many(y, lo, hi, kind, ms, c, kv, ks, knots):
    for k in range(y.shape[0]):
        (l, h) = ci_gaussform(y[k], kind, ms, c, kv, ks, knots)
        lo[k] = l
        hi[k] = h

@njit(cache=True)
def _lower_nb(y, lo, kind, ms, c, kv, ks, knots):
    sq = np.sqrt(ms)
    for q in range(y.shape[0]):
        yy = y[q]
        best = 1000000000.0
        for k in range(knots.shape[0] - 1):
            m0 = knots[k]
            m1 = knots[k + 1]
            h0 = m0 + c * sd_well(m0, kind, kv, ks) / sq
            h1 = m1 + c * sd_well(m1, kind, kv, ks) / sq
            if h1 < yy:
                continue
            if h0 >= yy:
                cand = m0
            else:
                cand = m0 + (yy - h0) * (m1 - m0) / (h1 - h0)
            if cand < best:
                best = cand
        lo[q] = best

def clip_level(v, L, U):
    return np.minimum(np.maximum(v, L), U)

def candidates(ybar, L, U, rng, n_flat=6, n_rand=8):
    n = len(L)
    out = [U.copy(), L.copy(), clip_level(pava_dec(np.ascontiguousarray(ybar, float)), L, U)]
    lev = np.linspace(L.min(), U.max(), n_flat + 2)[1:-1]
    for v in lev:
        out.append(clip_level(v, L, U))
    for k in range(1, n):
        c = L.copy()
        c[:k] = U[:k]
        out.append(c)
    for _ in range(n_rand):
        t = rng.uniform(size=n)
        v = L + t * (U - L)
        out.append(clip_level(pava_dec(v), L, U))
    for _ in range(n_rand):
        t = rng.uniform(size=n)
        v = L + t * (U - L)
        out.append(np.maximum(L, np.minimum.accumulate(v)))
    C = np.array(out)
    assert np.all(np.diff(C, axis=1) <= 1e-12) and np.all(C >= L - 1e-12) and np.all(C <= U + 1e-12)
    return np.ascontiguousarray(C)
BASES = {'weak': 0.3, 'moderate': 0.7, 'strong': 0.95}

def hill_curve(E, n=4, h=1.5, ec50=1.0, span=2.0):
    d = simulate.dose_grid(n, ec50, span)
    return 1 - E * d ** h / (d ** h + 1)

def base_ab(base, I=5):
    y = hill_curve(BASES[base], n=I - 1)
    return (y[1:].copy(), y[1:].copy())

def pattern(kind, I=5, J=5):
    if kind in ('broad', 'highdose', 'corner'):
        return simulate.interaction_pattern(I, J, kind).reshape(I, J)[1:, 1:].ravel()
    if kind == 'saturating':
        (ii, jj) = np.meshgrid(np.arange(1, I), np.arange(1, J), indexing='ij')
        xy = ii / (I - 1) * (jj / (J - 1))
        p = 4 * xy * (1 - xy)
        return (p / p.max()).ravel()
    raise ValueError(kind)

def truth_int(base, cond, amp, I=5):
    (a, b) = base_ab(base, I)
    mB = np.outer(a, b).ravel()
    if cond == 'bliss':
        return mB
    pk = {'broad': 'broad', 'high': 'highdose', 'corner': 'corner', 'sat': 'saturating'}[cond.split('_')[1]]
    p = pattern(pk, I, I)
    if cond.startswith('ant'):
        return np.minimum(mB * np.exp(amp * p), np.maximum(1.2, mB))
    return mB * np.exp(-amp * p)

def wilson(k, n, z=1.959964):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (cen - half, cen + half)

def fmt_rate(k, n, d=3):
    (lo, hi) = wilson(k, n)
    return f'{k / n:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'
