"""Common-draw reference kernels for the antagonistic-surface search."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import numpy as np
from numba import njit
import paths
from sirsyn import seed_for, simulate
from sirsyn.geometry import adjacent_pairs
from sirsyn._kernels import dykstra as _DYK
RES = str(paths.SIMS)
(NSWEEP, TOL) = (5000, 1e-12)
MFLOOR = 0.02
NS = 6
NSTD = 200
SNAMES = ['std_iso', 'std_raw', 'std_rawref', 'plus_iso', 'plus_raw', 'mean_raw']
from model_curves import VIABILITY_MEAN as KV, VIABILITY_SD as KS

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
def ref_mean(a, b):
    (I, J) = (a.shape[0] + 1, b.shape[0] + 1)
    M = np.empty((I, J))
    M[0, 0] = 1.0
    for i in range(1, I):
        M[i, 0] = a[i - 1]
    for j in range(1, J):
        M[0, j] = b[j - 1]
    for i in range(1, I):
        for j in range(1, J):
            M[i, j] = a[i - 1] * b[j - 1]
    return M

@njit(cache=True)
def gen(mu, kind, par, e, kv, ks):
    (I, J) = mu.shape
    Y = np.empty((I, J))
    for i in range(I):
        for j in range(J):
            m = mu[i, j]
            if kind == 0:
                Y[i, j] = m + par * e[i, j]
            elif kind == 1:
                Y[i, j] = max(m, 1e-06) * np.exp(par * e[i, j] - 0.5 * par * par)
            elif kind == 2:
                Y[i, j] = m + np.interp(m, kv, ks) * e[i, j]
            else:
                Y[i, j] = m + par * max(m, 0.0) * e[i, j]
    Y[0, 0] = 1.0
    return Y

def wilson(k, n, z=1.959964):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (cen - half, cen + half)

def fmt_rate(k, n, digits=3):
    (lo, hi) = wilson(k, n)
    return f'{k / n:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'
CV_LN = float(np.sqrt(np.exp(0.12 ** 2) - 1.0))
TRUTH = {'A_gauss': (0, 0.08, 'n'), 'B_lnorm': (1, 0.12, 'n'), 'C_hetero': (2, 1.0, 'n'), 'D_t4': (0, 0.08, 't')}
BOOT = {'gauss_add': (0, 0.08, 'n', 0, 0.08), 'lnorm': (1, 0.12, 'n', 1, CV_LN), 'hetero': (2, 1.0, 'n', 2, 0.0), 't4_add': (0, 0.08, 't', 0, 0.08), 'gauss_cv': (3, CV_LN, 'n', 1, CV_LN)}
COMBOS = [('A_gauss', 'gauss_add', 'matched'), ('B_lnorm', 'lnorm', 'matched'), ('B_lnorm', 'gauss_cv', 'misspecified: Gaussian shape, correct variance function'), ('C_hetero', 'hetero', 'matched'), ('C_hetero', 'gauss_add', 'misspecified: constant SD 0.08'), ('D_t4', 't4_add', 'matched'), ('D_t4', 'gauss_add', 'misspecified: Gaussian tails, correct SD')]

def std_noise(rng, code, shape):
    if code == 'n':
        return rng.standard_normal(shape)
    return rng.standard_t(4, shape) / np.sqrt(2.0)

class Regions:

    def __init__(self, n):
        m = n - 1
        (self.n, self.m) = (n, m)
        (sets, types) = ([], [])
        seen = {}
        for i0 in range(1, m + 1):
            for j0 in range(1, m + 1):
                s = tuple(sorted(((a, b) for a in range(i0, m + 1) for b in range(j0, m + 1))))
                seen.setdefault(s, set()).add('rect')
        for i in range(1, m):
            for j in range(1, m):
                s = tuple(sorted([(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]))
                seen.setdefault(s, set()).add('block')
        for s in sorted(seen, key=lambda s: (len(s), s)):
            sets.append(list(s))
            types.append(seen[s])
        (self.sets, self.types) = (sets, types)
        self.nR = len(sets)
        (cells, cptr) = ([], [0])
        for s in sets:
            cells += [i * n + j for (i, j) in s]
            cptr.append(len(cells))
        self.cells = np.array(cells, dtype=np.int64)
        self.cptr = np.array(cptr, dtype=np.int64)
        self.size = np.array([len(s) for s in sets])
        self.M = np.zeros((self.nR, m * m), dtype=bool)
        for (r, s) in enumerate(sets):
            for (i, j) in s:
                self.M[r, (i - 1) * m + (j - 1)] = True

    def label(self, r):
        return ';'.join((f'{i}{j}' for (i, j) in self.sets[r]))
_REG = {}

def regions(n):
    if n not in _REG:
        _REG[n] = Regions(n)
    return _REG[n]

def geom(I, J):
    pa, pb = adjacent_pairs(I, J)
    return pa, pb, np.ones(I * J)


@njit(cache=True)
def sdfun(m, sdk, sdp, skv, sks):
    if sdk == 0:
        return sdp
    if sdk == 1:
        return sdp * max(m, 0.02)
    return np.interp(m, skv, sks)

@njit(cache=True)
def plate_stats(Y, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr, out):
    (I, J) = Y.shape
    nR = cptr.shape[0] - 1
    a = pava_dec(Y[1:, 0].copy())
    b = pava_dec(Y[0, 1:].copy())
    th = _DYK(Y.ravel().copy(), w1, pa, pb, 5000, 1e-12)
    s_iso = 0.0
    s_raw = 0.0
    s_rr = 0.0
    p_iso = 0.0
    p_raw = 0.0
    m_raw = 0.0
    ref = np.empty(I * J)
    sd = np.empty(I * J)
    for i in range(1, I):
        for j in range(1, J):
            k = i * J + j
            r = a[i - 1] * b[j - 1]
            s = sdfun(r, sdk, sdp, skv, sks)
            ref[k] = r
            sd[k] = s
            y = Y[i, j]
            d = r - th[k]
            if d > 0:
                p_iso += d * d
                s_iso += d / s * (d / s)
            d = r - y
            if d > 0:
                p_raw += d * d
                s_raw += d / s * (d / s)
            rr = Y[i, 0] * Y[0, j]
            d = rr - y
            m_raw += d
            if d > 0:
                s2 = sdfun(rr, sdk, sdp, skv, sks)
                s_rr += d / s2 * (d / s2)
    out[0] = s_iso
    out[1] = s_raw
    out[2] = s_rr
    out[3] = p_iso
    out[4] = p_raw
    out[5] = m_raw / ((I - 1) * (J - 1))
    yf = Y.ravel()
    for q in range(nR):
        u = 0.0
        uw = 0.0
        for t in range(cptr[q], cptr[q + 1]):
            c = cells[t]
            d = ref[c] - yf[c]
            u += d
            uw += d / sd[c]
        out[6 + q] = u
        out[6 + nR + q] = uw

@njit(cache=True)
def gen_design(mu, kind, par, e, gkv, gks, design_p):
    Y = gen(mu, kind, par, e, gkv, gks)
    if design_p:
        (I, J) = mu.shape
        for i in range(I):
            Y[i, 0] = mu[i, 0]
        for j in range(J):
            Y[0, j] = mu[0, j]
        Y[0, 0] = 1.0
    return Y

@njit(cache=True)
def boot_table(Y, bk, bp, E, gkv, gks, design_p, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr):
    (I, J) = Y.shape
    nR = cptr.shape[0] - 1
    nb = E.shape[0]
    T = np.empty((nb + 1, 6 + 2 * nR))
    plate_stats(Y, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr, T[0])
    a = pava_dec(Y[1:, 0].copy())
    b = pava_dec(Y[0, 1:].copy())
    mu0 = ref_mean(a, b)
    for bb in range(nb):
        Ys = gen_design(mu0, bk, bp, E[bb], gkv, gks, design_p)
        plate_stats(Ys, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr, T[bb + 1])
    return T

@njit(cache=True)
def std_moments(T, off, nR, B):
    n = T.shape[0]
    (r0, r1) = (B + 1, n) if n - 1 - B >= 2 else (1, B + 1)
    k = r1 - r0
    m = np.empty(nR)
    s = np.empty(nR)
    for q in range(nR):
        mm = 0.0
        for bb in range(r0, r1):
            mm += T[bb, off + q]
        mm /= k
        ss = 0.0
        for bb in range(r0, r1):
            d = T[bb, off + q] - mm
            ss += d * d
        m[q] = mm
        s[q] = max(np.sqrt(ss / (k - 1)), 1e-14)
    return (m, s)

@njit(cache=True)
def _ranks_p(v):
    n = v.shape[0]
    order = np.argsort(-v)
    out = np.empty(n)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and v[order[j + 1]] >= v[order[k]] - 1e-12 * max(1.0, abs(v[order[k]])):
            j += 1
        for q in range(k, j + 1):
            out[order[q]] = (j + 1.0) / n
        k = j + 1
    return out

@njit(cache=True)
def comb_p(T, nR, B):
    (m, s) = std_moments(T, 6, nR, B)
    zm = np.empty(B + 1)
    for bb in range(B + 1):
        mx = -1e+300
        for q in range(nR):
            z = (T[bb, 6 + q] - m[q]) / s[q]
            if z > mx:
                mx = z
        zm[bb] = mx
    pa = _ranks_p(T[:B + 1, 0].copy())
    pc = _ranks_p(zm)
    mn = np.minimum(pa, pc)
    c = 0
    for bb in range(B + 1):
        if mn[bb] <= mn[0] + 1e-12:
            c += 1
    return c / (B + 1.0)

@njit(cache=True)
def pvals_from_table(T, nR, alpha_claim, B):
    p = np.empty(6 + 3)
    p[8] = comb_p(T, nR, B)
    for k in range(6):
        t0 = T[0, k]
        thr = t0 - 1e-12 * max(1.0, abs(t0))
        c = 0
        for bb in range(1, B + 1):
            if T[bb, k] >= thr:
                c += 1
        p[k] = (1.0 + c) / (B + 1.0)
    masks = np.zeros(2, np.int64)
    zmax = np.empty(2)
    Z = np.empty((B + 1, nR))
    for v in range(2):
        off = 6 + v * nR
        (m, s) = std_moments(T, off, nR, B)
        for bb in range(B + 1):
            for q in range(nR):
                Z[bb, q] = (T[bb, off + q] - m[q]) / s[q]
        Mb = np.empty(B)
        for bb in range(1, B + 1):
            mx = -1e+300
            for q in range(nR):
                if Z[bb, q] > mx:
                    mx = Z[bb, q]
            Mb[bb - 1] = mx
        z0max = -1e+300
        for q in range(nR):
            if Z[0, q] > z0max:
                z0max = Z[0, q]
        zmax[v] = z0max
        c = 0
        for bb in range(B):
            if Mb[bb] >= z0max - 1e-12:
                c += 1
        p[6 + v] = (1.0 + c) / (B + 1.0)
        mk = 0
        for q in range(nR):
            c = 0
            for bb in range(B):
                if Mb[bb] >= Z[0, q] - 1e-12:
                    c += 1
            if (1.0 + c) / (B + 1.0) <= alpha_claim:
                mk |= 1 << q
        masks[v] = mk
    return (p, masks, zmax)

@njit(cache=True)
def run_dataset(Y, bk, bp, E, gkv, gks, design_p, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr, alpha_claim, B):
    T = boot_table(Y, bk, bp, E, gkv, gks, design_p, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr)
    (p, masks, zmax) = pvals_from_table(T, cptr.shape[0] - 1, alpha_claim, B)
    return (p, masks, T[0].copy())

@njit(cache=True)
def stats_many(Ys, pa, pb, w1, sdk, sdp, skv, sks, cells, cptr):
    n = Ys.shape[0]
    nR = cptr.shape[0] - 1
    out = np.empty((n, 6 + 2 * nR))
    for t in range(n):
        plate_stats(Ys[t], pa, pb, w1, sdk, sdp, skv, sks, cells, cptr, out[t])
    return out
SURF_E = {'weak': 0.3, 'moderate': 0.7, 'strong': 0.95}
SURF_AB = {'flat': (0.0, 0.0), 'half': (0.7, 0.0)}

def surface(kind, I):
    d = simulate.dose_grid(I - 1, 1.0, 2.0)
    if kind in SURF_AB:
        (ea, eb) = SURF_AB[kind]
        ya = 1 - ea * d ** 1.5 / (d ** 1.5 + 1)
        yb = 1 - eb * d ** 1.5 / (d ** 1.5 + 1)
        return np.outer(ya, yb)
    y = 1 - SURF_E[kind] * d ** 1.5 / (d ** 1.5 + 1)
    return np.outer(y, y)

def pattern(kind, I):
    m = I - 1
    P = np.zeros((I, I))
    if kind == 'hdblock':
        P[m - 1:, m - 1:] = 1.0
    elif kind == 'lshape':
        P[m, 1:] = 1.0
        P[1:, m] = 1.0
    elif kind in ('broad', 'highdose', 'lowdose', 'corner', 'bump'):
        P = simulate.interaction_pattern(I, I, kind).reshape(I, I)
    else:
        raise ValueError(kind)
    return P

def truth(cond, surf, I):
    mu = surface(surf, I)
    if cond == 'null':
        return mu
    parts = cond.split('_')
    if parts[0] == 'sat':
        g = float(parts[1])
        out = mu.copy()
        out[1:, 1:] = mu[1:, 1:] ** (1.0 + g)
        return out
    if parts[0] == 'mixed':
        amp = float(parts[1])
        blk = pattern('hdblock', I) > 0
        out = mu.copy()
        out[blk] = mu[blk] * np.exp(-amp)
        other = np.zeros((I, I), bool)
        other[1:, 1:] = True
        other &= ~blk
        out[other] = mu[other] + (mu[blk] - out[blk]).sum() / other.sum()
        return out
    (s, pat, amp) = parts
    amp = float(amp)
    return mu * np.exp((-amp if s == 'syn' else amp) * pattern(pat, I))

def excess(mu):
    return np.outer(mu[1:, 0], mu[0, 1:]) - mu[1:, 1:]

def claim_metrics(masks, eta, reg):
    syn = eta.ravel() > 1e-09
    null_r = ~np.any(reg.M & syn[None, :], axis=1)
    inside_r = np.all(~reg.M | syn[None, :], axis=1) & syn.any()
    bits = (masks[:, None] >> np.arange(reg.nR)[None, :] & 1).astype(bool)
    anyc = bits.any(1)
    false = (bits & null_r[None, :]).any(1)
    correct = (bits & ~null_r[None, :]).any(1)
    sharp = (bits & inside_r[None, :]).any(1)
    return dict(any=int(anyc.sum()), false=int(false.sum()), correct=int(correct.sum()), sharp=int(sharp.sum()))
