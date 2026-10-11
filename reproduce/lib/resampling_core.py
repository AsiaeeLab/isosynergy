"""Grid designs and paired reference-size comparison kernels."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import numpy as np
from numba import njit
import paths
import null_search_core as H
from sirsyn import seed_for
RES = str(paths.SIMS)
NSTD = 200
GRIDS = (4, 5, 6, 8, 10)
NSTD = H.NSTD
NOISES = {'add': ('A_gauss', 'gauss_add'), 'oneil': ('C_hetero', 'hetero')}
NOISE_LABEL = {'add': 'additive Gaussian, SD 0.08', 'oneil': "O'Neil SD curve"}
SURFS = ('weak', 'moderate', 'strong')
PATTERNS_FOR = {'weak': ('broad', 'highdose', 'corner'), 'moderate': ('broad', 'highdose', 'corner', 'sat'), 'strong': ('broad', 'highdose', 'sat')}
CANON = {'broad': 'moderate', 'highdose': 'moderate', 'corner': 'weak', 'sat': 'strong'}
AMPS_EXP = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
AMPS_SAT = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
ALPHAS = (0.05, 0.01, 0.005)
TESTS = ('tstd', 'scan', 'mean', 'comb', 'std_raw')

def amps_for(pattern):
    return AMPS_SAT if pattern == 'sat' else AMPS_EXP

def upos(I):
    u = np.full(I, np.nan)
    u[1:] = np.linspace(0.0, 1.0, I - 1)
    return u

def pattern(kind, I):
    u = upos(I)
    P = np.zeros((I, I))
    if kind == 'broad':
        P[1:, 1:] = 1.0
    elif kind == 'highdose':
        P[1:, 1:] = np.outer(u[1:], u[1:])
    elif kind == 'corner':
        r = np.maximum(0.0, 3.0 * (u[1:] - 2.0 / 3.0))
        P[1:, 1:] = np.outer(r, r)
    else:
        raise ValueError(kind)
    return P

def truth(kind, amp, surf, I):
    mu = H.surface(surf, I)
    if kind == 'null' or amp == 0:
        return mu
    if kind == 'sat':
        out = mu.copy()
        out[1:, 1:] = mu[1:, 1:] ** (1.0 + amp)
        return out
    return mu * np.exp(-amp * pattern(kind, I))

def sd_of(noise, m):
    m = np.asarray(m, float)
    if noise == 'add':
        return np.full(m.shape, 0.08)
    return np.interp(m, H.KV, H.KS)

def effects(mu, noise):
    eta = H.excess(mu)
    bl = np.outer(mu[1:, 0], mu[0, 1:])
    z = eta / sd_of(noise, bl)
    return dict(mean_eta=float(eta.mean()), max_eta=float(eta.max()), mean_z=float(z.mean()), max_z=float(z.max()), n_syn=int((eta > 1e-12).sum()), min_mu=float(mu[1:, 1:].min()))
_comb_p = H.comb_p
_std_moments = H.std_moments
_pava = H.pava_dec
_DYK = H._DYK
_gen_design = H.gen_design
_sdfun = H.sdfun
_ref_mean = H.ref_mean

@njit(cache=True)
def pvals_core(T, nR, B):
    p = np.empty(9)
    p[8] = _comb_p(T, nR, B)
    for k in range(6):
        t0 = T[0, k]
        thr = t0 - 1e-12 * max(1.0, abs(t0))
        c = 0
        for bb in range(1, B + 1):
            if T[bb, k] >= thr:
                c += 1
        p[k] = (1.0 + c) / (B + 1.0)
    for v in range(2):
        off = 6 + v * nR
        (m, s) = _std_moments(T, off, nR, B)
        z0max = -1e+300
        for q in range(nR):
            z = (T[0, off + q] - m[q]) / s[q]
            if z > z0max:
                z0max = z
        c = 0
        for bb in range(1, B + 1):
            mx = -1e+300
            for q in range(nR):
                z = (T[bb, off + q] - m[q]) / s[q]
                if z > mx:
                    mx = z
            if mx >= z0max - 1e-12:
                c += 1
        p[6 + v] = (1.0 + c) / (B + 1.0)
    return p

def mean_test(Ys, design):
    from scipy import stats
    if design == 'S':
        import companion_core as R
        return np.nan_to_num(R.v_kc(Ys, mode='2sp')['p_syn'], nan=1.0)
    bs = Ys[:, 1:, :1] * Ys[:, :1, 1:] - Ys[:, 1:, 1:]
    r = bs.shape[1] * bs.shape[2]
    bs = bs.reshape(bs.shape[0], -1)
    t = bs.mean(1) / (bs.std(1, ddof=1) / np.sqrt(r))
    return stats.t.sf(t, r - 1)

def law(noise):
    (tname, bname) = NOISES[noise]
    (tk, tp, tc) = H.TRUTH[tname]
    (gk, gp, gc, sk, sp) = H.BOOT[bname]
    assert tc == gc == 'n'
    return (tk, tp, gk, gp, sk, sp)
wilson = H.wilson
