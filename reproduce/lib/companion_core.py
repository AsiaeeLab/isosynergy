"""Variance estimates and exact-mean noise generation for companion tests."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import numpy as np
from scipy import stats, optimize
import paths
import companion_noise as C
RES = str(paths.SIMS)
LAM = C.BC_LAM
_EG = np.linspace(-9.0, 9.0, 3601)
_PG = stats.norm.pdf(_EG)
_PG = _PG / np.trapz(_PG, _EG)
EU2 = (1.0 - 0.4 ** 3) / (3 * 0.6)

def _bc_moments(u, s):
    x = np.maximum(u + s * _EG, 0.0) ** (1.0 / LAM)
    m1 = np.trapz(x * _PG, _EG)
    m2 = np.trapz(x * x * _PG, _EG)
    return (m1, m2 - m1 * m1)

def _bc_mix_moments(u, s):
    (m1a, v1a) = _bc_moments(u, s)
    (m1b, v1b) = _bc_moments(u, s * C.K_MIX)
    p = C.P_MIX
    m = (1 - p) * m1a + p * m1b
    e2 = (1 - p) * (v1a + m1a ** 2) + p * (v1b + m1b ** 2)
    return (m, e2 - m * m)

class CellNoise:

    def __init__(self, mu, family, level, null='v'):
        self.mu = np.asarray(mu, float)
        (self.family, self.level, self.null) = (family, level, null)
        mu = np.clip(self.mu, 1e-09, None)
        fam = family
        if fam in ('G', 'L'):
            sd = level * np.ones_like(mu) if fam == 'G' else level * C.log_sd_function(np.clip(mu, 0, 1.2))
            if null == 'v':
                loc = np.log(mu) - sd ** 2 / 2
            else:
                loc = np.log(mu)
            (self.loc, self.sd) = (loc, sd)
            m = np.exp(loc + sd ** 2 / 2)
            self.var = m * m * (np.exp(sd ** 2) - 1)
        elif fam == 'V':
            assert null == 'v'
            (self.loc, self.sd) = (mu.copy(), level * C.sd_v_pooled(mu))
            self.var = self.sd ** 2
        elif fam == 'H':
            assert null == 'v'
            sdb = level * C.sd_v_robust(mu) * C.BULK_ADJ
            (self.loc, self.sd) = (mu.copy(), sdb)
            gmag = level * C.sd_v_robust(mu) / C._SD055
            self.gmag = gmag
            self.var = sdb ** 2 * (1 - C.P_MIX + C.P_MIX * C.K_MIX ** 2) + C.P_GROSS * EU2 * gmag ** 2
        elif fam in ('B', 'HB'):
            assert null == 'v'
            if fam == 'B':
                s = level * C.sd_v_pooled(mu) * LAM / mu ** (1 - LAM)
                mom = _bc_moments
            else:
                s = level * C.sd_v_robust(mu) * LAM / mu ** (1 - LAM) * C.BULK_ADJ_B
                mom = _bc_mix_moments
            loc = np.empty_like(mu)
            var = np.empty_like(mu)
            for idx in np.ndindex(mu.shape):
                (target, si) = (mu[idx], s[idx])
                u0 = target ** LAM
                f = lambda u: mom(u, si)[0] - target
                (lo, hi) = (u0 - 6 * si, u0 + 0.5 * si)
                while f(lo) > 0:
                    lo -= 2 * si
                while f(hi) < 0:
                    hi += 2 * si
                u = optimize.brentq(f, lo, hi, xtol=1e-13)
                loc[idx] = u
                var[idx] = mom(u, si)[1]
            (self.loc, self.sd) = (loc, s)
            if fam == 'HB':
                gmag = level * C.sd_v_robust(mu) / C._SD055
                self.gmag = gmag
                var = var + C.P_GROSS * EU2 * gmag ** 2
            self.var = var
        else:
            raise ValueError(fam)

    def draw(self, design, nsim, rng, icc=0.0, m=1):
        n = self.mu.shape[0]
        mask = C.noisy_mask(n, design)
        e = rng.standard_normal((nsim, m) + self.mu.shape)
        if icc > 0:
            b = rng.standard_normal((nsim, m, 1, 1))
            e = np.sqrt(1 - icc) * e + np.sqrt(icc) * b
        fam = self.family
        if fam in ('G', 'L'):
            Y = np.exp(self.loc + self.sd * e)
        elif fam == 'V':
            Y = self.loc + self.sd * e
        elif fam == 'H':
            mix = np.where(rng.random(e.shape) < C.P_MIX, C.K_MIX, 1.0)
            Y = self.loc + self.sd * mix * e
            gross = rng.random(e.shape) < C.P_GROSS
            sgn = np.where(rng.random(e.shape) < 0.5, -1.0, 1.0)
            Y = Y + np.where(gross, sgn * rng.uniform(0.4, 1.0, e.shape) * self.gmag, 0.0)
        elif fam in ('B', 'HB'):
            if fam == 'B':
                Y = np.maximum(self.loc + self.sd * e, 0.0) ** (1.0 / LAM)
            else:
                mix = np.where(rng.random(e.shape) < C.P_MIX, C.K_MIX, 1.0)
                Y = np.maximum(self.loc + self.sd * mix * e, 0.0) ** (1.0 / LAM)
                gross = rng.random(e.shape) < C.P_GROSS
                sgn = np.where(rng.random(e.shape) < 0.5, -1.0, 1.0)
                Y = Y + np.where(gross, sgn * rng.uniform(0.4, 1.0, e.shape) * self.gmag, 0.0)
        Y = np.where(mask, Y, self.mu)
        Y[..., 0, 0] = 1.0
        return Y

def var_function(family, level, null='v', grid=None):
    grid = np.concatenate([np.geomspace(0.003, 0.05, 25), np.linspace(0.055, 1.25, 120)]) if grid is None else grid
    cn = CellNoise(grid[None, :], family, level, null)
    v = cn.var[0]
    (lg, lv) = (np.log(grid), np.log(v))
    return lambda m: np.exp(np.interp(np.log(np.clip(m, grid[0], grid[-1])), lg, lv))

def rank1_fit(Y, W=None, iters=300, tol=1e-12):
    (YA, YB, YI) = (Y[:, 1:, 0], Y[:, 0, 1:], Y[:, 1:, 1:])
    if W is None:
        (wA, wB, wI) = (np.ones_like(YA), np.ones_like(YB), np.ones_like(YI))
    else:
        (wA, wB, wI) = (W[:, 1:, 0], W[:, 0, 1:], W[:, 1:, 1:])
    (a, b) = (YA.copy(), YB.copy())
    for it in range(iters):
        a_new = (wA * YA + (wI * YI * b[:, None, :]).sum(2)) / (wA + (wI * b[:, None, :] ** 2).sum(2))
        b_new = (wB * YB + (wI * YI * a_new[:, :, None]).sum(1)) / (wB + (wI * a_new[:, :, None] ** 2).sum(1))
        d = max(np.abs(a_new - a).max(), np.abs(b_new - b).max())
        (a, b) = (a_new, b_new)
        if d < tol:
            break
    return (a, b)

def _fitted(a, b):
    n = a.shape[0]
    (I, J) = (a.shape[1] + 1, b.shape[1] + 1)
    F = np.empty((n, I, J))
    F[:, 0, 0] = 1.0
    F[:, 1:, 0] = a
    F[:, 0, 1:] = b
    F[:, 1:, 1:] = a[:, :, None] * b[:, None, :]
    return F

def _cvec(a, b):
    (n, I1) = a.shape
    J1 = b.shape[1]
    r = I1 * J1
    c = np.empty((n, I1 + 1, J1 + 1))
    c[:, 0, 0] = 0.0
    c[:, 1:, 0] = (b.mean(1) / I1)[:, None]
    c[:, 0, 1:] = (a.mean(1) / J1)[:, None]
    c[:, 1:, 1:] = -1.0 / r
    return c

def bhat(Y):
    return Y[:, 1:, 0].mean(1) * Y[:, 0, 1:].mean(1) - Y[:, 1:, 1:].mean((1, 2))

def _pt(est, se, df):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, est / np.where(se > 0, se, 1.0), np.nan)
    if np.isinf(df).all() if np.ndim(df) else np.isinf(df):
        return dict(est=est, se=se, t=t, df=df, p_syn=stats.norm.sf(t), p_ant=stats.norm.cdf(t))
    return dict(est=est, se=se, t=t, df=df, p_syn=stats.t.sf(t, df), p_ant=stats.t.cdf(t, df))
_OSD_V = C._V_KNOT
_OSD_S = C._V_POOLED

def g_oneil(m):
    return np.interp(np.clip(m, 0.0, 1.25), _OSD_V, _OSD_S)

def v_kc(Y, mode='hom', g=None, vfun=None, icc=0.0):
    Y = np.array(Y, float, copy=True)
    Y[:, 0, 0] = 1.0
    (n, I, J) = Y.shape
    r = (I - 1) * (J - 1)
    est = bhat(Y)
    nc = np.ones((I, J), bool)
    nc[0, 0] = False
    if mode in ('hom', '2s', '2sp', 'ext'):
        (a, b) = rank1_fit(Y)
        W = np.ones_like(Y)
    elif mode == 'w':
        g = g_oneil if g is None else g
        (a, b) = rank1_fit(Y)
        for _ in range(2):
            F = _fitted(a, b)
            W = 1.0 / g(np.clip(F, 0.005, 1.25)) ** 2
            (a, b) = rank1_fit(Y, W)
    else:
        raise ValueError(mode)
    F = _fitted(a, b)
    c = _cvec(a, b)
    res = Y - F
    res[:, 0, 0] = 0.0
    if mode == 'ext':
        v = vfun(np.clip(F, 0.001, None))
        v[:, 0, 0] = 0.0
        sd = np.sqrt(v)
        var1 = (1 - icc) * (c * c * v).sum((1, 2)) + icc * (c * sd).sum((1, 2)) ** 2
        VA = v[:, 1:, 0].sum(1) / (I - 1) ** 2
        VB = v[:, 0, 1:].sum(1) / (J - 1) ** 2
        return _pt(est, np.sqrt(var1 + VA * VB), np.inf)
    if mode in ('hom', 'w'):
        sw = np.sqrt(W)
        rt = res * sw
        ct = np.where(nc, c / sw, 0.0)
        cc = (ct * ct).sum((1, 2))
        proj = (ct * rt).sum((1, 2))
        rest = (rt * rt).sum((1, 2)) - proj * proj / cc
        s2 = np.maximum(rest, 1e-300) / (r - 1)
        VA = s2 * (1.0 / W[:, 1:, 0]).sum(1) / (I - 1) ** 2
        VB = s2 * (1.0 / W[:, 0, 1:]).sum(1) / (J - 1) ** 2
        se = np.sqrt(s2 * cc + VA * VB)
        return _pt(est, se, float(r - 1))
    return _v2s(Y, a, b, res, c, est, pos=mode == '2sp')

def _v2s(Y, a, b, res, c, est, chunk=2000, pos=False):
    (n, I, J) = Y.shape
    r = (I - 1) * (J - 1)
    N = I * J - 1
    idx = np.arange(1, I * J)
    isE = np.zeros(I * J, bool)
    isE[1:J] = True
    isE[np.arange(1, I) * J] = True
    isE = isE[idx]
    isI = ~isE
    se = np.empty(n)
    dfo = np.empty(n)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        (aa, bb) = (a[sl], b[sl])
        m = aa.shape[0]
        T = np.zeros((m, I, J, I - 1 + (J - 1) + 1))
        for i in range(1, I):
            T[:, i, 0, i - 1] = 1.0
            T[:, i, 1:, i - 1] = bb
        for j in range(1, J):
            T[:, 0, j, I - 1 + j - 1] = 1.0
            T[:, 1:, j, I - 1 + j - 1] = aa
        T[..., -1] = c[sl]
        T = T.reshape(m, I * J, -1)[:, idx, :]
        (Q, _) = np.linalg.qr(T)
        R = np.eye(N)[None] - Q @ np.swapaxes(Q, 1, 2)
        rv = res[sl].reshape(m, -1)[:, idx]
        rr = np.einsum('nkl,nl->nk', R, rv)
        qE = (rr[:, isE] ** 2).sum(1)
        qI = (rr[:, isI] ** 2).sum(1)
        R2 = R * R
        mEE = R2[:, isE][:, :, isE].sum((1, 2))
        mEI = R2[:, isE][:, :, isI].sum((1, 2))
        mII = R2[:, isI][:, :, isI].sum((1, 2))
        det = mEE * mII - mEI * mEI
        sE = (mII * qE - mEI * qI) / det
        sI = (mEE * qI - mEI * qE) / det
        tot = qE + qI
        negE = sE < 0
        negI = sI < 0
        sE = np.where(negE, 0.0, sE)
        sI = np.where(negE, tot / (mEI + mII), sI)
        sI = np.where(negI & ~negE, 0.0, sI)
        sE = np.where(negI & ~negE, tot / (mEE + mEI), sE)
        cv = c[sl].reshape(m, -1)[:, idx]
        lE = (cv[:, isE] ** 2).sum(1)
        lI = (cv[:, isI] ** 2).sum(1)
        VA = sE / (I - 1)
        VB = sE / (J - 1)
        hE = (mII * lE - mEI * lI) / det
        hI = (mEE * lI - mEI * lE) / det
        if pos:
            hI = np.maximum(hI, 0.0)
            v = hE * qE + hI * qI + VA * VB
        else:
            v = lE * sE + lI * sI + VA * VB
        se[sl] = np.sqrt(v)
        h = np.where(isE[None, :], hE[:, None], hI[:, None])
        S = np.where(isE[None, :], sE[:, None], sI[:, None])
        QS = R * h[:, None, :] @ (R * S[:, None, :])
        num = np.einsum('nkk->n', QS)
        den = (QS * np.swapaxes(QS, 1, 2)).sum((1, 2))
        dfo[sl] = np.clip(num * num / np.maximum(den, 1e-300), 1.0, r - 1)
    return _pt(est, se, dfo)

def log_kc(Y, floor=0.01):
    Z = C.to_log(np.array(Y, float), floor)
    (ps, pa, est) = C.t_known_control(Z)
    return dict(est=est, p_syn=ps, p_ant=pa)

def v_fixed_edges(Y):
    (ps, pa, est) = C.t_viability_fixed_edges(None, Y)
    return dict(est=est, p_syn=ps, p_ant=pa)

def mixed_truth_v(n, surface, kind, peak):
    yref = C.bliss_ref(n, surface)
    if kind == 'hilo':
        (S, A) = (C.pattern(n, 'highdose'), C.pattern(n, 'lowdose'))
    else:
        cc = C.pattern(n, 'checker')
        (S, A) = (np.maximum(cc, 0.0), np.maximum(-cc, 0.0))

    def y_of(k):
        return np.minimum(1.0, yref * np.exp(-(peak * S - k * A)))

    def f(k):
        return C.mean_excess_v(y_of(k), yref)
    hi = 50.0
    if f(hi) > 0:
        return (None, np.nan)
    k = optimize.brentq(f, 0.0, hi, xtol=1e-12)
    return (y_of(k), k)

def wilson(k, n):
    return C.wilson(k, n)
