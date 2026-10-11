"""Noise families and mean-excess contrasts for Supplementary Section S3."""
import paths
import numpy as np
from scipy import stats, optimize
from sirsyn import simulate as sim, seed_for

from model_curves import VIABILITY_MEAN, VIABILITY_SD, VIABILITY_ROBUST_SD, POSITIVE_MEAN, POSITIVE_LOG_SD

def log_sd_function(v):
    return np.interp(np.asarray(v,dtype=float),POSITIVE_MEAN,POSITIVE_LOG_SD)
_V_KNOT=VIABILITY_MEAN
_V_POOLED=VIABILITY_SD
_V_ROBUST=VIABILITY_ROBUST_SD

def sd_v_pooled(y):
    return np.interp(y, _V_KNOT, _V_POOLED)

def sd_v_robust(y):
    return np.interp(y, _V_KNOT, _V_ROBUST)
(P_MIX, K_MIX, P_GROSS) = (0.045, 4.0, 0.004)
BULK_ADJ = 0.957
_SD055 = float(np.interp(0.55, _V_KNOT, _V_ROBUST))
BC_LAM = 0.3
BULK_ADJ_B = 0.958
SURFACES = {'both_active': (0.95, 0.95, 1.5, 1.5), 'one_weak': (0.9, 0.1, 1.5, 1.5), 'partial': (0.5, 0.5, 1.5, 1.5), 'both_weak': (0.12, 0.12, 1.5, 1.5)}
SPAN = 2.0

def single_agents(n, surface):
    (EA, EB, hA, hB) = SURFACES[surface]
    d = sim.dose_grid(n - 1, 1.0, SPAN)
    yA = 1.0 - EA * d ** hA / (d ** hA + 1.0)
    yB = 1.0 - EB * d ** hB / (d ** hB + 1.0)
    return (yA, yB)

def bliss_ref(n, surface):
    (yA, yB) = single_agents(n, surface)
    return np.outer(yA, yB)

def pattern(n, kind):
    (ii, jj) = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    inter = (ii > 0) & (jj > 0)
    if kind == 'corner':
        k = 3 if n >= 8 else 2
        p = ((ii >= n - k) & (jj >= n - k)).astype(float)
    elif kind == 'checker':
        p = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    elif kind in ('broad', 'highdose', 'lowdose'):
        p = sim.interaction_pattern(n, n, kind).reshape(n, n)
    else:
        raise ValueError(kind)
    return np.where(inter, p, 0.0)

def mean_excess_v(y, yref):
    return float((yref - y)[1:, 1:].mean())

def mean_excess_log(y, yref):
    return float((np.log(yref) - np.log(y))[1:, 1:].mean())

def synergy_truth(n, surface, kind, target):
    yref = bliss_ref(n, surface)
    P = pattern(n, kind)

    def f(a):
        return mean_excess_v(yref * np.exp(-a * P), yref) - target
    hi = 12.0
    if f(hi) < 0:
        return (None, mean_excess_v(yref * np.exp(-hi * P), yref))
    a = optimize.brentq(f, 0.0, hi, xtol=1e-10)
    return (yref * np.exp(-a * P), a)

def antagonism_truth(n, surface, kind, target):
    yref = bliss_ref(n, surface)
    P = pattern(n, kind)

    def f(a):
        return -mean_excess_v(np.minimum(1.0, yref * np.exp(a * P)), yref) - target
    hi = 12.0
    if f(hi) < 0:
        return (None, -mean_excess_v(np.minimum(1.0, yref * np.exp(hi * P)), yref))
    a = optimize.brentq(f, 0.0, hi, xtol=1e-10)
    return (np.minimum(1.0, yref * np.exp(a * P)), a)

def mixed_truth(n, surface, kind, peak):
    yref = bliss_ref(n, surface)
    if kind == 'hilo':
        (S, A) = (pattern(n, 'highdose'), pattern(n, 'lowdose'))
    elif kind == 'checker':
        c = pattern(n, 'checker')
        (S, A) = (np.maximum(c, 0.0), np.maximum(-c, 0.0))
    else:
        raise ValueError(kind)

    def y_of(k):
        return np.minimum(1.0, yref * np.exp(-(peak * S - k * A)))

    def f(k):
        return mean_excess_log(y_of(k), yref)
    hi = 50.0
    if f(hi) > 0:
        return (None, float('nan'))
    k = optimize.brentq(f, 0.0, hi, xtol=1e-12)
    return (y_of(k), k)

def noisy_mask(n, design):
    m = np.ones((n, n), bool)
    m[0, 0] = False
    if design in ('P', 'R'):
        m[0, :] = False
        m[:, 0] = False
    return m

def simulate(y, design, noise, level, nsim, rng, m=1, icc=0.0):
    n = y.shape[0]
    mask = noisy_mask(n, design)
    e = rng.standard_normal((nsim, m, n, n))
    if icc > 0:
        b = rng.standard_normal((nsim, m, 1, 1))
        e = np.sqrt(1 - icc) * e + np.sqrt(icc) * b
    e = np.where(mask, e, 0.0)
    if noise in ('G', 'L'):
        sd = level if noise == 'G' else level * log_sd_function(np.clip(y, 0.0, 1.2))
        Z = np.log(y) + sd * e
        Z[..., 0, 0] = 0.0
        return (np.exp(Z), Z)
    if noise == 'V':
        Y = y + level * sd_v_pooled(y) * e
    elif noise == 'H':
        sdb = level * sd_v_robust(y) * BULK_ADJ
        mix = np.where(rng.random(e.shape) < P_MIX, K_MIX, 1.0)
        Y = y + sdb * mix * e
        gross = (rng.random(e.shape) < P_GROSS) & mask
        mag = rng.uniform(0.4, 1.0, e.shape) * (level * sd_v_robust(y) / _SD055)
        sgn = np.where(rng.random(e.shape) < 0.5, -1.0, 1.0)
        Y = Y + np.where(gross, sgn * mag, 0.0)
    elif noise == 'B':
        s = level * sd_v_pooled(y) * BC_LAM / np.maximum(y, 1e-09) ** (1 - BC_LAM)
        Y = np.maximum(y ** BC_LAM + s * e, 0.0) ** (1.0 / BC_LAM)
    elif noise == 'HB':
        s = level * sd_v_robust(y) * BC_LAM / np.maximum(y, 1e-09) ** (1 - BC_LAM) * BULK_ADJ_B
        mix = np.where(rng.random(e.shape) < P_MIX, K_MIX, 1.0)
        Y = np.maximum(y ** BC_LAM + s * mix * e, 0.0) ** (1.0 / BC_LAM)
        gross = (rng.random(e.shape) < P_GROSS) & mask
        mag = rng.uniform(0.4, 1.0, e.shape) * (level * sd_v_robust(y) / _SD055)
        sgn = np.where(rng.random(e.shape) < 0.5, -1.0, 1.0)
        Y = Y + np.where(gross, sgn * mag, 0.0)
    else:
        raise ValueError(noise)
    Y = np.where(mask, Y, y)
    Y[..., 0, 0] = 1.0
    return (Y, None)

def to_log(Y, floor):
    Z = np.log(np.maximum(Y, floor))
    Z[..., 0, 0] = 0.0
    return Z

def excess(Z):
    return Z[..., 1:, :1] + Z[..., :1, 1:] - Z[..., :1, :1] - Z[..., 1:, 1:]

def _tp(est, se, df):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, est / np.where(se > 0, se, 1.0), np.where(est > 0, np.inf, np.where(est < 0, -np.inf, 0.0)))
    return (stats.t.sf(t, df), stats.t.cdf(t, df), est)

def contrast(Z, known_control):
    Z = np.array(Z, dtype=float, copy=True)
    if known_control:
        Z[:, 0, 0] = 0.0
    (n, I, J) = Z.shape
    r = (I - 1) * (J - 1)
    PL = Z.mean(2, keepdims=True) + Z.mean(1, keepdims=True) - Z.mean((1, 2), keepdims=True)
    ee = ((Z - PL) ** 2).sum((1, 2))
    est = excess(Z).mean((1, 2))
    cc = I * J / r
    num = est / np.sqrt(cc)
    rest = ee - num * num
    s = np.sqrt(np.maximum(rest, 1e-300) / (r - 1))
    se = s * np.sqrt(cc - 1.0) if known_control else s * np.sqrt(cc)
    return _tp(est, se, r - 1)

def t_iid(Z, Y=None):
    return contrast(Z, False)

def t_known_control(Z, Y=None):
    return contrast(Z, True)

def demidenko_miller(Z, Y=None):
    (n, I, J) = Z.shape
    (x, y, zz) = (Z[:, 1:, 0], Z[:, 0, 1:], Z[:, 1:, 1:].reshape(n, -1))
    (nA, nB) = (I - 1, J - 1)
    df = nA + nB + nA * nB - 3
    s2 = (((x - x.mean(1, keepdims=True)) ** 2).sum(1) + ((y - y.mean(1, keepdims=True)) ** 2).sum(1) + ((zz - zz.mean(1, keepdims=True)) ** 2).sum(1)) / df
    est = x.mean(1) + y.mean(1) - zz.mean(1) - Z[:, 0, 0]
    return _tp(est, np.sqrt(s2 * (1 / nA + 1 / nB + 1 / (nA * nB))), df)

def t_fixed_edges(Z, Y=None):
    d = excess(Z).reshape(Z.shape[0], -1)
    r = d.shape[1]
    return _tp(d.mean(1), d.std(1, ddof=1) / np.sqrt(r), r - 1)

def _bscores(Y):
    return (Y[:, 1:, :1] * Y[:, :1, 1:] / Y[:, :1, :1] - Y[:, 1:, 1:]).reshape(Y.shape[0], -1)

def t_viability_fixed_edges(Z, Y):
    b = _bscores(Y)
    r = b.shape[1]
    return _tp(b.mean(1), b.std(1, ddof=1) / np.sqrt(r), r - 1)

def sfp_pooled(Z, Y):
    b = _bscores(Y)
    nn = b.shape[1]
    mb = b.mean(1)
    se = b.std(1, ddof=1) / np.sqrt(nn)
    with np.errstate(divide='ignore', invalid='ignore'):
        zz = np.where(se > 0, np.abs(mb) / np.where(se > 0, se, 1.0), np.inf)
        p2 = np.where(np.isfinite(zz), np.exp(-0.717 * zz - 0.416 * zz * zz), 0.0)
    p2 = np.minimum(1.0, p2)
    ps = np.where(mb > 0, p2 / 2, 1 - p2 / 2)
    pa = np.where(mb < 0, p2 / 2, 1 - p2 / 2)
    return (ps, pa, mb)

def t_fixed_edges_pure_error(Zr):
    (n, m) = Zr.shape[:2]
    d = excess(Zr)
    dbar = d.mean(1)
    r = dbar.shape[1] * dbar.shape[2]
    s2 = ((d - dbar[:, None]) ** 2).sum((1, 2, 3)) / (r * (m - 1))
    return _tp(dbar.mean((1, 2)), np.sqrt(s2 / (r * m)), r * (m - 1))

def t_block_level(Zr):
    (n, m) = Zr.shape[:2]
    E = excess(Zr).reshape(n, m, -1).mean(2)
    return _tp(E.mean(1), E.std(1, ddof=1) / np.sqrt(m), m - 1)

def _pow(Y, lam):
    return np.maximum(Y, 0.0) ** lam

def _pscores(Y, lam):
    Yp = _pow(Y, lam)
    return Yp[..., 1:, :1] * Yp[..., :1, 1:] / Yp[..., :1, :1] - Yp[..., 1:, 1:]

def scores_pure_error(D):
    (n, m, r) = D.shape
    dbar = D.mean(1)
    s2 = ((D - dbar[:, None]) ** 2).sum((1, 2)) / (r * (m - 1))
    return _tp(dbar.mean(1), np.sqrt(s2 / (r * m)), r * (m - 1))

def scores_block_level(D):
    (n, m, r) = D.shape
    E = D.mean(2)
    return _tp(E.mean(1), E.std(1, ddof=1) / np.sqrt(m), m - 1)

def replicate_tests(Zr, Yr):
    (n, m) = Zr.shape[:2]
    out = {'t_fixed_edges_pure_error': t_fixed_edges_pure_error(Zr), 't_block_level': t_block_level(Zr)}
    for (tag, lam) in [('viability', 1.0), ('power03', BC_LAM)]:
        D = _pscores(Yr, lam).reshape(n, m, -1)
        out[f't_{tag}_pure_error'] = scores_pure_error(D)
        out[f't_{tag}_block_level'] = scores_block_level(D)
    return out

def wilson(k, n, z=1.959964):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (cen - half, cen + half)
