"""BH adjustment with explicit missing-p-value handling and a robust empirical null.

Missing values count as one by default and retain missing adjusted values. The
empirical null uses inverse survival probabilities, median and robust scale."""
import numpy as np
from scipy import stats


def bh_qvalues(p, nan_policy="one"):
    """Benjamini-Hochberg adjusted p-values (q-values). nan_policy 'one': NaN counts as p = 1 in the family;
    'omit': NaN entries are left out of the family. NaN inputs get NaN q-values in both cases."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if nan_policy == "one":
        pp = np.where(ok, p, 1.0)
        idx = np.arange(p.size)
    elif nan_policy == "omit":
        pp = p[ok]
        idx = np.where(ok)[0]
    else:
        raise ValueError("nan_policy must be 'one' or 'omit'")
    n = pp.size
    if n == 0:
        return q
    o = np.argsort(pp, kind="stable")
    adj = np.minimum.accumulate((pp[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    qq = np.empty(n)
    qq[o] = np.minimum(adj, 1.0)
    q[idx] = qq
    q[~ok] = np.nan
    return q


def bh_reject(p, q=0.05, nan_policy="one"):
    """Boolean BH rejections at level q (NaN p-values are never rejected)."""
    qv = bh_qvalues(p, nan_policy)
    return np.where(np.isfinite(qv), qv <= q, False)


def empirical_null(p):
    """Relative null within a family (estimates centered within the family): z = norm.isf(p), null center
    = median, scale = 1.4826 MAD (fallback IQR / 1.349, then 1). Returns (p relative to N(mu0, s0^2), mu0, s0).
    NaN inputs stay NaN."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    z = stats.norm.isf(np.clip(np.where(ok, p, 0.5), 1e-300, 1.0))
    z = np.where(np.isfinite(z), z, -37.0)
    zz = z[ok]
    if zz.size == 0:
        return np.full(p.shape, np.nan), np.nan, np.nan
    mu0 = float(np.median(zz))
    s0 = float(1.4826 * np.median(np.abs(zz - mu0)))
    if not s0 > 0:
        q75, q25 = np.percentile(zz, [75, 25])
        s0 = float((q75 - q25) / 1.349) if q75 > q25 else 1.0
    out = stats.norm.sf((z - mu0) / s0)
    out[~ok] = np.nan
    return out, mu0, s0
