"""Replicate screening applied identically to data and bootstrap plates.

At least three available replicates are needed. The most discrepant replicate
is screened using its distance from the median of the others in model SDs.
Two replicates cannot identify which measurement is discrepant."""
import warnings

import numpy as np


def screen(R, c, sd, skip_edges=False, avail=None, min_avail=3):
    """R: (n, m, I, J) replicate plates; avail: (m, I, J) or (n, m, I, J) bool availability (None = all).
    Returns (means (n, I, J), dropped (n, I, J) int: index of the dropped replicate or -1).
    Wells with no available replicate get NaN."""
    R = np.asarray(R, float)
    n, m, I, J = R.shape
    if avail is None and np.all(np.isfinite(R)):
        return _screen_complete(R, c, sd, skip_edges, min_avail)
    if avail is None:
        avail = np.isfinite(R)
    A = np.broadcast_to(np.asarray(avail, bool), R.shape)
    Rn = np.where(A, R, np.nan)
    cnt = A.sum(1)
    dev = np.full(R.shape, -np.inf)
    if m >= 2:
        for k in range(m):
            others = np.delete(Rn, k, axis=1)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(others, axis=1)
                d = np.abs(Rn[:, k] - med) / sd(med)
            dev[:, k] = np.where(A[:, k] & np.isfinite(d), d, -np.inf)
    dmax = dev.max(1)
    kmax = dev.argmax(1)
    flag = (dmax > c) & (cnt >= min_avail)
    _mask_never(flag, skip_edges)
    tot = np.where(A, R, 0.0).sum(1)
    drop = np.take_along_axis(np.where(A, R, 0.0), kmax[:, None], axis=1)[:, 0]
    with np.errstate(all="ignore"):
        means = np.where(flag, (tot - drop) / (cnt - 1), tot / cnt)
    means = np.where(cnt > 0, means, np.nan)
    return means, np.where(flag, kmax, -1)


def _mask_never(flag, skip_edges):
    flag[:, 0, 0] = False
    if skip_edges:
        flag[:, 0, :] = False
        flag[:, :, 0] = False


def _screen_complete(R, c, sd, skip_edges, min_avail):
    """Fast path for complete replicate stacks."""
    n, m, I, J = R.shape
    if m < min_avail:
        return R.mean(1), np.full((n, I, J), -1)
    dev = np.empty_like(R)
    for k in range(m):
        others = np.delete(R, k, axis=1)
        med = np.median(others, axis=1)
        dev[:, k] = np.abs(R[:, k] - med) / sd(med)
    dmax = dev.max(1)
    kmax = dev.argmax(1)
    flag = dmax > c
    _mask_never(flag, skip_edges)
    tot = R.sum(1)
    drop = np.take_along_axis(R, kmax[:, None], axis=1)[:, 0]
    means = np.where(flag, (tot - drop) / (m - 1), tot / m)
    return means, np.where(flag, kmax, -1)
