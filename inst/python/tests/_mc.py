"""Monte Carlo harness for the calibration tests: each job simulates datasets with its own seed and runs
sirsyn.fit (stage 1 only, B = 399, for the calibration checks). At most 12 worker processes."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import sirsyn as S  # noqa: E402
from sirsyn import simulate as sim  # noqa: E402

WORKERS = min(12, int(os.environ.get("SIRSYN_TEST_WORKERS", "12")))
LAWS = {"A": lambda: S.ParametricNoise.additive(0.08), "B": lambda: S.ParametricNoise.lognormal(0.12),
        "C": lambda: S.ParametricNoise.oneil()}


def compare(k, n, p_ref, n_ref):
    """Two-sample z statistic of a rate k / n against a reference estimate p_ref from n_ref datasets."""
    p = k / n
    pp = (k + p_ref * n_ref) / (n + n_ref)
    se = np.sqrt(pp * (1 - pp) * (1.0 / n + 1.0 / n_ref))
    return (p - p_ref) / se


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _job_global(args):
    """design S / P single plates: returns (n, 3) p-values (T_std, scan, mean companion)."""
    design, I, surf, law, n, seed = args
    nz = LAWS[law]()
    rng = np.random.default_rng(seed)
    mu = sim.truth("null", surf, I)
    Ys = sim.simulate(mu, nz, design, n, rng)
    out = np.empty((n, 3))
    for t in range(n):
        r = S.fit(Ys[t], design=design, noise=nz, seed=[seed, t], two_stage=False, fragility="never")
        out[t] = (r.p_tstd, r.p_scan, r.p_mean)
    return out


def _job_missing(args):
    """design S single plates; per dataset: complete grid, 3 MCAR missing combination wells, one MCAR
    missing single-agent well under rule R. Returns (n, 3) T_std p-values."""
    I, surf, law, n, seed = args
    nz = LAWS[law]()
    rng = np.random.default_rng(seed)
    mu = sim.truth("null", surf, I)
    Ys = sim.simulate(mu, nz, "S", n, rng)
    out = np.empty((n, 3))
    for t in range(n):
        Y = Ys[t]
        r0 = S.fit(Y, noise=nz, seed=[seed, t], two_stage=False, fragility="never")
        Ym = Y.copy()
        for c in sim.mcar_cells(rng, I, I, 3):
            Ym[c] = np.nan
        r1 = S.fit(Ym, noise=nz, seed=[seed, t], two_stage=False, fragility="never")
        Ya = Y.copy()
        q = rng.integers(0, 2 * (I - 1))
        if q < I - 1:
            Ya[q + 1, 0] = np.nan
        else:
            Ya[0, q - (I - 1) + 1] = np.nan
        r2 = S.fit(Ya, noise=nz, seed=[seed, t], two_stage=False, fragility="never")
        out[t] = (r0.p_tstd, r1.p_tstd, r2.p_tstd)
    return out


def _job_screen(args):
    """replicated design S, 4 replicates, clean data: T_std p with and without the replicate screen (same
    data, same reference draws). Returns (n, 2) and the number of data flags."""
    I, surf, law, n, seed = args
    nz = LAWS[law]()
    rng = np.random.default_rng(seed)
    mu = sim.truth("null", surf, I)
    Rs = sim.simulate(mu, nz, "S", n, rng, m=4)
    out = np.empty((n, 3))
    for t in range(n):
        a = S.fit(Rs[t], design="R", noise=nz, seed=[seed, t], two_stage=False, screen_replicates=True)
        b = S.fit(Rs[t], design="R", noise=nz, seed=[seed, t], two_stage=False, screen_replicates=False)
        out[t] = (a.p_tstd, b.p_tstd, len(a.log.screened))
    return out


def run(job, base_args, n_total, chunk, key):
    jobs = [tuple(base_args) + (min(chunk, n_total - s), S.seed_for("sirsyn-test", key, s))
            for s in range(0, n_total, chunk)]
    if WORKERS > 1:
        with mp.get_context("fork").Pool(WORKERS) as p:
            parts = p.map(job, jobs)
    else:
        parts = [job(j) for j in jobs]
    return np.concatenate(parts)


def rate(p, alpha):
    p = np.asarray(p)
    return int(np.sum(p <= alpha * (1 + 1e-9))), p.size


def _norm_sf(z):
    return stats.norm.sf(z)
