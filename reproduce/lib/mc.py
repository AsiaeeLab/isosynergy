"""Monte Carlo execution and deterministic per-dataset seeds."""
import os
import sys
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_k] = '1'
import paths
HERE = str(paths.LIB)
ROOT = str(paths.ROOT)
FINAL = str(paths.ROOT / 'stages' / '02_realdata')
INPUTS = str(paths.INPUTS)
RESDIR = str(paths.SIMS)
import multiprocessing as _mp
import numpy as np
import sirsyn as S
from sirsyn import simulate as sim
MAX_WORKERS = 16

def workers(default):
    w = int(os.environ.get('SIR_WORKERS', 16))
    return max(1, min(w, MAX_WORKERS))

def pool_map(func, jobs, workers=4, chunksize=1, progress=None):
    workers = max(1, min(int(workers), MAX_WORKERS))
    if workers == 1:
        return [func(j) for j in jobs]
    ctx = _mp.get_context('fork')
    out = []
    with ctx.Pool(workers, maxtasksperchild=200) as pool:
        for (k, r) in enumerate(pool.imap(func, jobs, chunksize=chunksize)):
            out.append(r)
            if progress and (k + 1) % progress == 0:
                print(f'  {k + 1}/{len(jobs)} jobs', flush=True)
    return out

def data_seed(*keys):
    return S.seed_for('paperfigs', *keys)

def wilson(k, n, z=1.959964):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def fit_sim(Y, noise, design, ref_seed, **kw):
    args = dict(design=design, noise=noise, seed=ref_seed, two_stage=False, B=399, n_std=200, fragility='never')
    args.update(kw)
    if paths.SMOKE:
        args.update(B=19, n_std=9, two_stage=False)
    return S.fit(Y, **args)

def fit(*args, **kwargs):
    """Fit a dataset with the requested settings, or the small smoke reference."""
    if paths.SMOKE:
        kwargs.update(B=39, B2=99, n_std=9)
    return S.fit(*args, **kwargs)

def smoke_args(args):
    """Keep all design cells and reduce only their dataset counts for smoke runs."""
    if hasattr(args, 'workers'):
        args.workers = paths.WORKERS
    if paths.SMOKE:
        if hasattr(args, 'n'):
            args.n = 4
        if hasattr(args, 'screens'):
            args.screens = 2
        if hasattr(args, 'n_scale'):
            args.n_scale = 0.002
    return args
