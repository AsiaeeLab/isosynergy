"""Isotonic SIR statistic against raw-well Bliss-excess statistics under the same bootstrap reference.

Every dataset is analyzed once by the package. The reference plates drawn for T_SIR are also used to
calibrate two raw-well statistics computed without any fit:
  raw_pos = sum over combination cells of ((Y_i0 Y_0j - Y_ij) / s(Y_i0 Y_0j))_+^2
  raw_max = max over combination cells of (Y_i0 Y_0j - Y_ij) / s(Y_i0 Y_0j)
The datasets and reference seeds of the size and power cells are those of sim_h2h_synth, so the T_SIR
p-values reproduce Fig. 3A-C and Supplementary Tables S1 and S3.
"""
import argparse
import os
import time
import mc
import numpy as np
import pandas as pd
import sirsyn.core as CORE
import sim_h2h_synth as H2H
(S, sim) = (mc.S, mc.sim)

POWER = [('moderate', 'syn_broad_0.20'), ('moderate', 'syn_hdblock_0.80'), ('weak', 'syn_corner_1.00'),
         ('moderate', 'syn_bump_0.30'), ('moderate', 'syn_lowdose_0.10'), ('strong', 'sat_0.60'), ('weak', 'mixed_0.60')]
MISSING = [('moderate', 'syn_broad_0.20'), ('moderate', 'syn_hdblock_0.80'), ('weak', 'syn_corner_1.00'),
           ('strong', 'sat_0.60')]
CAP = []
_orig_init = CORE._Grid.__init__
_orig_stats = CORE._Grid.stats


def _init(self, obs, sd):
    _orig_init(self, obs, sd)
    self._sdfun = sd


def raw_stats(Ys, grid):
    Ys = np.asarray(Ys, float)
    obs = grid.obs.reshape(grid.I, grid.J)[1:, 1:]
    pred = Ys[:, 1:, 0][:, :, None] * Ys[:, 0, 1:][:, None, :]
    z = (pred - Ys[:, 1:, 1:]) / grid._sdfun(pred)
    z = np.where(obs[None], z, np.nan)
    pos = np.nansum(np.maximum(z, 0.0) ** 2, axis=(1, 2))
    mx = np.nanmax(z.reshape(z.shape[0], -1), axis=1)
    return np.stack([pos, mx], 1)


def _stats(self, Ys):
    CAP.append(raw_stats(Ys, self))
    return _orig_stats(self, Ys)


CORE._Grid.__init__ = _init
CORE._Grid.stats = _stats


def tail_p(t, ref):
    return (1.0 + np.sum(ref >= t - 1e-12)) / (ref.size + 1.0)


def analyze(Y, nz, design, seed, B=399, n_std=200):
    CAP.clear()
    r = mc.fit_sim(Y, nz, design, seed)
    obs, ref = CAP[0][0], CAP[1][:B]
    em = np.asarray(r.excess_map, float)
    zr = (Y[1:, 0][:, None] * Y[0, 1:][None, :] - Y[1:, 1:]) if Y.ndim == 2 else None
    return dict(p_tstd=r.p_tstd, p_scan=r.p_scan, p_rawpos=tail_p(obs[0], ref[:, 0]),
                p_rawmax=tail_p(obs[1], ref[:, 1]), iso_map=em, raw_map=zr)


def job(args):
    (part, design, I, surf, cond, start, n, chunk_tag) = args
    ds = mc.data_seed(chunk_tag, design, I, surf, cond, 'C', start)
    rng = np.random.default_rng(ds)
    mu = H2H.truth(cond, surf, I)
    syn = sim.bliss_excess(mu) > 1e-9
    nz = S.ParametricNoise.oneil()
    Ys = sim.simulate(mu, nz, design, n, rng)
    rng_m = np.random.default_rng(mc.data_seed('raw_bench_missing', design, I, surf, cond, start))
    cells = [(i, j) for i in range(1, I) for j in range(1, I)]
    rows = []
    for t in range(n):
        Y = Ys[t]
        if part == 'missing':
            Y = Y.copy()
            for q in rng_m.permutation(len(cells))[:2]:
                Y[cells[q]] = np.nan
        out = analyze(Y, nz, design, [ds, t])
        row = dict(t=t, start=start, p_tstd=out['p_tstd'], p_scan=out['p_scan'], p_rawpos=out['p_rawpos'],
                   p_rawmax=out['p_rawmax'])
        if syn.any() and part == 'power':
            im = np.where(np.isfinite(out['iso_map']), out['iso_map'], -np.inf)
            rm = np.where(np.isfinite(out['raw_map']), out['raw_map'], -np.inf)
            row['loc_iso'] = bool(syn.ravel()[int(np.argmax(im))])
            row['loc_raw'] = bool(syn.ravel()[int(np.argmax(rm))])
        rows.append(row)
    df = pd.DataFrame(rows)
    (df['part'], df['design'], df['I'], df['surface'], df['cond']) = (part, design, I, surf, cond)
    df['n_syn_cells'] = int(syn.sum())
    return df


def cells():
    out = []
    for design in ('S', 'P'):
        for surf in ('weak', 'moderate', 'strong'):
            out += [('size', design, 5, surf, 'null', 2000, 500, 'h2h_size')]
        for (surf, cond) in POWER:
            out += [('power', design, 5, surf, cond, 1000, 250, 'h2h_power')]
    for design in ('S', 'P'):
        for (surf, cond) in MISSING:
            out += [('missing', design, 5, surf, cond, 1000, 250, 'raw_bench_missing_data')]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=mc.workers(16))
    a = mc.smoke_args(ap.parse_args())
    jobs = []
    for (part, design, I, surf, cond, n, chunk, tag) in cells():
        if mc.paths.SMOKE:
            n, chunk = 4, 4
        jobs += [(part, design, I, surf, cond, s, min(chunk, n - s), tag) for s in range(0, n, chunk)]
    t0 = time.time()
    res = pd.concat(mc.pool_map(job, jobs, workers=a.workers, progress=20), ignore_index=True)
    f = os.path.join(mc.RESDIR, 'raw_benchmark.parquet')
    res.to_parquet(f, index=False)
    al = 0.05
    g = res.groupby(['part', 'design', 'surface', 'cond'])
    summ = g.agg(n=('t', 'size'), tstd=('p_tstd', lambda p: np.mean(p <= al)), scan=('p_scan', lambda p: np.mean(p <= al)),
                 rawpos=('p_rawpos', lambda p: np.mean(p <= al)), rawmax=('p_rawmax', lambda p: np.mean(p <= al)),
                 loc_iso=('loc_iso', 'mean') if 'loc_iso' in res else ('t', 'size'),
                 loc_raw=('loc_raw', 'mean') if 'loc_raw' in res else ('t', 'size')).reset_index()
    summ.to_csv(os.path.join(mc.RESDIR, 'raw_benchmark_summary.csv'), index=False)
    print(summ.to_string(), flush=True)
    print(f'wrote {f}: {len(res)} rows, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
