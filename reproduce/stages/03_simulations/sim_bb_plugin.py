"""sim_bb_plugin calculations."""
import mc
import argparse
import os
import time
import numpy as np
import pandas as pd
(S, sim) = (mc.S, mc.sim)
SL = 0.12
CV = float(np.sqrt(np.exp(SL ** 2) - 1.0))
_sd_data = pd.read_csv(os.path.join(mc.INPUTS, 'sd_by_mean.csv'))
_sd_data = _sd_data[_sd_data.scale == 'identity_raw'].sort_values('mean_viability_raw')
KV15 = np.ascontiguousarray(_sd_data.mean_viability_raw.to_numpy(float))
KS15 = np.ascontiguousarray(_sd_data.pooled_sd.to_numpy(float))
NOISE = {'add': S.ParametricNoise.additive(0.08), 'logn': S.ParametricNoise('lognormal', SL, sd=S.SDCurve('table', kv=[0.01, 1.2], ks=[CV * 0.01, CV * 1.2])), 'oneil': S.ParametricNoise.sd_table(S.SDCurve('table', kv=KV15, ks=KS15))}
BASES = {'weak': 0.3, 'moderate': 0.7, 'strong': 0.95}
POWER = ('syn_broad', 'syn_high', 'syn_corner', 'syn_sat')
_amps = pd.read_csv(os.path.join(mc.INPUTS, 'bb_amplitudes.csv'))
AMPS = {(r.noise, r.base, r.cond): float(r.amp) for r in _amps.itertuples()}

def base_curve(base):
    d = sim.dose_grid(4, 1.0, 2.0)
    return (1 - BASES[base] * d ** 1.5 / (d ** 1.5 + 1))[1:]

def pattern(kind):
    if kind in ('broad', 'highdose', 'corner'):
        return sim.interaction_pattern(5, 5, kind)[1:, 1:].ravel()
    (ii, jj) = np.meshgrid(np.arange(1, 5), np.arange(1, 5), indexing='ij')
    xy = ii / 4 * (jj / 4)
    p = 4 * xy * (1 - xy)
    return (p / p.max()).ravel()

def truth_int(base, cond, amp):
    a = base_curve(base)
    mB = np.outer(a, a).ravel()
    if cond == 'bliss':
        return mB
    pk = {'broad': 'broad', 'high': 'highdose', 'corner': 'corner', 'sat': 'saturating'}[cond.split('_')[1]]
    return mB * np.exp(-amp * pattern(pk))

def sdw(m, noise):
    if noise == 'add':
        return np.full(np.shape(m), 0.08)
    if noise == 'logn':
        return CV * np.maximum(np.clip(m, 0.0, 1.2), 0.01)
    return np.interp(np.clip(m, 0.0, 1.2), KV15, KS15)

def gen_row(noise, base, ms, cond, amp, n):
    a = base_curve(base)
    rng = np.random.default_rng(S.seed_for(bytes([114, 54, 98]).decode('ascii'), 'data', noise, base, ms, cond))
    r = ms if noise == 'logn' else 1
    ZA = rng.standard_normal((n, 4, r))
    ZB = rng.standard_normal((n, 4, r))
    ZI = rng.standard_normal((n, 16))
    mint = truth_int(base, cond, amp)
    if noise == 'logn':
        eA = a[None] * np.exp(SL * ZA - 0.5 * SL * SL).mean(2)
        eB = a[None] * np.exp(SL * ZB - 0.5 * SL * SL).mean(2)
        yi = mint[None] * np.exp(SL * ZI - 0.5 * SL * SL)
    else:
        eA = a[None] + sdw(a, noise)[None] * ZA[:, :, 0] / np.sqrt(ms)
        eB = a[None] + sdw(a, noise)[None] * ZB[:, :, 0] / np.sqrt(ms)
        yi = mint[None] + sdw(mint, noise)[None] * ZI
        if noise == 'oneil':
            (eA, eB, yi) = (np.maximum(eA, 0.0), np.maximum(eB, 0.0), np.maximum(yi, 0.0))
    Y = np.empty((n, 5, 5))
    Y[:, 0, 0] = 1.0
    Y[:, 1:, 0] = eA
    Y[:, 0, 1:] = eB
    Y[:, 1:, 1:] = yi.reshape(n, 4, 4)
    return Y

def rows_all(n_scale=1.0):
    R = []
    for noise in ('add', 'logn', 'oneil'):
        for base in BASES:
            for ms in (1, 4, 16, 64, 256):
                R.append((noise, base, ms, 'bliss', 0.0, 2000, max(1, int(2000 * n_scale))))
                for cond in POWER:
                    R.append((noise, base, ms, cond, AMPS[noise, base, cond], 1000, max(1, int(1000 * n_scale))))
    return R

def job(args):
    (noise, base, ms, cond, amp, n, k0, k1) = args
    Y = gen_row(noise, base, ms, cond, amp, n)
    fs = int(mc.data_seed('figS2', noise, base, ms, cond))
    out = []
    for k in range(k0, k1):
        res = mc.fit(Y[k], design='P', noise=NOISE[noise], sa_replicates=float(ms), sa_rho=0.0, seed=[fs, k], two_stage=False, B=399, n_std=200, fragility='never')
        out.append(dict(noise=noise, base=base, ms=ms, cond=cond, amp=amp, k=k, p_tstd=res.p_tstd, t_std=res.t_std))
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--n_scale', type=float, default=1.0)
    ap.add_argument('--tag', default='')
    a = mc.smoke_args(ap.parse_args())
    jobs = []
    for (noise, base, ms, cond, amp, n, nfit) in rows_all(a.n_scale):
        for k0 in range(0, nfit, 250):
            jobs.append((noise, base, ms, cond, amp, n, k0, min(nfit, k0 + 250)))
    t0 = time.time()
    out = mc.pool_map(job, jobs, workers=a.workers, progress=50)
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(os.path.join(mc.RESDIR, f'bb_plugin{a.tag}.parquet'), index=False)
    print(f'done: {len(d)} rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
