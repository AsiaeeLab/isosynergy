"""Simultaneous coverage of single-agent confidence bands."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_core as K
BETA = 0.005
CHUNK = 200000

def edges(rng, a, ms, noise, n):
    k = len(a)
    if noise == 'logn':
        out = np.zeros((n, k))
        for r in range(ms):
            out += np.exp(K.SL_LOGN * rng.standard_normal((n, k)) - 0.5 * K.SL_LOGN ** 2)
        return a[None, :] * out / ms
    Z = rng.standard_normal((n, k))
    if noise == 'add':
        return a[None, :] + K.S0_ADD * Z / np.sqrt(ms)
    s = np.array([K.sd_well(v, 2, K.KV, K.KS) for v in a])
    return np.maximum(a[None, :] + s[None, :] * Z / np.sqrt(ms), 0.0)

def run_setting(I, noise, base, ms, n):
    a = K.base_ab(base, I)[0]
    nw = 2 * (I - 1)
    methods = ['exact', 'plugin_sd', 'lower'] + (['gaussform'] if noise == 'logn' else [])
    specs = {m: K.BandSpec(noise, ms, BETA, nw, m) for m in methods}
    acc = {m: dict(cover=0, empty=0, miss_lo=0, miss_hi=0, width=0.0, well_miss=0) for m in methods}
    rng = np.random.default_rng(K.seed('cov', I, noise, base, ms))
    done = 0
    while done < n:
        m_ = min(CHUNK, n - done)
        YA = edges(rng, a, ms, noise, m_)
        YB = edges(rng, a, ms, noise, m_)
        for (meth, bs) in specs.items():
            (LA, UA, okA) = bs.band(YA)
            (LB, UB, okB) = bs.band(YB)
            if meth == 'lower':
                inA = np.all(a[None, :] >= LA - 1e-12, axis=1)
                inB = np.all(a[None, :] >= LB - 1e-12, axis=1)
                cov = inA & inB
                acc[meth]['miss_lo'] += int((~cov).sum())
                acc[meth]['width'] += float((a[None, :] - LA).mean(1).sum() + (a[None, :] - LB).mean(1).sum()) / 2
                (loA, _) = bs.intervals(YA)
                (loB, _) = bs.intervals(YB)
                acc[meth]['well_miss'] += int((loA > a[None, :] + 1e-12).sum() + (loB > a[None, :] + 1e-12).sum())
            else:
                inA = okA & np.all((a[None, :] >= LA - 1e-12) & (a[None, :] <= UA + 1e-12), axis=1)
                inB = okB & np.all((a[None, :] >= LB - 1e-12) & (a[None, :] <= UB + 1e-12), axis=1)
                cov = inA & inB
                acc[meth]['empty'] += int((~okA).sum() + (~okB).sum())
                acc[meth]['miss_lo'] += int((np.any(a[None, :] < LA - 1e-12, 1) | np.any(a[None, :] < LB - 1e-12, 1)).sum())
                acc[meth]['miss_hi'] += int((np.any(a[None, :] > UA + 1e-12, 1) | np.any(a[None, :] > UB + 1e-12, 1)).sum())
                w = np.where(okA[:, None], UA - LA, np.nan)
                w2 = np.where(okB[:, None], UB - LB, np.nan)
                acc[meth]['width'] += float(np.nansum(w.mean(1)) + np.nansum(w2.mean(1))) / 2
                (loA, hiA) = bs.intervals(YA)
                (loB, hiB) = bs.intervals(YB)
                acc[meth]['well_miss'] += int(((loA > a[None, :] + 1e-12) | (hiA < a[None, :] - 1e-12)).sum() + ((loB > a[None, :] + 1e-12) | (hiB < a[None, :] - 1e-12)).sum())
            acc[meth]['cover'] += int(cov.sum())
        done += m_
    rows = []
    for (meth, d) in acc.items():
        (lo, hi) = K.wilson(d['cover'], n)
        rows.append(dict(grid=f'{I}x{I}', noise=noise, base=base, ms=ms, method=meth, n=n, c=specs[meth].c, gamma_well=specs[meth].gamma if meth != 'plugin_sd' else BETA / 8, coverage=d['cover'] / n, cov_lo=lo, cov_hi=hi, target=1 - BETA, n_empty=d['empty'], n_miss_lo=d['miss_lo'], n_miss_hi=d['miss_hi'], per_well_miss=d['well_miss'] / (n * nw), mean_width=d['width'] / n))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000000)
    ap.add_argument('--ext', action='store_true', help='extension: 5 x 5, m_s = 64 and 256 only, output bb_coverage_ext.csv')
    args = ap.parse_args()
    if K.paths.SMOKE:
        args.n = 100
    if args.ext:
        rows = []
        for noise in K.NOISES:
            for base in ('weak', 'moderate', 'strong'):
                for ms in (64, 256):
                    rows += run_setting(5, noise, base, ms, args.n)
                    print(noise, base, ms, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(K.RES, 'bb_coverage_ext.csv'), index=False)
        pd.set_option('display.width', 250)
        print(df.drop(columns=['target']).round(5).to_string(index=False))
        return
    for noise in K.NOISES:
        for ms in (1, 4, 16):
            bs = K.BandSpec(noise, ms, BETA, 8, 'lower')
            mm = np.linspace(0, 1.2, 24001)
            h = mm + bs.c * np.array([K.sd_well(v, K.KIND[noise], K.KV, K.KS) for v in mm]) / np.sqrt(ms)
            assert np.all(np.diff(h) > 0), (noise, ms)
    rng = np.random.default_rng(K.seed('cov', 'lnq'))
    chk = []
    for ms in (1, 4, 16):
        g = K.sidak_gamma(BETA, 8)
        q = K.logn_mean_quantiles(ms, [g / 2, 1 - g / 2])
        R = np.zeros(100 if K.paths.SMOKE else 10000000)
        for r in range(ms):
            R += np.exp(K.SL_LOGN * rng.standard_normal(100 if K.paths.SMOKE else 10000000) - 0.5 * K.SL_LOGN ** 2)
        R /= ms
        chk.append(dict(ms=ms, q_lo=q[0], q_hi=q[1], mc_tail_lo=(R < q[0]).mean(), mc_tail_hi=(R > q[1]).mean(), target_tail=g / 2))
        del R
    chk = pd.DataFrame(chk)
    print(chk.to_string(index=False), flush=True)
    chk.to_csv(os.path.join(K.RES, 'bb_quantile_check.csv'), index=False)
    rows = []
    t0 = time.time()
    for I in (5, 4):
        for noise in K.NOISES:
            for base in ('weak', 'moderate', 'strong'):
                for ms in (1, 4, 16):
                    rows += run_setting(I, noise, base, ms, args.n)
                    print(I, noise, base, ms, f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(K.RES, 'bb_coverage.csv'), index=False)
    pd.set_option('display.width', 250)
    print(df.drop(columns=['target']).round(5).to_string(index=False))
if __name__ == '__main__':
    main()
