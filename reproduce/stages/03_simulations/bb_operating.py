"""Berger–Boos oracle reference, amplitude selection and supremum search."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_core as K
G = K.G5
(BETA, ALPHA) = (0.005, 0.05)
B_REF = 999
B_OR = 200000
B_FRESH = 9999
MS = (1, 4, 16)
MS_EXT = (64, 256)
BASES = ('weak', 'moderate', 'strong')
SIZE_CONDS = {'bliss': 0.0, 'ant_broad': 0.25, 'ant_high': 0.6}
POWER_CONDS = ('syn_broad', 'syn_high', 'syn_corner', 'syn_sat')
AMP_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
(N_SIZE, N_POWER) = (2000, 1000)
(N_SEARCH_RAND, N_SEARCH_REJ_CAP) = (100, 250)
CACHE = os.path.join(K.RES, 'bb_cache')
os.makedirs(CACHE, exist_ok=True)

def lo_clip(noise):
    return 0.001 if noise == 'logn' else 0.0

def ref_path(noise, base, ms):
    return os.path.join(CACHE, f'bb_oracle_{noise}_{base}_ms{ms}.npy')

def job_ref(args):
    (noise, base, ms) = args
    (a, b) = K.base_ab(base)
    rng = np.random.default_rng(K.seed('oracle', noise, base, ms))
    out = []
    for _ in range(max(1, B_OR // 50000)):
        (ZA, ZB, ZI) = K.draw_noise(rng, min(B_OR, 50000), 5, 5, ms, noise)
        out.append(K.ref_T_bliss(a, b, 5, 5, ms, K.KIND[noise], ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS))
    T = np.sort(np.concatenate(out))
    np.save(ref_path(noise, base, ms), T)
    return (noise, base, ms)

def p_from_sorted(srt, t):
    n = len(srt)
    k = n - np.searchsorted(srt, np.asarray(t) - 1e-09 * np.maximum(1.0, np.abs(t)), side='left')
    return (1 + k) / (n + 1)

def job_calib(args):
    (noise, base, cond) = args
    srt = np.load(ref_path(noise, base, 4))
    (a, b) = K.base_ab(base)
    rows = []
    for amp in AMP_GRID:
        rng = np.random.default_rng(K.seed('calib', noise, base, cond, amp))
        Y = K.gen_data(rng, 4 if K.paths.SMOKE else 4000, a, b, K.truth_int(base, cond, amp), 5, 5, 4, noise)
        T = K.T_batch_nb(Y, 5, 5, G.pa, G.pb, G.w, K.KIND[noise], K.KV, K.KS)
        rows.append(dict(noise=noise, base=base, cond=cond, amp=amp, oracle_power=float((p_from_sorted(srt, T) <= ALPHA).mean())))
    return rows

def job_oraclecheck(args):
    (noise, base, ms) = args
    srt = np.load(ref_path(noise, base, ms))
    (a, b) = K.base_ab(base)
    rng = np.random.default_rng(K.seed('oraclecheck', noise, base, ms))
    Y = K.gen_data(rng, 100000, a, b, np.outer(a, b).ravel(), 5, 5, ms, noise)
    T = K.T_batch_nb(Y, 5, 5, G.pa, G.pb, G.w, K.KIND[noise], K.KV, K.KS)
    p = p_from_sorted(srt, T)
    return dict(noise=noise, base=base, ms=ms, n=len(T), k05=int((p <= 0.05).sum()), k01=int((p <= 0.01).sum()), rate05=float((p <= 0.05).mean()), rate01=float((p <= 0.01).mean()))

def job_oraclecheck2(args):
    (noise, base, ms, rep) = args
    (a, b) = K.base_ab(base)
    rng = np.random.default_rng(K.seed('oraclecheck2', noise, base, ms, rep))
    (ZA, ZB, ZI) = K.draw_noise(rng, 200000, 5, 5, ms, noise)
    ref = np.sort(K.ref_T_bliss(a, b, 5, 5, ms, K.KIND[noise], ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS))
    (ZA, ZB, ZI) = K.draw_noise(rng, 200000, 5, 5, ms, noise)
    T = K.ref_T_bliss(a, b, 5, 5, ms, K.KIND[noise], ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
    p = p_from_sorted(ref, T)
    return dict(noise=noise, base=base, ms=ms, rep=rep, n=len(T), k05=int((p <= 0.05).sum()), k01=int((p <= 0.01).sum()))

def load_amps():
    df = pd.read_csv(os.path.join(K.RES, 'bb_amplitudes.csv'))
    return {(r.noise, r.base, r.cond): r.amp for r in df.itertuples()}

def ref_noise(noise, base, ms, cond, k, B=None, tag='ref'):
    if B is None: B = B_REF
    rng = np.random.default_rng(K.seed(tag, noise, base, ms, cond, k))
    return K.draw_noise(rng, B, 5, 5, ms, noise)

def analyze(Y, noise, ms, Z, spec2, specL):
    kind = K.KIND[noise]
    (ZA, ZB, ZI) = Z
    Bn = ZI.shape[0]
    t = K.t_std(Y, 5, 5, G.pa, G.pb, G.w, kind, K.KV, K.KS)
    (yA, yB) = (Y[G.EA], Y[G.EB])
    lo = lo_clip(noise)
    aP = np.clip(K.pava_dec(yA), lo, 1.2)
    bP = np.clip(K.pava_dec(yB), lo, 1.2)
    c_pi = K.count_one(aP, bP, t, 5, 5, ms, kind, ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
    (LA, UA, okA) = spec2.band(yA)
    (LB, UB, okB) = spec2.band(yB)
    ok = bool(okA and okB)
    if ok:
        c_env = K.count_one(np.ascontiguousarray(UA), np.ascontiguousarray(UB), t, 5, 5, ms, kind, ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
    else:
        c_env = Bn
    (lA, _, okLA) = specL.band(yA)
    (lB, _, okLB) = specL.band(yB)
    okL = bool(okLA and okLB)
    if okL:
        ell = np.outer(lA, lB).ravel()
        c_cond = K.count_conditional(Y, ell, t, 5, 5, kind, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
    else:
        c_cond = Bn
    p = lambda c: (1 + c) / (Bn + 1)
    return dict(T=t, band_ok=ok, p_plugin=p(c_pi), c_env=int(c_env), p_env=p(c_env), p_BB_env=min(1.0, p(c_env) + BETA) if ok else 1.0, p_cond=min(1.0, p(c_cond) + BETA) if okL else 1.0, width_A=float(np.mean(UA - LA)) if okA else np.nan, width_B=float(np.mean(UB - LB)) if okB else np.nan)

def rows_all(amps):
    R = []
    for noise in K.NOISES:
        for base in BASES:
            for ms in MS:
                for (cond, amp) in SIZE_CONDS.items():
                    R.append((noise, base, ms, cond, amp, N_SIZE, 'size'))
                for cond in POWER_CONDS:
                    R.append((noise, base, ms, cond, amps[noise, base, cond], N_POWER, 'power'))
    return R

def gen_row(noise, base, ms, cond, amp, n):
    (a, b) = K.base_ab(base)
    rng = np.random.default_rng(K.seed('data', noise, base, ms, cond))
    return K.gen_data(rng, n, a, b, K.truth_int(base, cond, amp), 5, 5, ms, noise)

def job_oc(args):
    (noise, base, ms, cond, amp, n, kind_row) = args
    t0 = time.time()
    Y = gen_row(noise, base, ms, cond, amp, n)
    spec2 = K.BandSpec(noise, ms, BETA, 8, 'exact')
    specL = K.BandSpec(noise, ms, BETA, 8, 'lower')
    srt = np.load(ref_path(noise, base, ms))
    out = []
    for k in range(n):
        Z = ref_noise(noise, base, ms, cond, k)
        rec = analyze(np.ascontiguousarray(Y[k]), noise, ms, Z, spec2, specL)
        rec['p_oracle'] = float(p_from_sorted(srt, rec['T']))
        rec.update(noise=noise, base=base, ms=ms, cond=cond, amp=amp, row=kind_row, k=k)
        out.append(rec)
    print(noise, base, ms, cond, f'{time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(out)

def cand_type(i, n=4, n_flat=6, n_rand=5):
    if i == 0:
        return 'U'
    if i == 1:
        return 'L'
    if i == 2:
        return 'iso'
    if i < 3 + n_flat:
        return 'flat'
    if i < 3 + n_flat + (n - 1):
        return 'step'
    return 'random'

def search(Y, noise, ms, Z, spec2, rng, c_env_main):
    kind = K.KIND[noise]
    (ZA, ZB, ZI) = Z
    t = K.t_std(Y, 5, 5, G.pa, G.pb, G.w, kind, K.KV, K.KS)
    (yA, yB) = (Y[G.EA], Y[G.EB])
    (LA, UA, okA) = spec2.band(yA)
    (LB, UB, okB) = spec2.band(yB)
    if not (okA and okB):
        return None
    CA = K.candidates(yA, LA, UA, rng, n_flat=6, n_rand=5)
    CB = K.candidates(yB, LB, UB, rng, n_flat=6, n_rand=5)
    pairs = np.array([(i, j) for i in range(len(CA)) for j in range(len(CB))], dtype=np.int64)
    args = (5, 5, ms, kind, ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
    c_env = K.count_one(CA[0], CB[0], t, *args)
    assert c_env == c_env_main, (c_env, c_env_main)
    cnt = K.count_pairs_es(CA, CB, pairs[1:], t, c_env, *args)
    best = c_env
    q_best = -1
    if cnt.max() > best:
        q_best = int(np.argmax(cnt))
        best = int(cnt[q_best])
    if q_best >= 0:
        (ia, ib) = pairs[1 + q_best]
    else:
        (ia, ib) = (0, 0)
    (a, b) = (CA[ia].copy(), CB[ib].copy())
    prod_best = best
    n_eval = len(pairs)
    for sweep in range(2):
        improved = False
        for drug in (0, 1):
            cur = a if drug == 0 else b
            (L, U) = (LA, UA) if drug == 0 else (LB, UB)
            for i in range(4):
                lo_i = max(L[i], cur[i + 1] if i < 3 else L[i])
                hi_i = min(U[i], cur[i - 1] if i > 0 else U[i])
                if hi_i - lo_i < 1e-09:
                    continue
                for v in np.linspace(lo_i, hi_i, 5):
                    cand = cur.copy()
                    cand[i] = v
                    (aa, bb) = (cand, b) if drug == 0 else (a, cand)
                    c = K.count_one_es(np.ascontiguousarray(aa), np.ascontiguousarray(bb), t, best, *args)
                    n_eval += 1
                    if c > best:
                        best = c
                        cur[i] = v
                        improved = True
            if drug == 0:
                a = cur
            else:
                b = cur
        if not improved:
            break
    rec = dict(T=t, c_env=int(c_env), c_prod=int(prod_best), c_search=int(best), n_eval=n_eval, argmax_A=cand_type(int(ia)), argmax_B=cand_type(int(ib)), ascent_gain=int(best - prod_best), a_star=np.round(a, 5).tolist(), b_star=np.round(b, 5).tolist(), UA=np.round(UA, 5).tolist(), UB=np.round(UB, 5).tolist())
    return (rec, a, b, UA, UB)

def job_search(args):
    (noise, base, ms, cond, amp, n, kind_row, ks, c_env_list) = args
    t0 = time.time()
    Y = gen_row(noise, base, ms, cond, amp, n)
    spec2 = K.BandSpec(noise, ms, BETA, 8, 'exact')
    kind = K.KIND[noise]
    out = []
    for (k, c_env_main) in zip(ks, c_env_list):
        Z = ref_noise(noise, base, ms, cond, k)
        rng = np.random.default_rng(K.seed('cands', noise, base, ms, cond, k))
        t1 = time.time()
        res = search(np.ascontiguousarray(Y[k]), noise, ms, Z, spec2, rng, int(c_env_main))
        if res is None:
            continue
        (rec, a, b, UA, UB) = res
        rec['sec'] = time.time() - t1
        if rec['c_search'] > rec['c_env']:
            (ZA, ZB, ZI) = ref_noise(noise, base, ms, cond, k, B=B_FRESH, tag='fresh')
            args2 = (5, 5, ms, kind, ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
            TU = K.ref_T_bliss(np.ascontiguousarray(UA), np.ascontiguousarray(UB), *args2)
            Ts = K.ref_T_bliss(np.ascontiguousarray(a), np.ascontiguousarray(b), *args2)
            thr = rec['T'] - 1e-09 * max(1.0, rec['T'])
            d = (Ts >= thr).astype(float) - (TU >= thr)
            rec.update(pf_env=(1 + (TU >= thr).sum()) / (B_FRESH + 1), pf_star=(1 + (Ts >= thr).sum()) / (B_FRESH + 1), pf_diff=d.mean(), pf_diff_se=d.std(ddof=1) / np.sqrt(B_FRESH))
        rec.update(noise=noise, base=base, ms=ms, cond=cond, amp=amp, row=kind_row, k=int(k))
        out.append(rec)
    print('search', noise, base, ms, cond, len(ks), f'{time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(out)

def search_jobs(oc, chunk=25):
    J = []
    amps = load_amps()
    for (noise, base, ms, cond, amp, n, kind_row) in rows_all(amps):
        s = oc[(oc.noise == noise) & (oc.base == base) & (oc.ms == ms) & (oc.cond == cond)]
        rng = np.random.default_rng(K.seed('searchpick', noise, base, ms, cond))
        pick = set(rng.choice(s.k.to_numpy(), min(len(s), N_SEARCH_RAND), replace=False).tolist())
        rej = s[(s.p_BB_env <= ALPHA) & s.band_ok].k.to_numpy()
        if len(rej) > N_SEARCH_REJ_CAP:
            rej = rng.choice(rej, N_SEARCH_REJ_CAP, replace=False)
        pick |= set(rej.tolist())
        ks = np.array(sorted(pick))
        ok = s.set_index('k').loc[ks]
        ks = ks[ok.band_ok.to_numpy()]
        cenv = s.set_index('k').loc[ks].c_env.to_numpy()
        for i in range(0, len(ks), chunk):
            J.append((noise, base, ms, cond, amp, n, kind_row, ks[i:i + chunk], cenv[i:i + chunk]))
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    W = K.paths.WORKERS
    if K.paths.SMOKE:
        global B_OR, N_SIZE, N_POWER, B_REF, B_FRESH, N_SEARCH_RAND, N_SEARCH_REJ_CAP
        (B_OR, N_SIZE, N_POWER, B_REF, B_FRESH) = (100, 4, 4, 39, 99)
        (N_SEARCH_RAND, N_SEARCH_REJ_CAP) = (2, 2)
    t0 = time.time()
    if a.part == 'refs':
        jobs = [(n, b, m) for n in K.NOISES for b in BASES for m in MS]
        with Pool(W) as p:
            p.map(job_ref, jobs)
    elif a.part == 'calib':
        jobs = [(n, b, c) for n in K.NOISES for b in BASES for c in POWER_CONDS]
        with Pool(W) as p:
            res = p.map(job_calib, jobs)
        df = pd.DataFrame([r for rr in res for r in rr])
        df.to_csv(os.path.join(K.RES, 'bb_amplitude_grid.csv'), index=False)
        pick = []
        for ((noise, base, cond), s) in df.groupby(['noise', 'base', 'cond']):
            i = (s.oracle_power - 0.7).abs().idxmin()
            pick.append(s.loc[i])
        pk = pd.DataFrame(pick)
        pk.to_csv(os.path.join(K.RES, 'bb_amplitudes.csv'), index=False)
        print(pk.to_string(index=False))
    elif a.part == 'oc':
        jobs = rows_all(load_amps())
        jobs.sort(key=lambda r: -r[5])
        with Pool(W) as p:
            res = p.map(job_oc, jobs, chunksize=1)
        df = pd.concat(res, ignore_index=True)
        df['B'] = B_REF
        df.to_parquet(os.path.join(K.RES, 'bb_operating.parquet'), index=False)
    elif a.part == 'search':
        oc = pd.read_parquet(os.path.join(K.RES, 'bb_operating.parquet'))
        jobs = search_jobs(oc)
        print('search jobs', len(jobs), 'datasets', sum((len(j[7]) for j in jobs)), flush=True)
        with Pool(W) as p:
            res = p.map(job_search, jobs, chunksize=1)
        df = pd.concat(res, ignore_index=True)
        df['B'] = B_REF
        df.to_parquet(os.path.join(K.RES, 'bb_search.parquet'), index=False)
    elif a.part == 'ocext':
        jobs = [(n, b, m) for n in K.NOISES for b in BASES for m in MS_EXT]
        with Pool(W) as p:
            p.map(job_ref, jobs)
        amps = load_amps()
        jobs = []
        for noise in K.NOISES:
            for base in BASES:
                for ms in MS_EXT:
                    jobs.append((noise, base, ms, 'bliss', 0.0, N_SIZE, 'size'))
                    for cond in POWER_CONDS:
                        jobs.append((noise, base, ms, cond, amps[noise, base, cond], N_POWER, 'power'))
        jobs.sort(key=lambda r: -r[2] * r[5])
        with Pool(W) as p:
            res = p.map(job_oc, jobs, chunksize=1)
        df = pd.concat(res, ignore_index=True)
        df['B'] = B_REF
        df.to_parquet(os.path.join(K.RES, 'bb_operating_ext.parquet'), index=False)
    elif a.part == 'searchext':
        oc = pd.read_parquet(os.path.join(K.RES, 'bb_operating_ext.parquet'))
        oc = oc[oc.ms == 64]
        J = []
        for ((noise, base, ms, cond), s) in oc.groupby(['noise', 'base', 'ms', 'cond']):
            rej = s[(s.p_BB_env <= ALPHA) & s.band_ok].k.to_numpy()
            rng = np.random.default_rng(K.seed('searchextpick', noise, base, ms, cond))
            if cond != 'bliss' and len(rej) > 100:
                rej = np.sort(rng.choice(rej, 100, replace=False))
            cenv = s.set_index('k').loc[rej].c_env.to_numpy()
            (amp, n, kr) = (s.amp.iloc[0], N_SIZE if cond == 'bliss' else N_POWER, s.row.iloc[0])
            for i in range(0, len(rej), 20):
                J.append((noise, base, ms, cond, amp, n, kr, rej[i:i + 20], cenv[i:i + 20]))
        print('searchext jobs', len(J), 'datasets', sum((len(j[7]) for j in J)), flush=True)
        with Pool(W) as p:
            res = p.map(job_search, J, chunksize=1)
        df = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=['noise', 'base', 'ms', 'cond', 'k', 'c_search'])
        df['B'] = B_REF
        df.to_parquet(os.path.join(K.RES, 'bb_search_ext.parquet'), index=False)
    print(f'[{a.part}] {time.time() - t0:.0f}s')
if __name__ == '__main__':
    main()
