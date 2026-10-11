"""sim_screen calculations."""
import argparse
import os
import time
import mc
import numpy as np
import pandas as pd
from sirsyn.replicates import screen as rep_screen
(S, sim) = (mc.S, mc.sim)
LAWS = {'A': lambda : S.ParametricNoise.additive(0.08), 'B': lambda : S.ParametricNoise.lognormal(0.12), 'C': lambda : S.ParametricNoise.oneil()}
SURFS = ('weak', 'moderate', 'strong')
NREP = 4
LEVELS = {'size': (0.0, 0.004, 0.01, 0.02), 'power': (0.0, 0.02), 'confirm': (0.0, 0.02)}
POWER_CONDS = [('moderate', f'syn_broad_{a}') for a in ('0.05', '0.10')] + [('moderate', f'syn_hdblock_{a}') for a in ('0.20', '0.40')] + [('weak', f'syn_corner_{a}') for a in ('0.50', '1.00')] + [('strong', f'sat_{g}') for g in ('0.15', '0.30')]
CHUNK = 250
NSTD = 2
MAGS = np.load(os.path.join(mc.INPUTS, 'gross_errors.npz'))['absdev']

def run_job(job):
    (part, I, surf, cond, law, start, n) = job
    t0 = time.time()
    nz = LAWS[law]()
    mu = sim.truth(cond, surf, I)
    ds = mc.data_seed('fig6_screen', part, I, surf, cond, law, start)
    cs = mc.data_seed('fig6_screen', 'cont', part, I, surf, cond, law, start)
    Rs = sim.simulate(mu, nz, 'S', n, np.random.default_rng(ds), m=NREP)
    rows = []
    for p in LEVELS[part]:
        (Rc, cm) = sim.contaminate(Rs, np.random.default_rng(cs), p, MAGS)
        for t in range(n):
            ne = int(cm[t, 0, :].sum() + cm[t, :, 0].sum())
            for arm in ('noscreen', 'screen'):
                r = mc.fit_sim(Rc[t], nz, 'R', [ds, t], screen_replicates=arm == 'screen', n_std=NSTD)
                rows.append(dict(t=t, start=start, level=p, arm=arm, p=r.p_tstd, n_cont=int(cm[t].sum()), n_cont_edge=ne, n_screened=len(r.log.screened), ref_flags=float(r.log.reference_flags_per_plate or 0.0)))
    df = pd.DataFrame(rows)
    df['p'] = df['p'].astype('float32')
    (df['part'], df['I'], df['surface'], df['cond'], df['law']) = (part, I, surf, cond, law)
    df['secs'] = time.time() - t0
    return df

def jobs_for(part, n_override=0, cell=None):
    J = []
    if part == 'confirm':
        (I, law, surf) = cell
        N = n_override or 20000
        return [(part, I, surf, 'null', law, s, min(CHUNK, N - s)) for s in range(0, N, CHUNK)]
    for I in (5, 4):
        for law in 'ABC':
            if part == 'size':
                conds = [(s, 'null') for s in SURFS]
                N = n_override or 2000
            else:
                conds = POWER_CONDS
                N = n_override or 1000
            for (surf, cond) in conds:
                for s in range(0, N, CHUNK):
                    J.append((part, I, surf, cond, law, s, min(CHUNK, N - s)))
    return J

def check(n_sets):
    rows = []
    for law in 'ABC':
        nz = LAWS[law]()
        for I in (5, 4):
            for (surf, cond) in POWER_CONDS:
                mu1 = sim.truth(cond, surf, I)
                mu0 = sim.truth('null', surf, I)
                syn = np.zeros((I, I), bool)
                syn[1:, 1:] = sim.bliss_excess(mu1) > 1e-09
                rng = np.random.default_rng(mc.data_seed('fig6_screen', 'check', law, I, cond))
                e = nz.standard(rng, (n_sets, NREP, I, I))
                (R1, R0) = (nz.generate(mu1, e), nz.generate(mu0, e))
                R1[..., 0, 0] = R0[..., 0, 0] = 1.0
                (m1, d1) = rep_screen(R1, 5.0, nz.sd)
                (m0, d0) = rep_screen(R0, 5.0, nz.sd)
                (f1, f0) = ((d1 >= 0)[:, syn], (d0 >= 0)[:, syn])
                rows.append(dict(law=law, I=I, cond=cond, surface=surf, n_sets=n_sets, n_syn_wells=int(syn.sum()), flag_rate_alt=f1.mean(), flag_rate_null=f0.mean(), share_flag_status_differs=(f1 != f0).mean(), bias_screened=float((m1[:, syn] - mu1[syn]).mean()), bias_unscreened=float((R1.mean(1)[:, syn] - mu1[syn]).mean())))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['size', 'power', 'confirm', 'check'])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--cell', default='5,B,weak')
    a = mc.smoke_args(ap.parse_args())
    tag = f'_n{a.n}' if a.n and not mc.paths.SMOKE else ''
    t0 = time.time()
    if a.part == 'check':
        res = check(a.n or 20000)
        print(res.groupby('law')[['flag_rate_alt', 'flag_rate_null', 'share_flag_status_differs']].agg(['mean', 'max']))
    else:
        c = a.cell.split(',')
        J = jobs_for(a.part, a.n, (int(c[0]), c[1], c[2]))
        if a.part == 'confirm':
            tag = f'_{c[0]}{c[1]}{c[2]}' + tag
        J.sort(key=lambda j: -j[1])
        print(f'{a.part}: {len(J)} jobs, {sum((j[6] for j in J))} datasets', flush=True)
        out = mc.pool_map(run_job, J, workers=mc.workers(a.workers), progress=20)
        res = pd.concat(out, ignore_index=True)
    path = os.path.join(mc.RESDIR, f'fig6_screen_{a.part}{tag}.parquet')
    res.to_parquet(path, index=False)
    print(f'done {a.part}: {len(res)} rows, {time.time() - t0:.0f} s -> {path}', flush=True)
if __name__ == '__main__':
    main()
