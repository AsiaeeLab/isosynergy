"""Numerical propositions in Supplementary Sections S2 and S3."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proposition_core as C
RES = C.RES
LOG = str(C.paths.LOGS)
os.makedirs(LOG, exist_ok=True)
C.set_threads(1)
ALPHAS = [0.05, 0.01, 0.005, 0.001]

def say(*a):
    print(*a, flush=True)

def p1():
    rng = np.random.default_rng(C.seed('p1'))
    rows = []
    worst = dict(dmu=np.inf, deta=-np.inf, dTp=-np.inf, dTm=-np.inf, da=0.0)
    n_edge_up = 0
    n_pairs = 0
    for wkind in ['equal', 'lognormal_fixed', 'edge_derived']:
        for setting in range(6):
            for rep in range(4 if C.paths.SMOKE else 1500):
                if setting < 4:
                    kind = ['weak', 'moderate', 'strong', 'potent'][setting]
                    (ya, yb) = C.base_curves(kind)
                    mu = C.bliss_surface(ya, yb)
                    mu = C.with_eta(mu, rng.normal(0, 0.1, 16))
                    Y = mu + C.sd_oneil(np.clip(mu, 0, 1.2)) * rng.standard_normal(25)
                elif setting == 4:
                    Y = rng.standard_normal(25)
                else:
                    Y = np.round(rng.uniform(0, 1.2, 25), 1)
                Y[0] = 1.0
                if wkind == 'equal':
                    w = C.W1.copy()
                elif wkind == 'lognormal_fixed':
                    w = np.exp(rng.normal(0, 1, 25))
                else:
                    a = C.pava_dec(Y[C.EA], np.ones(4))
                    b = C.pava_dec(Y[C.EB], np.ones(4))
                    pred = np.ones(25)
                    pred[C.EA] = np.clip(Y[C.EA], 0, 1.2)
                    pred[C.EB] = np.clip(Y[C.EB], 0, 1.2)
                    pred[C.INT] = np.clip(np.outer(a, b).ravel(), 0, 1.2)
                    w = 1.0 / C.sd_oneil(pred) ** 2
                ws = w[C.INT].copy()
                inc = np.zeros(25)
                mode = rep % 3
                if mode == 0:
                    inc[rng.choice(C.INT)] = rng.exponential(0.3)
                elif mode == 1:
                    k = rng.integers(2, 6)
                    inc[rng.choice(C.INT, k, replace=False)] = rng.exponential(0.2, k)
                else:
                    inc[C.INT] = rng.exponential(0.1, 16)
                Y2 = Y + inc
                (tp1, tm1, e1, m1, a1, b1) = C.pipe(Y, w, ws, C.PA, C.PB, C.TOL, C.SWEEPS)
                (tp2, tm2, e2, m2, a2, b2) = C.pipe(Y2, w, ws, C.PA, C.PB, C.TOL, C.SWEEPS)
                dmu = (m2 - m1).min()
                deta = (e2 - e1).max()
                n_pairs += 1
                worst['dmu'] = min(worst['dmu'], dmu)
                worst['deta'] = max(worst['deta'], deta)
                worst['dTp'] = max(worst['dTp'], tp2 - tp1)
                worst['dTm'] = max(worst['dTm'], tm2 - tm1)
                worst['da'] = max(worst['da'], np.abs(a2 - a1).max(), np.abs(b2 - b1).max())
                edge_up = max((m2[C.EA] - m1[C.EA]).max(), (m2[C.EB] - m1[C.EB]).max())
                n_edge_up += edge_up > 1e-08
                rows.append(dict(weights=wkind, setting=setting, mode=mode, dmu_min=dmu, deta_max=deta, dTp=tp2 - tp1, dTm=tm2 - tm1, fitted_edge_rise=edge_up))
    df = pd.DataFrame(rows)
    summ = df.groupby('weights').agg(n=('dmu_min', 'size'), dmu_min=('dmu_min', 'min'), deta_max=('deta_max', 'max'), dTp_max=('dTp', 'max'), dTm_max=('dTm', 'max'), frac_fitted_edge_rises=('fitted_edge_rise', lambda x: (x > 1e-08).mean()))
    summ.to_csv(os.path.join(RES, 'order_order_preservation.csv'))
    say('P1 order preservation, pipeline (1D edge fits + P_M on 25 wells), fixed weights')
    say(summ.to_string())
    say('worst over all', worst, 'pairs', n_pairs)
    rng = np.random.default_rng(C.seed('p1', 'pmedge'))
    (ya, yb) = C.base_curves('weak')
    mu = C.bliss_surface(ya, yb)
    nviol = 0
    best = None
    ntry = 2000 if C.paths.SMOKE else 20000
    for t in range(ntry):
        Y = mu + 0.05 * rng.standard_normal(25)
        Y[0] = 1.0
        inc = np.zeros(25)
        inc[rng.choice(C.INT)] = rng.exponential(0.2)
        r1 = C.pipe_pmedge(Y, C.W1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
        r2 = C.pipe_pmedge(Y + inc, C.W1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
        d = r2[0] - r1[0]
        if d > 1e-09:
            nviol += 1
            if best is None or d > best[0]:
                best = (d, Y.tolist(), inc.tolist(), r1[0], r2[0])
    say(f'P1-A edges from P_M: T_plus increased after raising one interior well in {nviol} of {ntry} trials; largest increase {best[0]:.4g} (from {best[3]:.4g} to {best[4]:.4g})')
    Y = mu.copy()
    Y[0] = 1.0
    r1 = C.pipe_pmedge(Y, C.W1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
    Y2 = Y.copy()
    Y2[1 * 5 + 1] = 1.05
    r2 = C.pipe_pmedge(Y2, C.W1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
    separate_fit = C.pipe(Y2, C.W1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
    exA = dict(Y_before=Y.reshape(5, 5).round(4).tolist(), Y_after_cell11=1.05, Tplus_pmedge_before=r1[0], Tplus_pmedge_after=r2[0], eta_pmedge_after=r2[2].reshape(4, 4).round(4).tolist(), Tplus_separate_edges_after=separate_fit[0])
    say('P1-A deterministic (noise-free weak Bliss surface, cell (1,1) raised to 1.05):', json.dumps({k: v for (k, v) in exA.items() if k.startswith('Tplus')}))
    rng = np.random.default_rng(C.seed('p1', 'ddw'))
    nB = 0
    nB_mu = 0
    bestB = None
    ntry = 2000 if C.paths.SMOKE else 20000
    for t in range(ntry):
        kind = ['weak', 'moderate', 'strong', 'potent'][t % 4]
        (ya, yb) = C.base_curves(kind)
        mu = C.bliss_surface(ya, yb)
        Y = mu + C.sd_oneil(mu) * rng.standard_normal(25)
        Y[0] = 1.0
        inc = np.zeros(25)
        inc[rng.choice(C.INT)] = rng.exponential(0.1)
        Y2 = Y + inc
        w1 = 1.0 / C.sd_oneil(np.clip(Y, 0, 1.2)) ** 2
        w2 = 1.0 / C.sd_oneil(np.clip(Y2, 0, 1.2)) ** 2
        r1 = C.pipe(Y, w1, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
        r2 = C.pipe(Y2, w2, C.WS1, C.PA, C.PB, C.TOL, C.SWEEPS)
        dmu = (r2[3] - r1[3]).min()
        nB_mu += dmu < -1e-09
        if r2[0] - r1[0] > 1e-09:
            nB += 1
            if bestB is None or r2[0] - r1[0] > bestB[0]:
                bestB = (r2[0] - r1[0], r1[0], r2[0], dmu, kind)
    say(f'P1-B data-dependent weights 1/s(Y)^2: some fitted mean decreased in {nB_mu} of {ntry}; T_plus increased in {nB} of {ntry}; largest increase {bestB[0]:.4g} (from {bestB[1]:.4g} to {bestB[2]:.4g}, base {bestB[4]})')
    y = np.array([0.02, 0.1])
    s = lambda m: C.sd_oneil(m)
    f1 = (y / s(y) ** 2).sum() / (1 / s(y) ** 2).sum()
    y2 = np.array([0.02, 0.12])
    f2 = (y2 / s(y2) ** 2).sum() / (1 / s(y2) ** 2).sum()
    say(f'P1-B hand example (chain of 2, weights 1/s(y)^2): y = (0.02, 0.10) -> fit {f1:.5f}; y = (0.02, 0.12) -> fit {f2:.5f}')
    out = dict(worst=worst, n_pairs=n_pairs, pmedge=dict(n_viol=nviol, n=ntry, best_increase=best[0]), pmedge_example=exA, ddw=dict(n_T_viol=nB, n_mu_viol=nB_mu, n=ntry, best=bestB[:4]), chain=dict(y1=[0.02, 0.1], fit1=f1, y2=[0.02, 0.12], fit2=f2))
    json.dump(out, open(os.path.join(RES, 'order_summary.json'), 'w'), indent=1, default=float)

def p3():
    N = stats.norm
    out = []
    curves = {}
    for col in ['pooled_sd']:
        (kv, ks) = C.oneil_curve(col, 'identity_raw')
        curves[col] = (kv, ks)
    seg_rows = []
    for (name, (kv, ks)) in curves.items():
        for k in range(len(kv) - 1):
            (m0, m1) = (kv[k], kv[k + 1])
            sl = (ks[k + 1] - ks[k]) / (m1 - m0)
            mass = N.cdf(-1 / abs(sl)) if sl != 0 else 0.0
            mm = np.linspace(m0, m1, 201)
            ss = np.interp(mm, kv, ks)
            if sl > 0:
                un = np.maximum(N.cdf(-1 / sl) - N.cdf(-mm / ss), 0).max()
            else:
                un = mass
            (cv0, cv1) = (ks[k] / m0, ks[k + 1] / m1)
            seg_rows.append(dict(curve=name, mu_lo=m0, mu_hi=m1, sd_lo=ks[k], sd_hi=ks[k + 1], slope=sl, viol_mass_nofloor=mass, viol_mass_floor0=un, cv_lo=cv0, cv_hi=cv1, tail='lower' if sl > 0 else 'upper' if sl < 0 else 'none'))
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(os.path.join(RES, 'noise_reversal_segments.csv'), index=False)
    pd.set_option('display.width', 220)
    say(seg.round(5).to_string(index=False))
    (kv, ks) = curves['pooled_sd']
    rng = np.random.default_rng(C.seed('p3'))
    z = rng.standard_normal(100 if C.paths.SMOKE else 2000000)
    grid = np.linspace(0.01, 1.2, 120)
    fin = []
    for (i0, m0) in enumerate(grid):
        s0 = np.interp(m0, kv, ks)
        for m1 in grid[i0 + 1:]:
            s1 = np.interp(m1, kv, ks)
            if s1 > s0:
                p_nf = N.cdf(-(m1 - m0) / (s1 - s0))
                zc = -(m1 - m0) / (s1 - s0)
                p_f = max(N.cdf(zc) - N.cdf(-m0 / s0), 0.0)
            elif s1 < s0:
                p_nf = N.cdf(-(m1 - m0) / (s0 - s1))
                p_f = p_nf
            else:
                p_nf = p_f = 0.0
            fin.append(dict(m0=m0, m1=m1, s0=s0, s1=s1, p_viol_nofloor=p_nf, p_viol_floor0=p_f))
    fin = pd.DataFrame(fin)
    fin.to_csv(os.path.join(RES, 'noise_reversal_finite_shifts.csv.gz'), index=False)
    for (lab, sub) in [('mu in [0,1]', fin[fin.m1 <= 1.0]), ('mu in [0,1.2]', fin)]:
        for colp in ['p_viol_nofloor', 'p_viol_floor0']:
            r = sub.loc[sub[colp].idxmax()]
            say(f'{lab}: max {colp} = {r[colp]:.3g} at m0 = {r.m0:.3f} -> m1 = {r.m1:.3f}')
    say('median p_viol_nofloor over pairs m1 <= 1:', fin[fin.m1 <= 1].p_viol_nofloor.median())

def crit(tp, alphas=ALPHAS):
    return {a: np.quantile(tp, 1 - a) for a in alphas}

def p4(selection):
    rows = []
    (n_ref, n_ev) = (100, 100) if C.paths.SMOKE else (400000, 200000)
    models = [selection[0]]
    for model in models:
        for kind in [selection[1]]:
            (ya, yb) = C.base_curves(kind)
            mu0 = C.bliss_surface(ya, yb)
            rng = np.random.default_rng(C.seed('p4', model, kind, 'ref'))
            (tpr, tmr) = C.run(C.gen(model, mu0, C.draw_xi(rng, n_ref)))
            (cp, cm) = (crit(tpr), crit(tmr))
            rng = np.random.default_rng(C.seed('p4', model, kind, 'ev'))
            xi = C.draw_xi(rng, n_ev)
            Y0 = C.gen(model, mu0, xi)
            (tp0, tm0) = C.run(Y0)
            pats = []
            for v in range(16):
                for u in [0.02, 0.05, 0.1, 0.2, 0.4]:
                    e = np.zeros(16)
                    e[v] = -u
                    pats.append((f'cell{v // 4 + 1}{v % 4 + 1}_u{u}', e))
            for u in [0.02, 0.05, 0.1, 0.2]:
                pats.append((f'uniform_u{u}', -u * np.ones(16)))
                for r in range(4):
                    e = np.zeros((4, 4))
                    e[r, :] = -u
                    pats.append((f'row{r + 1}_u{u}', e.ravel()))
                    e = np.zeros((4, 4))
                    e[:, r] = -u
                    pats.append((f'col{r + 1}_u{u}', e.ravel()))
            for tgt in [0.15, 0.5, 0.75, 1.0, 1.1, 1.16, 1.2]:
                e = np.minimum(mu0[C.INT] - tgt, 0.0)
                pats.append((f'raise_to_{tgt}', e))
            for (name, e) in pats:
                mu = C.with_eta(mu0, e)
                mu[C.INT] = np.minimum(mu[C.INT], np.maximum(1.2, mu0[C.INT]))
                Y1 = C.gen(model, mu, xi)
                (tp1, tm1) = C.run(Y1)
                rec = dict(model=model, base=kind, pattern=name, max_mu=float(mu[C.INT].max()), coupling_viol=float((tp1 - tp0 > 1e-09).mean()))
                for a in ALPHAS:
                    r1 = tp1 >= cp[a]
                    r0 = tp0 >= cp[a]
                    d = r1.astype(float) - r0
                    rec[f'rej_{a}'] = r1.mean()
                    rec[f'rej0_{a}'] = r0.mean()
                    rec[f'diff_{a}'] = d.mean()
                    rec[f'se_{a}'] = d.std() / np.sqrt(n_ev)
                    rec[f'new_{a}'] = (r1 & ~r0).mean()
                    rec[f'mrej_{a}'] = (tm1 >= cm[a]).mean()
                rows.append(rec)
            say(model, kind, 'done')
    return rows

def p4_hill(selection):
    rows = []
    (n_ref, n_hc, n_val) = (100, 100, 100) if C.paths.SMOKE else (400000, 20000, 400000)
    for model in [selection[0]]:
        for kind in [selection[1]]:
            for a in [selection[2]]:
                (ya, yb) = C.base_curves(kind)
                mu0 = C.bliss_surface(ya, yb)
                rng = np.random.default_rng(C.seed('p4h', model, kind, a, 'ref'))
                (tpr, _) = C.run(C.gen(model, mu0, C.draw_xi(rng, n_ref)))
                c = np.quantile(tpr, 1 - a)
                rng = np.random.default_rng(C.seed('p4h', model, kind, a, 'hc'))
                xi = C.draw_xi(rng, n_hc)
                (tp0, _) = C.run(C.gen(model, mu0, xi))
                h = max(np.std(tpr) * 0.02, 1e-12)

                def score(u):
                    mu = mu0.copy()
                    mu[C.INT] = mu0[C.INT] + u
                    (tp, _) = C.run(C.gen(model, mu, xi))
                    return (1 / (1 + np.exp(-(tp - c) / h))).mean()
                ub = np.maximum(1.2 - mu0[C.INT], 0)
                (best_u, best) = (np.zeros(16), score(np.zeros(16)))
                for start in range(3):
                    u = np.zeros(16) if start == 0 else rng.uniform(0, 1, 16) * ub * (rng.uniform(size=16) < 0.3)
                    s = score(u)
                    step = 0.1
                    for it in range(2 if C.paths.SMOKE else 120):
                        cand = u.copy()
                        k = rng.integers(16)
                        cand[k] = np.clip(cand[k] + rng.normal(0, step), 0, ub[k])
                        sc = score(cand)
                        if sc > s:
                            (u, s) = (cand, sc)
                        if it % 40 == 39:
                            step *= 0.5
                    if s > best:
                        (best_u, best) = (u.copy(), s)
                rng = np.random.default_rng(C.seed('p4h', model, kind, a, 'val'))
                xi = C.draw_xi(rng, n_val)
                (tp0, _) = C.run(C.gen(model, mu0, xi))
                mu = mu0.copy()
                mu[C.INT] = mu0[C.INT] + best_u
                (tp1, _) = C.run(C.gen(model, mu, xi))
                d = (tp1 >= c).astype(float) - (tp0 >= c)
                rows.append(dict(model=model, base=kind, alpha=a, rej_bliss=(tp0 >= c).mean(), rej_worst=(tp1 >= c).mean(), diff=d.mean(), se=d.std() / np.sqrt(n_val), u=np.round(best_u, 3).tolist(), max_mu=float(mu[C.INT].max())))
                say(rows[-1])
    return rows

def raw_stats(Y):
    a = Y[:, C.EA]
    b = Y[:, C.EB]
    eta = (a[:, :, None] * b[:, None, :]).reshape(len(Y), 16) - Y[:, C.INT]
    return (np.maximum(eta, 0).__pow__(2).sum(1), eta.mean(1))

def p5(selection):
    rows = []
    (n_ref, n_ev) = (100, 100) if C.paths.SMOKE else (400000, 200000)
    hd = C.simulate.interaction_pattern(5, 5, 'highdose').reshape(5, 5)[1:, 1:].ravel()
    ld = C.simulate.interaction_pattern(5, 5, 'lowdose').reshape(5, 5)[1:, 1:].ravel()
    for model in [selection[0]]:
        for kind in [selection[1]]:
            (ya, yb) = C.base_curves(kind)
            mu0 = C.bliss_surface(ya, yb)
            B = mu0[C.INT]
            rng = np.random.default_rng(C.seed('p5', model, kind, 'ref'))
            Yr = C.gen(model, mu0, C.draw_xi(rng, n_ref))
            (tpr, tmr) = C.run(Yr)
            (_, rmr) = raw_stats(Yr)
            rng = np.random.default_rng(C.seed('p5', model, kind, 'ev'))
            xi = C.draw_xi(rng, n_ev)
            surfaces = {}
            for amp in [0.3, 0.6]:
                s_ = B - B * np.exp(-amp * hd)
                t_ = B * np.exp(amp * ld) - B
                surfaces[f'hi_syn_lo_ant_{amp}'] = s_ - s_.sum() / t_.sum() * t_
                surfaces[f'lo_syn_hi_ant_{amp}'] = -(s_ - s_.sum() / t_.sum() * t_)
            for u in [0.05, 0.1, 0.2]:
                e = np.full(16, u / 15.0)
                e[0] = -u
                surfaces[f'ant_cell11_u{u}_syn_rest'] = e
                e = np.full(16, u / 15.0)
                e[15] = -u
                surfaces[f'ant_cell44_u{u}_syn_rest'] = e
                e = np.zeros(16)
                e[0] = -u
                e[15] = u
                surfaces[f'ant_cell11_syn_cell44_u{u}'] = e
            for (name, e) in surfaces.items():
                e = e - e.mean()
                mu = C.with_eta(mu0, e)
                Y1 = C.gen(model, mu, xi)
                (tp1, tm1) = C.run(Y1)
                (_, rm1) = raw_stats(Y1)
                rec = dict(model=model, base=kind, surface=name, mean_eta=float(e.mean()), max_eta=float(e.max()), min_eta=float(e.min()))
                for a in [0.05, 0.01]:
                    rec[f'iso_mean_rej_{a}'] = (tm1 >= np.quantile(tmr, 1 - a)).mean()
                    rec[f'raw_mean_rej_{a}'] = (rm1 >= np.quantile(rmr, 1 - a)).mean()
                    rec[f'iso_Tplus_rej_{a}'] = (tp1 >= np.quantile(tpr, 1 - a)).mean()
                rec['se_05'] = np.sqrt(0.05 * 0.95 / n_ev)
                rows.append(rec)
    return rows

US = C.upper_sets()
NU = len(US)
MASK = np.zeros((NU, 16))
for (_k, _U) in enumerate(US):
    MASK[_k, list(_U)] = 1.0

def _pairs_within(U):
    cells = set(C.INT[list(U)].tolist())
    (pa, pb) = ([], [])
    for (p, q) in zip(C.PA, C.PB):
        if p in cells and q in cells:
            pa.append(p)
            pb.append(q)
    return (np.array(pa, dtype=np.int64), np.array(pb, dtype=np.int64))
PAIRS_U = [_pairs_within(U) for U in US]

def local_stats(Y, method):
    n = len(Y)
    a = np.array([C.pava_dec(y[C.EA], np.ones(4)) for y in Y]) if method != 'naive' else None
    if method == 'naive':
        eta = C.batch_eta(Y, C.W1, C.WS1, C.PA, C.PB, 1e-12, 5000)
        S = eta @ MASK.T
        P = np.maximum(eta, 0) ** 2 @ MASK.T
        return (S, P)
    b = np.array([C.pava_dec(y[C.EB], np.ones(4)) for y in Y])
    pred = (a[:, :, None] * b[:, None, :]).reshape(n, 16)
    if method == 'raw':
        eta = pred - Y[:, C.INT]
        return (eta @ MASK.T, np.maximum(eta, 0) ** 2 @ MASK.T)
    if method == 'restricted':
        S = np.empty((n, NU))
        P = np.empty((n, NU))
        for (k, U) in enumerate(US):
            (pa, pb) = PAIRS_U[k]
            if len(pa) == 0:
                mu = Y
            else:
                mu = C.restricted_batch(Y, pa, pb)
            e = pred[:, list(U)] - mu[:, C.INT[list(U)]]
            S[:, k] = e.sum(1)
            P[:, k] = (np.maximum(e, 0) ** 2).sum(1)
        return (S, P)
    raise ValueError(method)

def pvals_from_ref(T, Tref):
    n_ref = Tref.shape[0]
    out = np.empty_like(T)
    for k in range(T.shape[1]):
        srt = np.sort(Tref[:, k])
        out[:, k] = (1 + n_ref - np.searchsorted(srt, T[:, k] - 1e-12, side='left')) / (n_ref + 1)
    return out

def p6(selection):
    rows = []
    (n_ref, n_ev) = (100, 100) if C.paths.SMOKE else (40000, 20000)
    alpha = 0.05
    (ya, yb) = C.base_curves('moderate')
    mu0 = C.bliss_surface(ya, yb)
    B = mu0[C.INT].reshape(4, 4)
    scen = {}
    e = np.zeros((4, 4))
    e[1, 1] = B[1, 1] - 0.15
    scen['S1_cell22_to_0.15'] = e.ravel()
    e = np.zeros((4, 4))
    e[1, 1] = B[1, 1] - max(B[1, 2], B[2, 1])
    scen['S2_cell22_flat'] = e.ravel()
    e = np.zeros((4, 4))
    e[:2, :2] = np.maximum(B[:2, :2] - B[2, 2], 0)
    scen['S3_lowblock_to_B33'] = e.ravel()
    e = np.zeros((4, 4))
    e[1, :] = B[1, :] - B[2, :]
    scen['S4_row2_flat'] = e.ravel()
    e = np.zeros((4, 4))
    e[1, 1] = B[1, 1] - 0.15
    e[2:, 2:] = -0.03
    scen['S5_S1_plus_hd_antagonism'] = e.ravel()
    scen['S0_bliss'] = np.zeros(16)
    e = np.zeros((4, 4))
    e[3, 3] = 0.08
    scen['S7_topcorner_syn'] = e.ravel()
    for model in [selection[0]]:
        rng = np.random.default_rng(C.seed('p6', model, 'ref'))
        Yr = C.gen(model, mu0, C.draw_xi(rng, n_ref))
        rng = np.random.default_rng(C.seed('p6', model, 'ref2'))
        Yr2 = C.gen(model, mu0, C.draw_xi(rng, n_ref))
        (tpg_ref, _) = C.run(Yr)
        cg = np.quantile(tpg_ref, 1 - alpha)
        refs = {}
        for meth in ['naive', 'raw', 'restricted']:
            (S, P) = local_stats(Yr, meth)
            (S_ref2, P2) = local_stats(Yr2, meth)
            refs[meth] = dict(S=S, P=P)
            for (st, (T1, T2)) in {'sum': (S, S_ref2), 'pos': (P, P2)}.items():
                pmin = pvals_from_ref(T2, T1).min(1)
                refs[meth][st + '_cmin'] = np.quantile(pmin, alpha)
        for (sname, e) in scen.items():
            rng = np.random.default_rng(C.seed('p6', model, sname))
            mu = C.with_eta(mu0, e)
            Y = C.gen(model, mu, C.draw_xi(rng, n_ev))
            (tpg, _) = C.run(Y)
            null_U = np.array([np.all(e[list(U)] <= 1e-15) for U in US])
            for meth in ['naive', 'raw', 'restricted']:
                (S, P) = local_stats(Y, meth)
                for (st, T) in {'sum': S, 'pos': P}.items():
                    pv = pvals_from_ref(T, refs[meth]['S' if st == 'sum' else 'P'])
                    rej = pv <= refs[meth][st + '_cmin']
                    fwer = rej[:, null_U].any(1).mean() if null_U.any() else np.nan
                    pw = rej[:, ~null_U].any(1).mean() if (~null_U).any() else np.nan
                    rec = dict(model=model, scenario=sname, method=meth, stat=st, procedure='maxT_minp', n_null_U=int(null_U.sum()), FWER=fwer, any_true_rej=pw, global_Tplus_rej=(tpg >= cg).mean())
                    rows.append(rec)
                    if st == 'pos':
                        rv = pv <= alpha
                        rejc = np.ones_like(rv)
                        for (k, U) in enumerate(US):
                            sup = [kk for (kk, V) in enumerate(US) if set(U) <= set(V)]
                            rejc[:, k] = rv[:, sup].all(1)
                        fw = rejc[:, null_U].any(1).mean() if null_U.any() else np.nan
                        pw = rejc[:, ~null_U].any(1).mean() if (~null_U).any() else np.nan
                        rows.append(dict(model=model, scenario=sname, method=meth, stat=st, procedure='closed', n_null_U=int(null_U.sum()), FWER=fw, any_true_rej=pw, global_Tplus_rej=(tpg >= cg).mean()))
            say(model, sname, 'done')
    return rows


def main():
    import argparse
    from multiprocessing import Pool
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['order','coupling','patterns','hill','mean','localization']);a=ap.parse_args()
    if a.part == 'order':p1();return
    if a.part == 'coupling':p3();return
    if a.part == 'patterns':
        func=p4;name='heteroscedastic_hetero_patterns.csv'
        jobs=[(m,b) for m in ['hetero_oneil','hetero_oneil_floor','hetero_oneil_t3_floor','add_gauss'] for b in ['weak','moderate','strong','potent']]
    elif a.part == 'hill':
        func=p4_hill;name='heteroscedastic_hetero_hillclimb.csv'
        jobs=[(m,b,a) for m in ['hetero_oneil','hetero_oneil_floor','hetero_oneil_t3_floor'] for b in ['weak','strong','potent'] for a in [.01,.001]]
    elif a.part == 'mean':
        func=p5;name='mean_null_tmean_mixed.csv';jobs=[(m,b) for m in ['add_gauss','hetero_oneil_floor'] for b in ['weak','moderate','strong']]
    else:
        func=p6;name='localization_localization.csv';jobs=[(m,) for m in ['add_gauss','hetero_oneil_floor']]
    with Pool(min(C.paths.WORKERS,len(jobs))) as pool:parts=pool.map(func,jobs)
    pd.DataFrame([row for part in parts for row in part]).to_csv(C.paths.SIMS/name,index=False)

if __name__ == '__main__':main()
