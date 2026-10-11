"""sec6_numbers calculations."""
import glob
import json
import os
import mc
import numpy as np
import pandas as pd
R = mc.RESDIR
OUT = {}

def rd(name):
    return pd.read_parquet(os.path.join(R, name))

def rate(p, a=0.05):
    p = np.asarray(p)
    (k, n) = (int((p <= a).sum()), p.size)
    (lo, hi) = mc.wilson(k, n)
    return dict(rate=k / n, lo=float(lo), hi=float(hi), k=k, n=n)

def paired(p1, p0, a=0.05):
    d = (np.asarray(p1) <= a).astype(float) - (np.asarray(p0) <= a).astype(float)
    (m, se) = (d.mean(), d.std(ddof=1) / np.sqrt(d.size))
    return dict(diff=m, lo=m - 1.959964 * se, hi=m + 1.959964 * se, n=d.size)

def fmt(r):
    return f"{r['rate']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] (n={r['n']})"

def missing():
    d = rd('fig6_missing_size.parquet')
    o = {}
    for t in ('full', 'int1', 'int2', 'int3', 'nfill', 'rowdrop'):
        o[t] = dict(r05=rate(d[f'p_{t}']), r01=rate(d[f'p_{t}'], 0.01))
        if t != 'full':
            o[t]['paired05'] = paired(d[f'p_{t}'], d.p_full)
    d['pos'] = np.where(d.sa_dose == 1, 'lowest', np.where(d.sa_dose == d.I - 1, 'top', 'middle'))
    g = d.groupby(['I', 'surface', 'pos']).p_nfill.apply(lambda v: (v <= 0.05).mean()).unstack('pos')
    o['nfill_by_position'] = {f'{i}x{i} {sf}': g.loc[i, sf].round(4).to_dict() for (i, sf) in g.index}
    top = d[d.pos == 'top'].groupby(['I', 'surface']).p_nfill.agg(lambda v: (v <= 0.05).mean())
    k = top.idxmax()
    sub = d[(d.pos == 'top') & (d.I == k[0]) & (d.surface == k[1])]
    o['nfill_worst_top'] = dict(cell=list(map(str, k)), **rate(sub.p_nfill))
    o['worst_rows'] = {}
    for t in ('full', 'int1', 'int2', 'int3', 'rowdrop'):
        g = d.groupby(['I', 'law', 'surface'])[f'p_{t}'].apply(lambda v: (v <= 0.05).mean())
        kk = g.idxmax()
        o['worst_rows'][t] = dict(cell=list(map(str, kk)), rate=float(g.max()))
    p = rd('fig6_missing_sizeP.parquet')
    o['designP'] = {t: dict(r05=rate(p[f'p_{t}']), r01=rate(p[f'p_{t}'], 0.01)) for t in ('full', 'int1', 'int2', 'int3')}
    w = rd('fig6_missing_power.parquet')
    pw = w.groupby(['I', 'law', 'surface', 'cond'])[[f'p_{t}' for t in ('full', 'int1', 'int2', 'int3', 'rowdrop')]].agg(lambda v: (v <= 0.05).mean())
    o['power_change'] = {}
    for t in ('int1', 'int2', 'int3', 'rowdrop'):
        dl = pw[f'p_{t}'] - pw.p_full
        o['power_change'][t] = dict(mean=float(dl.mean()), min=float(dl.min()), worst='/'.join(map(str, dl.idxmin())))
    c = w[w.cond.str.startswith('syn_corner')]
    o['corner_missing_share'] = {f'int{q}': float((c[f'nsyn_miss{q}'] > 0).mean()) for q in (1, 2, 3)}
    obs = c[c.nsyn_miss3 == 0]
    o['corner_power_given_observed'] = dict(full=float((obs.p_full <= 0.05).mean()), int3=float((obs.p_int3 <= 0.05).mean()))
    big = []
    for t in ('int1', 'int2', 'int3'):
        dl = pw[f'p_{t}'] - pw.p_full
        for (kk, v) in dl[dl < -0.05].items():
            big.append(('/'.join(map(str, kk)), t, round(float(v), 3)))
    o['losses_over_0.05'] = big
    for f in sorted(glob.glob(os.path.join(R, 'fig6_missing_confirm_*.parquet'))):
        if '_n' in os.path.basename(f).split('confirm_')[1]:
            continue
        cf = pd.read_parquet(f)
        o[os.path.basename(f)] = {t: dict(r05=rate(cf[f'p_{t}']), r01=rate(cf[f'p_{t}'], 0.01), paired05=paired(cf[f'p_{t}'], cf.p_full) if t != 'full' else None) for t in ('full', 'int1', 'int2', 'int3', 'rowdrop')}
    OUT['missing'] = o

def screen():
    s = rd('fig6_screen_size.parquet')
    o = {}
    for ((lev, arm), z) in s.groupby(['level', 'arm']):
        o[f'{arm}_{lev}'] = dict(r05=rate(z.p), r01=rate(z.p, 0.01))
    piv = s.pivot_table(index=['I', 'law', 'surface', 'start', 't', 'level'], columns='arm', values='p')
    clean = piv.xs(0.0, level='level')
    o['clean_paired_screen_minus_noscreen'] = paired(clean.screen, clean.noscreen)
    for lev in (0.004, 0.01, 0.02):
        o[f'worst_row_screen_{lev}'] = (lambda g: dict(cell=list(map(str, g.idxmax())), rate=float(g.max())))(s[(s.arm == 'screen') & (s.level == lev)].groupby(['I', 'law', 'surface']).p.apply(lambda v: (v <= 0.05).mean()))
        o[f'worst_row_noscreen_{lev}'] = (lambda g: dict(cell=list(map(str, g.idxmax())), rate=float(g.max())))(s[(s.arm == 'noscreen') & (s.level == lev)].groupby(['I', 'law', 'surface']).p.apply(lambda v: (v <= 0.05).mean()))
    z = s[(s.arm == 'noscreen') & (s.level == 0.02)]
    o['noscreen_2pct_sa_error'] = rate(z[z.n_cont_edge > 0].p)
    o['noscreen_2pct_comb_only'] = rate(z[(z.n_cont_edge == 0) & (z.n_cont > 0)].p)
    o['noscreen_2pct_no_error'] = rate(z[z.n_cont == 0].p)
    o['share_datasets_with_sa_error_2pct'] = float((z.n_cont_edge > 0).mean())
    w = rd('fig6_screen_power.parquet')
    pw = w[w.level == 0.0].groupby(['I', 'law', 'surface', 'cond', 'arm']).p.apply(lambda v: (v <= 0.05).mean()).unstack('arm')
    dl = pw.screen - pw.noscreen
    o['clean_power_change'] = dict(mean=float(dl.mean()), min=float(dl.min()), max=float(dl.max()), rows=len(dl))
    ck = rd('fig6_screen_check.parquet')
    o['check_by_law'] = ck.groupby('law')[['share_flag_status_differs', 'flag_rate_alt', 'flag_rate_null', 'bias_screened', 'bias_unscreened']].agg(['mean', 'max']).round(6).to_dict()
    for f in sorted(glob.glob(os.path.join(R, 'fig6_screen_confirm_*.parquet'))):
        cf = pd.read_parquet(f)
        oo = {f'{arm}_{lev}': dict(r05=rate(z.p), r01=rate(z.p, 0.01)) for ((lev, arm), z) in cf.groupby(['level', 'arm'])}
        pv = cf.pivot_table(index=['start', 't'], columns=['level', 'arm'], values='p')
        oo['paired_screen2pct_vs_clean_noscreen'] = paired(pv[0.02, 'screen'], pv[0.0, 'noscreen'])
        o[os.path.basename(f)] = oo
    if os.path.exists(os.path.join(R, 'fig6_screen_bank.parquet')):
        b = rd('fig6_screen_bank.parquet')
        o['bank'] = {f'{arm}_{lev}': dict(r05=rate(z.p), r01=rate(z.p, 0.01), r005=rate(z.p, 0.005)) for ((lev, arm), z) in b.groupby(['level', 'arm'])}
    OUT['screen'] = o

def fragility():
    d = rd('fig6_fragility.parquet')
    c = d[d.p <= 0.05]
    o = {}
    sel = c[(c.I == 5) & (c.law == 'C')]
    rows = [('null', 0.0, 'moderate'), ('null', 0.004, 'moderate'), ('null', 0.01, 'moderate'), ('null', 0.02, 'moderate'), ('syn_corner_2.00', 0.0, 'weak'), ('dip_1.00', 0.0, 'moderate'), ('syn_hdblock_0.80', 0.0, 'moderate'), ('syn_broad_0.20', 0.0, 'moderate'), ('syn_broad_0.30', 0.0, 'moderate'), ('sat_0.60', 0.0, 'strong')]
    o['5x5_C'] = {}
    for (cond, lev, surf) in rows:
        z = sel[(sel.cond == cond) & (sel.level == lev) & (sel.surface == surf)]
        allz = d[(d.I == 5) & (d.law == 'C') & (d.cond == cond) & (d.level == lev) & (d.surface == surf)]
        o['5x5_C'][f'{cond}|{lev}|{surf}'] = dict(calls=len(z), call_rate=rate(allz.p)['rate'], cell=rate(1 - z.loco_flag.astype(float))['rate'] if len(z) else None, row=float(z.sa_flag.mean()) if len(z) else None)
        o['5x5_C'][f'{cond}|{lev}|{surf}']['cell'] = float(z.loco_flag.mean()) if len(z) else None
    single = c[c.cond.str.startswith('syn_corner') | c.cond.str.startswith('dip')]
    broad = c[c.cond.str.startswith('syn_broad')]
    o['pooled_single_cell'] = dict(cell=float(single.loco_flag.mean()), row=float(single.sa_flag.mean()), calls=len(single))
    o['pooled_broad'] = dict(cell=float(broad.loco_flag.mean()), row=float(broad.sa_flag.mean()), calls=len(broad))
    o['single_cell_range'] = c[c.cond.str.startswith('syn_corner') | c.cond.str.startswith('dip')].groupby(['I', 'law', 'cond']).loco_flag.mean().agg(['min', 'max']).to_dict()
    o['broad_range'] = broad.groupby(['I', 'law', 'cond']).loco_flag.mean().agg(['min', 'max']).to_dict()
    cn = c[(c.cond == 'null') & (c.level > 0)]
    sa_err = cn[cn.n_cont_edge > 0]
    comb = cn[(cn.n_cont_edge == 0) & (cn.n_cont_int > 0)]
    o['contaminated_false_calls'] = dict(n=len(cn), with_sa_error=len(sa_err), cell_flag_sa=float(sa_err.loco_flag.mean()), row_flag_sa=float(sa_err.sa_flag.mean()), dropped_is_contaminated=float(sa_err.sa_cont.mean()), comb_only=len(comb), cell_flag_comb=float(comb.loco_flag.mean()) if len(comb) else None, loco_cell_is_contaminated=float(comb.loco_cell_cont.mean()) if len(comb) else None)
    cl = c[(c.cond == 'null') & (c.level == 0)]
    o['clean_null_pooled'] = dict(cell=float(cl.loco_flag.mean()), row=float(cl.sa_flag.mean()), calls=len(cl))
    OUT['fragility'] = o

def monotone():
    d = rd('fig6_monotone_size.parquet')
    o = {}
    nm = d[d.nonmono_truth]
    for des in ('S', 'P'):
        g = nm[nm.design == des].groupby(['I', 'law', 'surface', 'cond']).p.apply(lambda v: (v <= 0.05).mean())
        k = g.idxmax()
        z = nm[(nm.design == des) & (nm.I == k[0]) & (nm.law == k[1]) & (nm.surface == k[2]) & (nm.cond == k[3])]
        o[f'max_nonmono_{des}'] = dict(cell=list(map(str, k)), **rate(z.p))
        g1 = nm[nm.design == des].groupby(['I', 'law', 'surface', 'cond']).p.apply(lambda v: (v <= 0.01).mean())
        o[f'max_nonmono_{des}_01'] = dict(cell=list(map(str, g1.idxmax())), rate=float(g1.max()))
        gm = nm[nm.design == des].groupby(['I', 'law', 'surface', 'cond']).p_mean.apply(lambda v: (v <= 0.05).mean())
        o[f'max_nonmono_{des}_mean'] = dict(cell=list(map(str, gm.idxmax())), rate=float(gm.max()))
        base = d[(d.design == des) & (d.surface == 'base') & (d.cond == 'null')]
        o[f'base_{des}'] = rate(base.p)
        bl = base.groupby(['I', 'law']).p.apply(lambda v: (v <= 0.05).mean())
        cnt = []
        for (kk, v) in g.items():
            b = bl.loc[kk[0], kk[1]]
            se = np.sqrt(b * (1 - b) / 2000 + v * (1 - v) / 2000)
            cnt.append((v - b) / se)
        o[f'max_z_above_baseline_{des}'] = float(np.max(cnt))
    S5 = d[d.design == 'S']
    hs = S5[S5.surface.isin(['horm_0.03', 'horm_0.07', 'horm_0.15', 'hormA_0.07', 'flat_0.03', 'flat_0.07']) & (S5.cond == 'null')]
    o['S_pooled_horm_inactive'] = dict(T=rate(hs.p), mean=rate(hs.p_mean))
    o['S_pooled_monotone'] = dict(T=rate(S5[(S5.surface == 'base') & (S5.cond == 'null')].p), mean=rate(S5[(S5.surface == 'base') & (S5.cond == 'null')].p_mean))
    w = rd('fig6_monotone_power.parquet')
    pw = w.groupby(['design', 'law', 'surface', 'cond']).p.apply(lambda v: (v <= 0.05).mean())
    pm = w.groupby(['design', 'law', 'surface', 'cond']).p_mean.apply(lambda v: (v <= 0.05).mean())
    o['power_T'] = {'/'.join(k): round(float(v), 3) for (k, v) in pw.items()}
    o['power_mean'] = {'/'.join(k): round(float(v), 3) for (k, v) in pm.items()}
    for f in sorted(glob.glob(os.path.join(R, 'fig6_monotone_confirm_*.parquet'))):
        cf = pd.read_parquet(f)
        o[os.path.basename(f)] = dict(T05=rate(cf.p), T01=rate(cf.p, 0.01), mean05=rate(cf.p_mean))
    OUT['monotone'] = o

def clean(o):
    if isinstance(o, dict):
        return {'|'.join(map(str, k)) if isinstance(k, tuple) else str(k): clean(v) for (k, v) in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o

def main():
    global OUT
    for f in (missing, screen, fragility, monotone):
        try:
            f()
        except FileNotFoundError as e:
            print(f'skip {f.__name__}: {e}')
    OUT = clean(OUT)
    with open(os.path.join(R, 'fig6_text_numbers.json'), 'w') as fh:
        json.dump(OUT, fh, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(json.dumps(OUT, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))
if __name__ == '__main__':
    main()
