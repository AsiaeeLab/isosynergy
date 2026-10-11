"""tabS5_bigl calculations."""
import mc
import json
import os
import numpy as np
import pandas as pd
import style as ST
import sim_bigl as SB
PATTERNS = [('syn_broad_0.10', 'broad'), ('syn_highdose_0.40', 'high dose'), ('syn_corner_0.60', 'single top-corner cell'), ('sat_0.25', 'saturating')]

def setting_rates(df, r, cond, col, alpha=0.05):
    out = {}
    for s in SB.SURFS:
        for n in SB.NOISES:
            st = SB.setting_name(r, s, n)
            x = df[(df.setting == st) & (df.cond == cond)][col].values
            out[st] = SB.rate(x, alpha)
    return out

def bigl_setting_rates(bd, cfg, r, cond, col):
    out = {}
    for s in SB.SURFS:
        for n in SB.NOISES:
            st = SB.setting_name(r, s, n)
            x = bd[(bd.config == cfg) & (bd.setting == st) & (bd.cond == cond)][col].values.astype(bool)
            if len(x):
                out[st] = SB.rate(x)
    return out

def rng_of(d):
    v = [t['p'] for t in d.values()]
    return (float(min(v)), float(max(v)))

def mean_of(d):
    return float(np.mean([t['p'] for t in d.values()]))

def main():
    o = SB.load_ours()
    bd = SB.bigl_decisions()
    J = {}
    for (col, key) in (('p_tstd', 'tstd'), ('p_scan', 'scan'), ('p_mean', 'mean'), ('p_tstd_est', 'tstd_est'), ('p_scan_est', 'scan_est')):
        for a in (0.05, 0.01):
            d = setting_rates(o, 4, 'null', col, a)
            J[f'size_r4_{key}_{a}'] = dict(range=rng_of(d), by_setting={k: v['p'] for (k, v) in d.items()}, ci={k: (v['lo'], v['hi']) for (k, v) in d.items()})
    for (col, key) in (('p_tstd', 'tstd'), ('p_scan', 'scan'), ('p_mean', 'mean')):
        for a in (0.05, 0.01):
            d = setting_rates(o, 1, 'null', col, a)
            J[f'size_r1_{key}_{a}'] = dict(range=rng_of(d), by_setting={k: v['p'] for (k, v) in d.items()})
    for cond in SB.ALTS:
        row = {}
        for (col, key) in (('p_tstd', 'tstd'), ('p_scan', 'scan'), ('p_mean', 'mean'), ('p_tstd_est', 'tstd_est')):
            d = setting_rates(o, 4, cond, col)
            row[key] = dict(mean=mean_of(d), by_setting={k: v['p'] for (k, v) in d.items()})
        for (cfg, col, key) in (('vb_eq', 'maxR_syn_q2a_005', 'vb_maxR_010'), ('vb_eq', 'maxR_syn_q_005', 'vb_maxR_005'), ('vb_eq', 'meanR_005', 'vb_meanR')):
            d = bigl_setting_rates(bd, cfg, 4, cond, col)
            row[key] = dict(mean=mean_of(d), by_setting={k: v['p'] for (k, v) in d.items()})
        J[f'power_{cond}'] = row
    diffs = []
    for cond in SB.ALTS:
        a = setting_rates(o, 4, cond, 'p_tstd')
        b = setting_rates(o, 4, cond, 'p_tstd_est')
        diffs += [abs(a[k]['p'] - b[k]['p']) for k in a]
    J['tstd_est_max_power_diff'] = float(max(diffs))
    m = SB.real_production()
    real = bd[bd.cond == 'real']
    J['real_prod_map'] = dict(n=int(len(m)), batch_counts=m.batch.value_counts().to_dict(), stage2=int((m.stage == 2).sum()))
    for (col, key) in (('p_tstd', 'tstd'), ('p_scan', 'scan'), ('p_companion', 'block_t')):
        J[f'real_{key}'] = SB.rate(m[col].values)
    for cfg in ('ship_eq', 'ship_model', 'vb_eq'):
        x = real[real.config == cfg]
        x = x.sort_values('idx')
        ok = x.ok.values
        aligned = m.iloc[x.idx.to_numpy(int)]
        J[f'real_bigl_{cfg}'] = dict(n_result=int(ok.sum()), maxR_syn=SB.rate(x.maxR_syn_q_005.values[ok]), meanR=SB.rate(x.meanR_005.values[ok]), meanR_sign=SB.rate(x.meanR_sign_005.values[ok]))
        J[f'agree_tstd_{cfg}_maxR'] = SB.kappa((aligned.p_tstd.values <= 0.05)[ok], x.maxR_syn_q_005.values[ok])
        J[f'agree_scan_{cfg}_maxR'] = SB.kappa((aligned.p_scan.values <= 0.05)[ok], x.maxR_syn_q_005.values[ok])
        J[f'agree_blockt_{cfg}_meanRsign'] = SB.kappa((aligned.p_companion.values <= 0.05)[ok], x.meanR_sign_005.values[ok])
        J[f'real_tstd_among_{cfg}'] = SB.rate(aligned.p_tstd.values[ok])
    rt = {}
    for r in (4, 1):
        t = o[o.r == r].t_fit.values
        rt[f'r{r}'] = dict(n=int(t.size), median=float(np.median(t)), mean=float(t.mean()), p95=float(np.quantile(t, 0.95)))
    J['runtime_ours'] = rt
    bt = pd.read_csv(os.path.join(mc.INPUTS, 'bigl_runtime.csv'))
    J['runtime_bigl'] = bt[bt.method.str.startswith('BIGL')].to_dict('records')
    with open(os.path.join(mc.RESDIR, 'bigl_summary.json'), 'w') as f:
        json.dump(J, f, indent=1, default=float)
    f3 = ST.fmt_rate
    (lo, hi) = J['size_r4_tstd_0.05']['range']
    (slo, shi) = J['size_r4_scan_0.05']['range']
    (t1, t4) = sorted((rt[bytes([114, 49]).decode('ascii')]['median'], rt[bytes([114, 52]).decode('ascii')]['median']))
    def rate_range(cfg, col):
        low, high = rng_of(bigl_setting_rates(bd, cfg, 4, 'null', col))
        return f'{low:.3f} to {high:.3f}'
    antagonism = bigl_setting_rates(bd, 'ship_eq', 4, 'ant_broad_0.10', 'maxR_syn_q_005')['r4_moderate_gauss']['p']
    J['bigl_null_distributed'] = bigl_setting_rates(bd, 'ship_eq', 4, 'null', 'maxR_syn_q_005')
    partial_gaussian = {k: v for k, v in J['bigl_null_distributed'].items()
                        if k in ('r4_weak_gauss', 'r4_moderate_gauss')}
    partial_low, partial_high = rng_of(partial_gaussian)
    J['bigl_null_product_maximum'] = bigl_setting_rates(bd, 'vb_eq', 4, 'null', 'maxR_any_x_005')
    J['bigl_null_product_mean'] = bigl_setting_rates(bd, 'vb_eq', 4, 'null', 'meanR_005')
    J['bigl_antagonism'] = antagonism
    rows = [('BIGL as distributed, maxR synergy label, exact viability Bliss (weak/moderate agents, Gaussian)', f'{partial_low:.3f} to {partial_high:.3f}'),
            ('BIGL as distributed, synergy label under 10\\% antagonism (moderate agents, Gaussian)', f'{antagonism:.3f}'),
            ('BIGL patched, two-sided maxR (rank rule) at exact Bliss', rate_range('vb_eq','maxR_any_x_005')),
            ('BIGL patched, meanR at exact Bliss', rate_range('vb_eq','meanR_005')),
            ('$T_{\\mathrm{SIR}}$ at exact Bliss (true noise law)', f'{f3(lo)} to {f3(hi)}'),
            ('Region scan at exact Bliss (true noise law)', f'{f3(slo)} to {f3(shi)}')]

    first = True
    for (cond, name) in PATTERNS:
        P = J[f'power_{cond}']
        v = f"{f3(P['tstd']['mean'])} / {f3(P['scan']['mean'])} / {f3(P['vb_maxR_010']['mean'])}"
        if first:
            rows.append((f'Power, {name}: $T_{{\\mathrm{{SIR}}}}$ / region scan / patched maxR at 0.10', v))
            first = False
        else:
            rows.append((f'Power, {name}', v))
    import bigl_summary
    raw = bigl_summary.load_bigl()
    real_raw = raw[raw.cond == 'real']
    first_real = real_raw[real_raw.config == 'ship_eq']
    n_fail = int((first_real.fit_err != 0).sum())
    distributed_real = real_raw[real_raw.config.isin(('ship_eq', 'ship_model'))]
    failures = [int(((g.fit_err == 0) & (g.surf_err != 0)).sum()) for _, g in distributed_real.groupby('config')]
    J['bigl_real_failures'] = dict(marginal=n_fail, total=len(first_real), surface_min=min(failures), surface_max=max(failures))
    observed = np.fromfile(mc.paths.INPUTS/'bigl_comparison_data/real_oneil300.f64', dtype='<f8').reshape(-1,4,5,5)
    J['bigl_real_noise'] = dict(marginal_sd_median=float(first_real.sigma0.median()),
        interior_sd_median=float(np.median(observed[:,:,1:,1:].std(axis=1,ddof=1).mean(axis=(1,2)))))
    meta = pd.read_csv(mc.paths.INPUTS/'bigl_real_meta.csv').set_index('k')
    doses = meta.loc[first_real.idx.to_numpy(int),'doseA'].str.split(';')
    lower = np.log(doses.map(lambda x: float(x[1])).to_numpy())
    upper = np.log(doses.map(lambda x: float(x[-1])).to_numpy())
    J['bigl_real_failures']['converged'] = int((first_real.fit_err == 0).sum())
    J['bigl_real_failures']['implausible'] = int(bigl_summary.degenerate(first_real,lower,upper).sum())
    single = raw[(raw.config == 'ship_eq') & raw.setting.str.startswith('r1_') & (raw.cond == 'null')]
    J['bigl_single_plate_failures'] = {setting:dict(n=len(g), marginal=int((g.fit_err != 0).sum()),
        surface=int(((g.fit_err == 0) & (g.surf_err != 0)).sum()),
        no_result_share=float((~g.ok).mean())) for setting,g in single.groupby('setting')}
    bt4 = bt[(bt.r == 4) & bt.method.str.contains('ship_eq|vb_eq|vb_model',regex=True)]
    low_time, high_time = bt4['median'].min(), bt4['median'].max()
    rows += [("Real O'Neil groups: marginal fit failures", f'{n_fail} of {len(first_real)}'),
             ("Real O'Neil groups: further Bliss-reference failures", f'{min(failures)} to {max(failures)}'),
             ('Runtime per matrix, BIGL ($B=100$) vs $T_{\\mathrm{SIR}}$ with region scan ($B=399+200$, one core)', f'median {low_time:.1f} to {high_time:.1f} s vs {t1:.3f} to {t4:.3f} s')]
    with open(os.path.join(mc.RESDIR, 'bigl_summary.json'), 'w') as f:
        json.dump(J, f, indent=1, default=float)

    body = ['\\begin{tabular}{@{}p{0.62\\linewidth}r@{}}', '\\toprule', 'Quantity & Value\\\\', '\\midrule']
    body += [f'{a} & {b}\\\\' for (a, b) in rows]
    body += ['\\bottomrule', '\\end{tabular}']
    ST.write_table('tabS5_bigl', '\n'.join(body))
    print('\n'.join(body))
    print(json.dumps({k: J[k] for k in ('tstd_est_max_power_diff', 'runtime_ours', 'real_tstd', 'real_scan', 'real_block_t', 'agree_tstd_ship_model_maxR')}, indent=1, default=float))
if __name__ == '__main__':
    main()
