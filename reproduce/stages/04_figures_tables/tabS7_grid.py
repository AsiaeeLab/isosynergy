"""tabS7_grid calculations."""
import mc
import style as ST
import json
import os
import numpy as np
import pandas as pd
GRIDS = (4, 5, 6, 8, 10)
PNAME = {'broad': 'Broad', 'highdose': 'High-dose', 'corner': 'Corner', 'sat': 'Saturating'}
TNAME = {'tstd': '$T_{\\mathrm{SIR}}$', 'scan': 'Region scan', 'mean': 'Mean test'}
ROWS = [('S', 'broad', 'tstd'), ('S', 'broad', 'scan'), ('S', 'broad', 'mean'), ('S', 'highdose', 'tstd'), ('S', 'corner', 'tstd'), ('S', 'corner', 'scan'), ('S', 'corner', 'mean'), ('S', 'sat', 'tstd'), ('P', 'broad', 'tstd'), ('P', 'broad', 'mean'), ('P', 'highdose', 'tstd'), ('P', 'corner', 'tstd'), ('P', 'corner', 'scan'), ('P', 'sat', 'tstd')]

def cell(e80, d, pat, t, I, a=0.05):
    x = e80[(e80.design == d) & (e80.noise == 'oneil') & (e80.pattern == pat) & (e80.test == t) & (e80.I == I) & np.isclose(e80.alpha, a)]
    assert len(x) == 1, (d, pat, t, I, a, len(x))
    return x.iloc[0]

def fmt(r, which):
    if r['reached']:
        return f"{r[which + '_eta80']:.3f}"
    return f"$>{r[which + '_eta_at_max']:.2f}$"

def main():
    e80 = pd.read_csv(os.path.join(ST.RESDIR, 'grid_effect80.csv'))
    sz = pd.read_csv(os.path.join(ST.RESDIR, 'grid_size.csv'))
    lines = ['\\begin{tabular}{@{}lllrrrrr@{}}', '\\toprule', 'Design & Pattern & Test & $4\\times4$ & $5\\times5$ & $6\\times6$ & $8\\times8$ & $10\\times10$\\\\', '\\midrule']
    prev = None
    md = []
    for (d, pat, t) in ROWS:
        if prev is not None and d != prev:
            lines.append('\\addlinespace')
        prev = d
        which = 'max' if pat == 'corner' else 'mean'
        vals = [fmt(cell(e80, d, pat, t, I), which) for I in GRIDS]
        lines.append(f'{d} & {PNAME[pat]} & {TNAME[t]} & ' + ' & '.join(vals) + '\\\\')
        md.append(f'{d} {pat} {t}: ' + ', '.join(vals))
    lines += ['\\bottomrule', '\\end{tabular}']
    ST.write_table('tabS7_grid', '\n'.join(lines))
    print('\n'.join(md))
    num = {}
    for d in ('S', 'P'):
        r = []
        for I in GRIDS:
            a1 = cell(e80, d, 'broad', 'tstd', I, 0.05)
            a2 = cell(e80, d, 'broad', 'tstd', I, 0.005)
            r.append(dict(I=I, e05=a1['mean_eta80'], e005=a2['mean_eta80'], ratio=a2['mean_eta80'] / a1['mean_eta80']))
        num[f'ratio_005_vs_05_tstd_broad_{d}'] = r
    rr = []
    for d in ('S', 'P'):
        for pat in ('broad', 'highdose', 'corner', 'sat'):
            which = 'max' if pat == 'corner' else 'mean'
            for I in GRIDS:
                a1 = cell(e80, d, pat, 'tstd', I, 0.05)
                a2 = cell(e80, d, pat, 'tstd', I, 0.005)
                if a1['reached'] and a2['reached']:
                    rr.append(dict(design=d, pattern=pat, I=I, ratio=a2[f'{which}_eta80'] / a1[f'{which}_eta80']))
    num['ratio_005_vs_05_tstd_all'] = rr
    out = {}
    for d in ('S', 'P'):
        for I in GRIDS:
            g = sz[(sz.design == d) & (sz.I == I)]
            rec = dict(n_cells=len(g), n=g.n.tolist())
            for t in ('tstd', 'scan', 'mean'):
                for a in (0.05, 0.01, 0.005):
                    k = g[f'k_{t}@{a}'].to_numpy()
                    n = g['n'].to_numpy()
                    rec[f'{t}@{a}'] = dict(min=float((k / n).min()), max=float((k / n).max()), pooled=float(k.sum() / n.sum()))
            out[f'{d}_{I}'] = rec
    num['null_size_by_grid'] = out
    gS = sz[sz.design == 'S'].copy()
    scan = {}
    for a in (0.05, 0.01, 0.005):
        j = gS[f'scan@{a}'].idxmax()
        scan[str(a)] = dict(max=float(gS.loc[j, f'scan@{a}']), lo=float(gS.loc[j, f'scan@{a}_lo']), hi=float(gS.loc[j, f'scan@{a}_hi']), where=f"{gS.loc[j, 'I']}x{gS.loc[j, 'I']} {gS.loc[j, 'surface']} {gS.loc[j, 'noise']}", n=int(gS.loc[j, 'n']), n_lower_above=int((gS[f'scan@{a}_lo'] > a).sum()), n_upper_below=int((gS[f'scan@{a}_hi'] < a).sum()), n_cells=len(gS))
        w = gS[gS.surface == 'weak']
        scan[str(a)]['weak_min'] = float(w[f'scan@{a}'].min())
        scan[str(a)]['weak_max'] = float(w[f'scan@{a}'].max())
    num['scan_null_design_S'] = scan
    gP = sz[sz.design == 'P']
    num['scan_null_design_P'] = {str(a): dict(min=float(gP[f'scan@{a}'].min()), max=float(gP[f'scan@{a}'].max()), pooled=float(gP[f'k_scan@{a}'].sum() / gP.n.sum())) for a in (0.05, 0.01)}
    with open(os.path.join(ST.RESDIR, 'grid_numbers.json'), 'w') as f:
        json.dump(num, f, indent=1, default=float)
    print(json.dumps({k: v for (k, v) in num.items() if k != 'null_size_by_grid'}, indent=1, default=float))
    for (k, v) in out.items():
        print(k, {t: f"{v[t + '@0.05']['min']:.4f}-{v[t + '@0.05']['max']:.4f} ({v[t + '@0.05']['pooled']:.4f})" for t in ('tstd', 'scan', 'mean')})
if __name__ == '__main__':
    main()
