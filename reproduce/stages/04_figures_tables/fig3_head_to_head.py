"""fig3_head_to_head calculations."""
import mc
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as ST
import h2h_lib as L
(COL, LAB) = (ST.COL, ST.LAB)
TESTS = ('tstd', 'scan', 'mean')
PAT = [('syn_broad_0.20', 'Broad'), ('syn_hdblock_0.80', 'High-dose block'), ('syn_corner_1.00', 'Top corner'), ('syn_bump_0.30', 'Middle bump'), ('syn_lowdose_0.10', 'Low dose'), ('sat_0.60', 'Saturating'), ('mixed_0.60', 'Zero net')]
PREVIEW_PARTIAL = os.environ.get('H2H_PARTIAL') == '1'
REAL = {'oneil': [('syn_broad_0.05', 'Broad'), ('syn_highdose_0.10', 'High dose'), ('sat_0.5', 'Saturating')], 'almanac': [('syn_broad_0.10', 'Broad'), ('syn_highdose_0.20', 'High dose'), ('sat_1.0', 'Saturating')]}
WD = 0.27

def panel_size(ax):
    df = L.load('size')
    R = L.rates(df[(df.ref == 'matched') & (df.cond == 'null')], ['design', 'I', 'law', 'surface'])
    groups = [('S', 5, 'o', 'S, 5$\\times$5'), ('S', 4, 's', 'S, 4$\\times$4'), ('P', 5, 'o', 'P, 5$\\times$5'), ('P', 4, 's', 'P, 4$\\times$4')]
    out = {}
    for (k, t) in enumerate(TESTS):
        for (m, (des, I, mk, _)) in enumerate(groups):
            w = L.worst(R, t, (R.design == des) & (R.I == I))
            out[t, des, I] = w
            x = k + (m - 1.5) * 0.18
            ax.errorbar(x, w['rate'], yerr=np.maximum(0, [[w['rate'] - w['lo']], [w['hi'] - w['rate']]]), fmt=mk, color=COL[t], mfc=COL[t] if des == 'S' else 'white', ms=3.8, lw=0.8, capsize=1.2)
    ax.axhline(0.05, color='0.3', ls='--', lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LAB[t] for t in TESTS], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.55, 2.55)
    lo = min((w['lo'] for w in out.values()))
    hi = max((w['hi'] for w in out.values()))
    ax.set_ylim(min(0.03, lo - 0.003), max(0.07, hi + 0.012))
    ax.set_ylabel('Worst null rejection rate')
    h = [Line2D([], [], marker=mk, ls='none', color='0.3', mfc='0.3' if des == 'S' else 'white', ms=3.8, label=lab) for (des, _, mk, lab) in groups]
    ax.legend(handles=h, loc='upper left', ncol=2, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.1, labelspacing=0.25)
    return out

def bars(ax, x, vals, tests):
    for (k, t) in enumerate(tests):
        ax.bar(x + (k - (len(tests) - 1) / 2) * WD, vals[t], WD, color=COL[t], label=LAB[t], lw=0)

def panel_power(ax, design):
    df = L.load('power')
    R = L.rates(df[df.design == design], ['cond']).set_index('cond')
    x = np.arange(len(PAT))
    vals = {t: [R.loc[c, t] for (c, _) in PAT] for t in TESTS}
    bars(ax, x, vals, TESTS)
    ax.text(x[-1] + WD, vals['mean'][-1] + 0.015, '*', ha='center', va='bottom', fontsize=8, color=COL['mean'])
    ax.set_xticks(x)
    ax.set_xticklabels([p for (_, p) in PAT], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(PAT) - 0.4)
    return R

def panel_real_null(ax):
    d = L.null_checks()
    sel = [('oneil', 'p_tstd', 'tstd', LAB['tstd']), ('oneil', 'p_scan', 'scan', 'Scan'), ('oneil', 'p_companion', 'mean', 'Block $t$'), ('almanac', 'p_tstd', 'tstd', LAB['tstd']), ('almanac', 'p_scan', 'scan', 'Scan'), ('almanac', 'p_companion', 'mean', 'V2Sp')]
    xs = [0, 1, 2, 3.5, 4.5, 5.5]
    levels = [(0.05, 'o'), (0.01, 's'), (0.005, '^')]
    out = []
    for (x, (study, col, t, _)) in zip(xs, sel):
        for (j, (a, mk)) in enumerate(levels):
            r = d[(d.study == study) & (d.column == col) & np.isclose(d.level, a)].iloc[0]
            (v, lo, hi) = (r.rate / a, r.lo / a, r.hi / a)
            out.append((study, col, a, r.rate, r.lo, r.hi, int(r.n)))
            ax.errorbar(x + (j - 1) * 0.25, v, yerr=np.maximum(0, [[v - lo], [hi - v]]), fmt=mk, color=COL[t], ms=3.4, lw=0.8, capsize=1.1, mfc=COL[t] if study == 'almanac' else 'white')
    ax.axhline(1.0, color='0.3', ls='--', lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[3] for s in sel], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 4.3)
    ax.set_xlim(-0.6, 6.1)
    ax.set_ylabel('Null rejection rate / $\\alpha$')
    ax.text(1.0, 4.25, "O'Neil, P", ha='center', va='top', fontsize=7)
    ax.text(4.5, 4.25, 'ALMANAC, S', ha='center', va='top', fontsize=7)
    h = [Line2D([], [], marker=mk, ls='none', color='0.3', mfc='0.3', ms=3.4, label=f'$\\alpha$={a:g}') for (a, mk) in levels]
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 0.9), handletextpad=0.05, borderaxespad=0.1, labelspacing=0.25)
    return out

def real_rates(study):
    fs = [f for f in sorted(glob.glob(f'{mc.RESDIR}/h2h_realpower_{study}*.parquet')) if 'smoke' not in os.path.basename(f)]
    if not fs:
        raise FileNotFoundError(study)
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).rename(columns={'p_companion': 'p_mean'})
    R = L.rates(df, ['cond'])
    R['eff_mean_avg'] = R.cond.map(df.groupby('cond').eff_mean.mean())
    return R.set_index('cond')

def panel_real_power(ax, study):
    R = real_rates(study)
    pats = REAL[study]
    tests = TESTS if study == 'oneil' else ('tstd', 'scan')
    x = np.arange(len(pats))
    vals = {t: [R.loc[c, t] for (c, _) in pats] for t in tests}
    bars(ax, x, vals, tests)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({R.loc[c, 'eff_mean_avg']:.3f})" for (c, p) in pats])
    ax.set_xlim(-0.6, len(pats) - 0.4)
    ax.set_ylim(0, 1.05)
    return R

def main():
    fig = plt.figure(figsize=(ST.TEXTWIDTH, 4.75))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.3, 1.3], hspace=0.75, wspace=0.34, left=0.085, right=0.995, top=0.955, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    try:
        A = panel_size(ax)
    except FileNotFoundError:
        if not PREVIEW_PARTIAL:
            raise
        A = {}
    ST.panel_label(ax, 'A', x=-0.36)
    axB = fig.add_subplot(gs[0, 1])
    RB = panel_power(axB, 'S')
    axB.set_ylabel('Power at level 0.05')
    axB.legend(loc='upper left', handlelength=1.0, borderaxespad=0.1, labelspacing=0.25)
    ST.panel_label(axB, 'B', x=-0.2)
    axC = fig.add_subplot(gs[0, 2], sharey=axB)
    RC = panel_power(axC, 'P')
    ST.panel_label(axC, 'C', x=-0.2)
    ax = fig.add_subplot(gs[1, 0])
    D = panel_real_null(ax)
    ST.panel_label(ax, 'D', x=-0.36)
    Rreal = {}
    for (k, study) in enumerate(('oneil', 'almanac')):
        ax = fig.add_subplot(gs[1, 1 + k])
        try:
            Rreal[study] = panel_real_power(ax, study)
        except FileNotFoundError:
            if not PREVIEW_PARTIAL:
                raise
        if k == 0:
            ax.set_ylabel('Power at level 0.05')
        ax.set_xlabel('Pattern (mean Bliss excess)')
        ST.panel_label(ax, 'EF'[k], x=-0.2)
    out = ST.save(fig, 'fig3_head_to_head' + ('_partial' if PREVIEW_PARTIAL else ''))
    print('A worst exact-Bliss rejection rate at 0.05 (matched laws):')
    for ((t, des, I), w) in A.items():
        print(f"  {t:5s} {des}{I}: {w['rate']:.4f} [{w['lo']:.4f}, {w['hi']:.4f}]  {w['cell']}")
    for (nm, R) in (('B (design S)', RB), ('C (design P)', RC)):
        print(nm)
        for (c, _) in PAT:
            r = R.loc[c]
            print(f"  {c:18s} eff {r.eff_mean:.3f}/{r.eff_max:.3f}  T_SIR {r.tstd:.3f}  scan {r.scan:.3f}  mean {r['mean']:.3f}")
    print('D (null_summary.csv):', *[f'{s} {c} {a}: {v:.4f} [{lo:.4f}, {hi:.4f}] n={n}' for (s, c, a, v, lo, hi, n) in D], sep='\n  ')
    for (study, R) in Rreal.items():
        print('E/F', study)
        for (c, r) in R.iterrows():
            (lo, hi) = mc.wilson(r[f'k_tstd'], r.n)
            print(f"  {c:18s} n {int(r.n)}  eff {r.eff_mean_avg:.3f}  T_SIR {r.tstd:.4f} [{lo:.4f}, {hi:.4f}]  scan {r.scan:.4f}  companion {r['mean']:.4f}")
    return out
if __name__ == '__main__':
    main()
