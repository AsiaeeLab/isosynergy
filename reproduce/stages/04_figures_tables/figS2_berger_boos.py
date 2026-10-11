"""Supplementary S2_berger_boos calculations."""
import mc
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as ST
MS = [1, 4, 16, 64, 256]
NOISES = ['add', 'logn', 'oneil']
NOISE_LAB = {'add': 'Additive', 'logn': 'Log-normal', 'oneil': "O'Neil SD"}
NOISE_MK = {'add': 'o', 'logn': 's', 'oneil': '^'}
BASES = ['weak', 'moderate', 'strong']
METH = [('oracle', 'Oracle (true curves)', ST.COL['oracle'], 'o'), ('plugin', 'Plug-in bootstrap (' + ST.TSTD + ' reference)', ST.COL['tstd'], 's'), ('bb', 'Berger-Boos search', ST.COL['bb'], 'v')]

def load(tag):
    R = mc.INPUTS
    rows = pd.concat([pd.read_csv(os.path.join(R, 'bb_table_rows.csv')), pd.read_csv(os.path.join(R, 'bb_table_rows_ext.csv'))], ignore_index=True)
    rows = rows[rows.cond.isin(['bliss', 'syn_broad', 'syn_high', 'syn_corner', 'syn_sat'])].copy()
    pl = pd.read_parquet(os.path.join(mc.RESDIR, f'bb_plugin{tag}.parquet'))
    g = pl.groupby(['noise', 'base', 'ms', 'cond']).agg(n_pl=('p_tstd', 'size'), k_pl=('p_tstd', lambda p: int((p <= 0.05).sum())))
    rows = rows.merge(g.reset_index(), on=['noise', 'base', 'ms', 'cond'], how='left')
    rows['oracle'] = rows['k_oracle_0.05'] / rows['n']
    rows['plugin'] = rows['k_pl'] / rows['n_pl']
    rows['bb'] = rows['k_BB_search_0.05'] / rows['n']
    rows.loc[rows.ms == 256, 'bb'] = np.nan
    return rows

def coverage():
    R = mc.INPUTS
    c = pd.concat([pd.read_csv(os.path.join(R, 'bb_coverage.csv')), pd.read_csv(os.path.join(R, 'bb_coverage_ext.csv'))], ignore_index=True)
    return c[(c.grid == '5x5') & (c.method == 'exact')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    rows = load(a.tag)
    cov = coverage()
    x = {m: i for (i, m) in enumerate(MS)}
    (fig, axs) = plt.subplots(2, 2, figsize=(6.3, 4.5))
    ax = axs[0, 0]
    for (ni, noise) in enumerate(NOISES):
        for (bi, base) in enumerate(BASES):
            d = cov[(cov.noise == noise) & (cov.base == base)].sort_values('ms')
            off = (ni - 1) * 0.24 + (bi - 1) * 0.07
            xx = np.array([x[m] for m in d.ms]) + off
            ax.errorbar(xx, d.coverage, yerr=np.maximum(0, [d.coverage - d.cov_lo, d.cov_hi - d.coverage]), fmt=NOISE_MK[noise], ms=3.2, color=ST.COL['bb'], mfc='white' if bi == 0 else ST.COL['bb'] if bi == 2 else '0.8', mew=0.8, elinewidth=0.7, capsize=0, label=NOISE_LAB[noise] if bi == 1 else None)
    ax.axhline(0.995, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(MS)))
    ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlim(-0.6, len(MS) - 0.4)
    ax.set_ylim(0.9935, 0.997)
    ax.set_yticks([0.994, 0.995, 0.996, 0.997])
    ax.set_xlabel('Single-agent replicates per dose, $m_s$')
    ax.set_ylabel('Simultaneous coverage')
    h = [Line2D([], [], color=ST.COL['bb'], marker=NOISE_MK[n], ls='', ms=3.5) for n in NOISES]
    h += [Line2D([], [], color=ST.COL['bb'], marker='o', ls='', ms=3.5, mfc=f, mew=0.8) for f in ('white', '0.8', ST.COL['bb'])]
    ax.legend(h, [NOISE_LAB[n] for n in NOISES] + ['Weak drugs', 'Moderate drugs', 'Strong drugs'], ncol=2, loc='upper right', handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    ST.panel_label(ax, 'A')
    ax = axs[0, 1]
    bl = rows[rows.cond == 'bliss']
    for (mi, (key, lab, col, mk)) in enumerate(METH):
        for m in MS:
            d = bl[bl.ms == m]
            v = d[key].to_numpy()
            if np.all(np.isnan(v)):
                continue
            jit = np.linspace(-0.07, 0.07, len(v))
            ax.plot(x[m] + (mi - 1) * 0.27 + jit, v, ls='', marker=mk, ms=2.8, color=col, alpha=0.9, mfc=col if key != 'plugin' else 'white', mew=0.7)
    ax.axhline(0.05, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(MS)))
    ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlim(-0.6, len(MS) - 0.4)
    ax.set_ylim(-0.003, 0.075)
    ax.set_xlabel('Single-agent replicates per dose, $m_s$')
    ax.set_ylabel('Rejection rate at exact Bliss')
    ST.panel_label(ax, 'B')
    pw = rows[rows.cond != 'bliss']
    summ = []
    ax = axs[1, 0]
    for (mi, (key, lab, col, mk)) in enumerate(METH):
        (xx, yy, lo, hi) = ([], [], [], [])
        for m in MS:
            v = pw[pw.ms == m][key].to_numpy()
            if np.all(np.isnan(v)):
                continue
            xx.append(x[m] + (mi - 1) * 0.1)
            yy.append(np.mean(v))
            (q1, q3) = np.quantile(v, [0.25, 0.75])
            lo.append(np.mean(v) - q1)
            hi.append(q3 - np.mean(v))
        ax.errorbar(xx, yy, yerr=np.maximum(0, [np.maximum(lo, 0), np.maximum(hi, 0)]), color=col, marker=mk, ms=3.5, mfc=col if key != 'plugin' else 'white', mew=0.8, lw=1.0, elinewidth=0.7, capsize=1.5, label=lab)
    (hC, lC) = ax.get_legend_handles_labels()
    ax.set_xticks(range(len(MS)))
    ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlim(-0.6, len(MS) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Single-agent replicates per dose, $m_s$')
    ax.set_ylabel('Mean power')
    ST.panel_label(ax, 'C')
    ax = axs[1, 1]
    for (mi, (key, lab, col, mk)) in enumerate(METH[1:], start=1):
        (xx, yy) = ([], [])
        for m in MS:
            d = pw[(pw.ms == m) & (pw.oracle >= 0.3)]
            r = (d[key] / d.oracle).to_numpy()
            if np.all(np.isnan(r)):
                continue
            xx.append(x[m])
            yy.append(np.nanmean(r))
        ax.plot(xx, yy, color=col, marker=mk, ms=3.5, mfc=col if key != 'plugin' else 'white', mew=0.8, lw=1.0)
    ax.axhline(1.0, color=ST.COL['oracle'], lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(MS)))
    ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlim(-0.6, len(MS) - 0.4)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Single-agent replicates per dose, $m_s$')
    ax.set_ylabel('Power relative to the oracle')
    ax.legend(hC, lC, loc='lower right', borderaxespad=0.2)
    ST.panel_label(ax, 'D')
    for m in MS:
        d = rows[rows.ms == m]
        b = d[d.cond == 'bliss']
        p = d[d.cond != 'bliss']
        p3 = p[p.oracle >= 0.3]
        rec = dict(ms=m, n_rows_power=len(p), n_rows_ratio=len(p3))
        for key in ('oracle', 'plugin', 'bb'):
            rec[f'size_min_{key}'] = b[key].min()
            rec[f'size_max_{key}'] = b[key].max()
            rec[f'power_mean_{key}'] = p[key].mean()
            rec[f'ratio_{key}'] = (p3[key] / p3.oracle).mean()
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(mc.RESDIR, f'figS2_summary{a.tag}.csv'), index=False)
    pd.set_option('display.width', 250)
    print(S.round(4).to_string())
    pl = rows[rows.cond == 'bliss']
    print('plug-in Bliss size by ms:', pl.groupby('ms').plugin.agg(['min', 'max', 'mean']).round(4).to_string())
    print('pooled plug-in Bliss size m_s<=16:', pl[pl.ms <= 16].k_pl.sum() / pl[pl.ms <= 16].n_pl.sum())
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    ST.save(fig, 'figS2_berger_boos' + a.tag)
if __name__ == '__main__':
    main()
