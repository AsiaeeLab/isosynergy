"""Supplementary S3_positive_controls calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
RES = str(mc.paths.REAL)
COL = {'ONEIL': ST.COL['oneil'], 'ALMANAC': ST.COL['almanac']}
(POS, NEG) = (ST.COL['oneil'], ST.COL['almanac'])
LAB = {'ONEIL': "O'Neil", 'ALMANAC': 'ALMANAC'}
GRAY = '0.45'
WIDTH = 6.45

def pct_within(d, col):
    r = d.groupby('cell_line')[col].rank(method='average')
    n = d.groupby('cell_line')[col].transform(lambda s: s.notna().sum())
    return (r - 0.5) / n

def drug_label(pair):
    out = []
    for t in pair.split(' + '):
        if t == 'MITOMYCINE':
            out.append('mitomycin C')
        elif any((ch.isdigit() for ch in t)):
            out.append(t)
        else:
            out.append(t.lower())
    return ' + '.join(out)

def inactive_clusters(st, ctrl_ids, S):
    f = 'screen_oneil_q.parquet' if st == 'ONEIL' else 'screen_almanac_q.parquet'
    d = pd.read_parquet(os.path.join(RES, f), columns=['unit_id', 'cell_line', 'drugA', 'drugB', 'mean_excess', 'edgeA_top', 'edgeB_top'])
    d['pct'] = pct_within(d, 'mean_excess')
    d['pair'] = np.where(d.drugA < d.drugB, d.drugA + ' + ' + d.drugB, d.drugB + ' + ' + d.drugA)
    w = d[(d.edgeA_top >= 0.9) & (d.edgeB_top >= 0.9) & ~d.unit_id.isin(ctrl_ids)]
    cl = w.groupby(w.pair + '|' + w.cell_line).pct.mean().values
    row = S[(S.study == st) & S.group.str.startswith('ON-N3' if st == 'ONEIL' else 'AL-N2')].iloc[0]
    assert len(w) == row.n_units and cl.size == row.n_clusters, (st, len(w), cl.size)
    assert abs(cl.mean() - row.mean_pct_mean_excess) < 1e-09 and abs(np.median(cl) - row.median_pct_mean_excess) < 1e-09
    print(f'{st} inactive pairs: {len(w):,} units, {cl.size:,} clusters, mean {cl.mean():.4f}, median {np.median(cl):.4f}')
    return cl

def strip_panel(ax, U, S, st, groups):
    u = U[U.study == st].copy()
    u['cluster'] = u.pair + '|' + u.cell_line
    for (x, (g, lab)) in enumerate(groups):
        h = u[u.group.str.startswith(g)].groupby('cluster').pct_mean_excess.mean().values
        row = S[(S.study == st) & S.group.str.startswith(g)].iloc[0]
        assert h.size == row.n_clusters, (g, h.size, row.n_clusters)
        col = NEG if lab.startswith('N') else POS
        jit = np.random.default_rng(x).uniform(-0.22, 0.22, h.size)
        ax.scatter(x + jit, h, s=6, color=col, alpha=0.55, edgecolors='none')
        if h.size >= 3:
            se = h.std(ddof=1) / np.sqrt(h.size)
            ax.errorbar(x, h.mean(), yerr=np.maximum(0, 1.96 * se), fmt='o', color='k', ms=3.2, lw=0.9, capsize=1.8)
        else:
            ax.plot(x, h.mean(), 'o', color='k', ms=3.2)
        ax.text(x, 1.035, f'{h.size:,}', ha='center', va='bottom', fontsize=7, color='0.3')
    ctrl_ids = set(u.unit_id)
    cl = inactive_clusters(st, ctrl_ids, S)
    xv = len(groups)
    vp = ax.violinplot([cl], positions=[xv], widths=0.75, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(NEG)
        b.set_edgecolor('none')
        b.set_alpha(0.35)
    se = cl.std(ddof=1) / np.sqrt(cl.size)
    ax.errorbar(xv, cl.mean(), yerr=np.maximum(0, 1.96 * se), fmt='o', color='k', ms=3.2, lw=0.9, capsize=1.8)
    ax.text(xv, 1.035, f'{cl.size:,}', ha='center', va='bottom', fontsize=7, color='0.3')
    ax.axhline(0.5, color=GRAY, lw=0.7, ls='--')
    ax.set_xticks(range(len(groups) + 1))
    ax.set_xticklabels([lab for (_, lab) in groups] + ['N3' if st == 'ONEIL' else 'N2'])
    ax.set_xlim(-0.6, xv + 0.6)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(f'{LAB[st]}: within-cell-line\npercentile of $\\hat{{B}}$')
    ax.set_xlabel('Control group')

def main():
    S = pd.read_csv(os.path.join(RES, 'controls_summary.csv'))
    U = pd.read_csv(os.path.join(RES, 'controls_units.csv'), low_memory=False)
    PL = pd.read_csv(os.path.join(RES, 'controls_pair_level.csv'))
    (fig, axs) = plt.subplots(2, 2, figsize=(WIDTH, 5.35), gridspec_kw=dict(width_ratios=[1.3, 1]))
    order_o = [('ON-A ', 'A'), ('ON-A2 ', 'A2'), ('ON-B ', 'B'), ('ON-S ', 'S'), ('ON-P ', 'P'), ('ON-X ', 'X'), ('ON-N1 ', 'N1'), ('ON-N2 ', 'N2')]
    order_a = [('AL-XG+ ', 'X+'), ('AL-XGw ', 'Xw'), ('AL-XG0 ', 'X0'), ('AL-XGu ', 'Xu'), ('AL-IV ', 'IV'), ('AL-N1 ', 'N1')]
    strip_panel(axs[0, 0], U, S, 'ONEIL', order_o)
    ST.panel_label(axs[0, 0], 'A', x=-0.2, y=1.06)
    strip_panel(axs[1, 0], U, S, 'ALMANAC', order_a)
    axs[1, 0].set_xticklabels(['X$+$', 'Xw', 'X0', 'Xu', 'IV', 'N1', 'N2'])
    ST.panel_label(axs[1, 0], 'C', x=-0.2, y=1.06)
    ax = axs[0, 1]
    PL = PL.sort_values('rank_mean_pct_mean_excess').reset_index(drop=True)
    y = np.arange(len(PL))
    col = np.where(PL.group.str.startswith('ON-N1'), NEG, POS)
    ax.scatter(PL.rank_mean_pct_mean_excess, y, color=col, s=15, zorder=3)
    ax.scatter(PL.rank_among_drug_sharing_pairs, y, facecolors='none', edgecolors=col, s=15, lw=0.9, zorder=3)
    ax.axvline(0.5, color=GRAY, lw=0.7, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([drug_label(p) for p in PL.pair], fontsize=7)
    ax.set_ylim(-0.7, len(PL) - 0.3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Rank (fraction of pairs below)')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', color='0.3', ms=4, label='All pairs'), Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', ms=4, label='Pairs sharing a drug')], loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
    ST.panel_label(ax, 'B', x=-0.75, y=1.06)
    ax = axs[1, 1]
    rows = [('ONEIL', 'ON-A + A2 + B pooled', 'Same-line'), ('ONEIL', 'ON-S ', 'S'), ('ONEIL', 'ON-P ', 'P'), ('ONEIL', 'ON-N1 ', 'N1'), ('ALMANAC', 'AL-XG+ ', 'X$+$'), ('ALMANAC', 'AL-ALL', 'All IV$+$'), ('ALMANAC', 'AL-N2 ', 'N2')]
    for (x, (st, g, lab)) in enumerate(rows):
        r = S[(S.study == st) & S.group.str.startswith(g)].iloc[0]
        ax.bar(x - 0.18, r.rate_tstd_p05, width=0.34, color=COL[st], alpha=0.9)
        ax.errorbar(x - 0.18, r.rate_tstd_p05, yerr=np.maximum(0, [[r.rate_tstd_p05 - r.rate_tstd_p05_lo], [r.rate_tstd_p05_hi - r.rate_tstd_p05]]), fmt='none', color='k', lw=0.8, capsize=1.5)
        ax.bar(x + 0.18, r.bg_rate_tstd_p05, width=0.34, color='0.25')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[2] for r in rows], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Share with $T_{\\mathrm{SIR}}$ $p\\leq0.05$')
    ax.set_ylim(0, 1.0)
    ax.legend(handles=[Patch(color=COL['ONEIL'], label="O'Neil"), Patch(color=COL['ALMANAC'], label='ALMANAC'), Patch(color='0.25', label='Background')], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0, columnspacing=0.9, borderaxespad=0.2)
    ST.panel_label(ax, 'D', x=-0.24, y=1.06)
    fig.tight_layout(w_pad=1.2, h_pad=1.8)
    ST.save(fig, 'figS3_positive_controls', pad=0.02)
if __name__ == '__main__':
    main()
