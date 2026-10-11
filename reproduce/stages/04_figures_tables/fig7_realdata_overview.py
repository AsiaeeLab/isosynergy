"""fig7_realdata_overview calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
RES = str(mc.paths.REAL)
COL = {'ONEIL': ST.COL['oneil'], 'ALMANAC': ST.COL['almanac']}
LAB = {'ONEIL': "O'Neil", 'ALMANAC': 'ALMANAC'}
GRAY = '0.45'

def density(ax, x, col, lab, lo, hi, bins=120):
    x = x[np.isfinite(x)]
    (h, e) = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi), density=True)
    ax.plot(0.5 * (e[1:] + e[:-1]), h, color=col, lw=1.1, label=lab)

def family_shares(dO, dA):
    fam = pd.read_csv(os.path.join(RES, 'family_calls.csv'))
    gO = dO.groupby('family')
    sO = pd.DataFrame(dict(units=gO.size(), bh_tstd=gO.q_tstd.apply(lambda q: int((q <= 0.05).sum())), bh_companion=gO.q_companion.apply(lambda q: int((q <= 0.05).sum()))))
    gA = dA.groupby('family')
    sA = pd.DataFrame(dict(units=gA.size(), nominal_tstd=gA.p_tstd.apply(lambda p: int((p <= 0.05).sum()))))
    f = fam.set_index('family')
    for (s, cols) in ((sO, ['units', 'bh_tstd', 'bh_companion']), (sA, ['units', 'nominal_tstd'])):
        d = f.loc[s.index, cols].values - s[cols].values
        assert np.abs(d).max() == 0, 'family counts differ from family_calls.csv'
    out = pd.concat([pd.DataFrame(dict(study='ONEIL', family=sO.index, units=sO.units.values, share_bh_tstd=(sO.bh_tstd / sO.units).values, share_bh_block_t=(sO.bh_companion / sO.units).values)), pd.DataFrame(dict(study='ALMANAC', family=sA.index, units=sA.units.values, share_p05_tstd=(sA.nominal_tstd / sA.units).values))], ignore_index=True)
    return out

def main():
    dO = pd.read_parquet(os.path.join(RES, 'screen_oneil_q.parquet'), columns=['family', 'activity', 'mean_excess', 'q_tstd', 'q_companion'])
    dA = pd.read_parquet(os.path.join(RES, 'screen_almanac_q.parquet'), columns=['family', 'activity', 'mean_excess', 'p_tstd'])
    rp = pd.read_parquet(os.path.join(RES, 'repro_pairs_oneil.parquet'))
    nul = pd.read_csv(os.path.join(RES, 'null_summary.csv'))
    null_alm = float(nul[(nul.study == 'almanac') & (nul.column == 'p_tstd') & (nul.level == 0.05)].rate.iloc[0])
    sh = family_shares(dO, dA)
    sh.to_csv(os.path.join(ST.RESDIR, 'fig7_family_shares.csv'), index=False)
    print(f"units: O'Neil {len(dO):,}, ALMANAC {len(dA):,}; pairs in two batches {len(rp)}; ALMANAC null rate T_std at 0.05 = {null_alm:.4f}")
    (o, a) = (sh[sh.study == 'ONEIL'], sh[sh.study == 'ALMANAC'])
    print(f"families: O'Neil {len(o)}, ALMANAC {len(a)}")
    print("medians: O'Neil BH T_std %.3f, BH block-level t %.3f; ALMANAC T_std p<=0.05 %.4f (range %.4f to %.4f; %d of %d families above the null rate)" % (o.share_bh_tstd.median(), o.share_bh_block_t.median(), a.share_p05_tstd.median(), a.share_p05_tstd.min(), a.share_p05_tstd.max(), int((a.share_p05_tstd > null_alm).sum()), len(a)))
    from scipy.stats import spearmanr
    for c in ('activity', 'mean_excess', 't_std'):
        print(c, 'Spearman batch 1 vs 2: %.3f' % spearmanr(rp[c + '_b0'], rp[c + '_b1']).statistic)
    fig = plt.figure(figsize=(ST.TEXTWIDTH, 4.55))
    gs = GridSpec(2, 3, figure=fig, left=0.095, right=0.985, top=0.95, bottom=0.11, wspace=0.45, hspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    for (st, d) in (('ONEIL', dO), ('ALMANAC', dA)):
        density(ax, d.activity.values, COL[st], LAB[st], -0.3, 1.05)
    ax.set_xlabel('Activity, mean of $1-Y$')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', handlelength=1.2)
    ST.panel_label(ax, 'A', x=-0.3)
    ax = fig.add_subplot(gs[0, 1])
    for (st, d) in (('ONEIL', dO), ('ALMANAC', dA)):
        density(ax, d.mean_excess.values, COL[st], LAB[st], -0.3, 0.3, bins=150)
    ax.set_xlim(-0.31, 0.31)
    ax.axvline(0, color=GRAY, lw=0.7, ls='--')
    ax.set_xlabel('Mean Bliss excess $\\hat{B}$')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', handlelength=1.2, bbox_to_anchor=(1.04, 1.0))
    ST.panel_label(ax, 'B', x=-0.3)
    for (k, (col, lab, lim, cell)) in enumerate((('activity', 'Activity', (-0.2, 1.0), gs[0, 2]), ('mean_excess', 'Mean Bliss excess $\\hat{B}$', (-0.25, 0.45), gs[1, 0]))):
        ax = fig.add_subplot(cell)
        ax.scatter(rp[col + '_b0'], rp[col + '_b1'], s=5, color=COL['ONEIL'], alpha=0.6, edgecolors='none')
        ax.plot(lim, lim, color=GRAY, lw=0.7, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'{lab}, batch 1')
        ax.set_ylabel(f'{lab}, batch 2')
        ST.panel_label(ax, 'CD'[k], x=-0.3)
    ax = fig.add_subplot(gs[1, 1])
    x0 = np.log10(1 + rp.t_std_b0.values)
    x1 = np.log10(1 + rp.t_std_b1.values)
    ax.scatter(x0, x1, s=5, color=COL['ONEIL'], alpha=0.6, edgecolors='none')
    m = max(x0.max(), x1.max()) * 1.05
    ax.plot([0, m], [0, m], color=GRAY, lw=0.7, ls='--')
    ax.set_xlabel('$\\log_{10}(1+T_{\\mathrm{SIR}})$, batch 1')
    ax.set_ylabel('$\\log_{10}(1+T_{\\mathrm{SIR}})$, batch 2')
    ST.panel_label(ax, 'E', x=-0.3)
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 2], width_ratios=[1.75, 1.0], wspace=0.95)
    axO = fig.add_subplot(sub[0, 0])
    axA = fig.add_subplot(sub[0, 1])
    rng = np.random.default_rng(3)
    for (x, c) in ((0, 'share_bh_tstd'), (1, 'share_bh_block_t')):
        v = o[c].values
        axO.scatter(x + rng.uniform(-0.2, 0.2, v.size), v, s=7, color=COL['ONEIL'], alpha=0.75, edgecolors='none')
        axO.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color='k', lw=1.2)
    axO.set_xticks([0, 1])
    axO.set_xticklabels([ST.TSTD, 'Block-\nlevel $t$'])
    axO.set_xlim(-0.55, 1.55)
    axO.set_ylim(-0.02, 1.02)
    axO.set_ylabel('Share with BH $q\\leq0.05$')
    axO.set_xlabel("O'Neil")
    v = a.share_p05_tstd.values
    axA.axhline(null_alm, color=GRAY, lw=0.8, ls='--', zorder=0)
    axA.scatter(rng.uniform(-0.2, 0.2, v.size), v, s=9, facecolors='none', edgecolors=COL['ALMANAC'], lw=0.7)
    axA.plot([-0.3, 0.3], [np.median(v)] * 2, color='k', lw=1.2)
    axA.set_xticks([0])
    axA.set_xticklabels([ST.TSTD])
    axA.set_xlim(-0.5, 0.5)
    axA.set_ylim(0, max(0.16, v.max() * 1.08))
    axA.set_yticks([0, 0.05, 0.1, 0.15])
    axA.set_ylabel('Share with $p\\leq0.05$')
    axA.set_xlabel('ALMANAC')
    for axx in (axO, axA):
        axx.xaxis.set_label_coords(0.5, -0.29)
    ST.panel_label(axO, 'F', x=-0.62)
    ST.save(fig, 'fig7_realdata_overview', pad=0.02)
if __name__ == '__main__':
    main()
