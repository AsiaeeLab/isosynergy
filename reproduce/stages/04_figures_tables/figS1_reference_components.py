"""Supplementary S1_reference_components calculations."""
import mc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as ST
LIGHT = '#56B4E9'
XM = {1.0: 0, 4.0: 1, 8.0: 2, np.inf: 3}
XLAB = ['1', '4', '8', '$\\infty$']

def cl_interval(rej, cluster):
    df = pd.DataFrame(dict(r=np.asarray(rej, float), g=np.asarray(cluster)))
    g = df.groupby('g')['r'].agg(['sum', 'count'])
    n = g['count'].sum()
    p = g['sum'].sum() / n
    k = len(g)
    resid = g['sum'] - p * g['count']
    se = np.sqrt(k / (k - 1) * np.sum(resid ** 2)) / n
    return (p, max(0.0, p - 1.96 * se), p + 1.96 * se)

def load(tag=''):
    R = mc.RESDIR
    d = pd.read_parquet(os.path.join(R, f'stress_shared_null{tag}.parquet'))
    f = pd.read_csv(os.path.join(R, f'stress_shared_null_family{tag}.csv'), keep_default_na=False, na_values=[''])
    s = pd.read_parquet(os.path.join(R, f'stress_shift{tag}.parquet'))
    rp = os.path.join(R, f'stress_reps{tag}.parquet')
    r = pd.read_parquet(rp) if os.path.exists(rp) else None
    return (d, f, s, r)

def per_matrix(d, col, m, ref):
    g = d[(d.share == 'shared') & (d.m_s == m) & (d.ref == (ref if np.isfinite(m) else bytes([82, 48]).decode('ascii')))]
    return cl_interval(g[col] <= 0.05, g.screen)

def fwer(f, m, share, ref):
    g = f[(f.m_s == m) & (f.share == share) & (f.ref == ref)]
    (k, n) = (int(g.any_disc.sum()), len(g))
    (lo, hi) = mc.wilson(k, n)
    return (k / n, float(lo), float(hi), n)

def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    (d, f, s, rp) = load(tag)
    (fig, axs) = plt.subplots(2, 2, figsize=(6.3, 4.8))
    axs = axs.ravel()
    ax = axs[0]
    tests = [('p_t1', ST.COL['tstd'], 'o'), ('p_s1', ST.COL['scan'], 's'), ('p_m', ST.COL['mean'], '^')]
    for (k, (col, c, mk)) in enumerate(tests):
        for (ref, ls, fill) in ((bytes([82, 48]).decode('ascii'), '-', True), (bytes([82, 49]).decode('ascii'), (0, (3, 1.5)), False)):
            (xx, yy, lo, hi) = ([], [], [], [])
            for (m, x) in XM.items():
                (p, l, h) = per_matrix(d, col, m, ref)
                xx.append(x + (k - 1) * 0.09)
                yy.append(p)
                lo.append(p - l)
                hi.append(h - p)
            ax.errorbar(xx, yy, yerr=np.maximum(0, [lo, hi]), color=c, ls=ls, marker=mk, ms=3.6, mfc=c if fill else 'white', mew=0.8, lw=1.0, elinewidth=0.7, capsize=0)
    ax.axhline(0.05, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_yscale('log')
    ax.set_ylim(0.012, 1.6)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.4'])
    ax.minorticks_off()
    ax.set_xticks(range(4))
    ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xlabel('Single-agent replicates, $m_s$')
    ax.set_ylabel('Rejection rate at exact Bliss')
    h = [Line2D([], [], color=c, marker=mk, ms=3.6, lw=1.0) for (_, c, mk) in tests]
    lab = [ST.TSTD, 'Region scan', 'Mean test (fixed-edge $t$)']
    ax.legend(h, lab, loc='upper right', ncol=1, handlelength=1.8, borderaxespad=0.1)
    ST.panel_label(ax, 'A', x=-0.2)
    ax = axs[1]
    series = [('shared', bytes([82, 48]).decode('ascii'), ST.COL['tstd'], 'o', '-', True), ('indep', bytes([82, 48]).decode('ascii'), LIGHT, 's', '-', True), ('shared', bytes([82, 49]).decode('ascii'), ST.COL['tstd'], 'o', (0, (3, 1.5)), False), ('indep', bytes([82, 49]).decode('ascii'), LIGHT, 's', (0, (3, 1.5)), False)]
    off = [-0.09, 0.09, -0.09, 0.09]
    for (q, (sh, ref, c, mk, ls, fill)) in enumerate(series):
        (xx, yy, lo, hi) = ([], [], [], [])
        for m in (1.0, 4.0, 8.0):
            (p, l, hh, n) = fwer(f, m, sh, ref)
            xx.append(XM[m] + off[q])
            yy.append(p)
            lo.append(p - l)
            hi.append(hh - p)
        ax.errorbar(xx, yy, yerr=np.maximum(0, [lo, hi]), color=c, ls=ls, marker=mk, ms=3.6, mfc=c if fill else 'white', mew=0.8, lw=1.0, elinewidth=0.7, capsize=0)
    (p, l, hh, n) = fwer(f, np.inf, 'shared', bytes([82, 48]).decode('ascii'))
    ax.errorbar([3], [p], yerr=np.maximum(0, [[p - l], [hh - p]]), color='0.15', marker='D', ms=3.6, ls='', elinewidth=0.7, capsize=0)
    ax.axhline(0.05, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(range(4))
    ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xlabel('Single-agent replicates, $m_s$')
    ax.set_ylabel('Probability of any BH call')
    h = [Line2D([], [], color=ST.COL['tstd'], marker='o', ms=3.6, lw=1.0), Line2D([], [], color=LIGHT, marker='s', ms=3.6, lw=1.0), Line2D([], [], color='0.15', marker='D', ms=3.6, ls='')]
    lab = ['Shared curves', 'Independent curves', 'Exact curves']
    ax.legend(h, lab, loc='upper right', ncol=1, handlelength=1.8, borderaxespad=0.1)
    ST.panel_label(ax, 'B', x=-0.2)
    ax = axs[2]
    R = [0.0, 0.1, 0.25, 0.5]
    for (m, c, mk) in ((np.inf, ST.COL['tstd'], 'o'), (4.0, LIGHT, 's')):
        for (ref, ls, fill, dx) in (('without', '-', True, -0.006), ('with', (0, (3, 1.5)), False, 0.006)):
            (xx, yy, lo, hi) = ([], [], [], [])
            for r in R:
                g = s[(s.m_s == m) & (s.ratio == r) & (s.ref == (ref if r > 0 else 'without'))]
                (k, n) = (int((g.p_t <= 0.05).sum()), len(g))
                (l, hh) = mc.wilson(k, n)
                xx.append(r + dx)
                yy.append(k / n)
                lo.append(k / n - l)
                hi.append(hh - k / n)
            ax.errorbar(xx, yy, yerr=np.maximum(0, [lo, hi]), color=c, ls=ls, marker=mk, ms=3.6, mfc=c if fill else 'white', mew=0.8, lw=1.0, elinewidth=0.7, capsize=0)
    ax.axhline(0.05, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(R)
    ax.set_xticklabels(['0', '0.1', '0.25', '0.5'])
    ax.set_xlim(-0.03, 0.53)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel('Plate shift, $\\tau/\\sigma$')
    ax.set_ylabel('Rejection rate at exact Bliss')
    h = [Line2D([], [], color=ST.COL['tstd'], marker='o', ms=3.6, lw=1.0), Line2D([], [], color=LIGHT, marker='s', ms=3.6, lw=1.0)]
    lab = ['Exact curves ($m_s=\\infty$)', '$m_s=4$, curves redrawn']
    ax.legend(h, lab, loc='upper left', ncol=1, handlelength=1.8, borderaxespad=0.1)
    ST.panel_label(ax, 'C', x=-0.2)
    ax = axs[3]
    if rp is not None:
        AL = [0.05, 0.01, 0.005]
        for (q, (col, lab, c, mk, ls, fill)) in enumerate((('p_single', 'Single draw at half SD', ST.COL['tstd'], 'o', '-', True), ('p_matched', 'Mean of four independent draws', ST.COL['tstd'], 'o', (0, (3, 1.5)), False))):
            (xx, yy, lo, hi) = ([], [], [], [])
            for (k, al) in enumerate(AL):
                (kk, n) = (int((rp[col] <= al + 1e-12).sum()), len(rp))
                (l, hh) = mc.wilson(kk, n)
                xx.append(k + (q - 0.5) * 0.1)
                yy.append(kk / n / al)
                lo.append((kk / n - l) / al)
                hi.append((hh - kk / n) / al)
            ax.errorbar(xx, yy, yerr=np.maximum(0, [lo, hi]), color=c, ls=ls, marker=mk, ms=3.6, mfc=c if fill else 'white', mew=0.8, lw=1.0, elinewidth=0.7, capsize=0, label=lab)
        ax.axhline(1.0, color='0.3', lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['0.05', '0.01', '0.005'])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(0, 1.75)
        ax.set_xlabel('Level $\\alpha$')
        ax.set_ylabel('Rejection rate / $\\alpha$')
        ax.legend(loc='upper right', handlelength=2.4, borderaxespad=0.1, title='Reference plate')
        ST.panel_label(ax, 'D', x=-0.2)
    else:
        ax.set_visible(False)
    hf = [Line2D([], [], color='0.3', marker='o', ms=3.6, lw=1.0), Line2D([], [], color='0.3', marker='o', ms=3.6, lw=1.0, ls=(0, (3, 1.5)), mfc='white', mew=0.8)]
    fig.legend(hf, ['Reference without the noise component', 'Reference with the noise component'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=2.0)
    fig.tight_layout(w_pad=2.0, h_pad=1.5, rect=(0, 0, 1, 0.955))
    ST.save(fig, 'figS1_reference_components' + tag)
if __name__ == '__main__':
    main()
