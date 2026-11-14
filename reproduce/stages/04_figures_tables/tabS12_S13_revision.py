"""Supplementary Tables S12 (isotonic against raw-well statistics) and S13 (offsets in the O'Neil configuration)."""
import os
import pandas as pd
import style as ST

LAB = {'syn_broad_0.20': 'Broad', 'syn_bump_0.30': 'Middle bump', 'syn_hdblock_0.80': 'High-dose block',
       'syn_lowdose_0.10': 'Low dose', 'sat_0.60': 'Saturating', 'syn_corner_1.00': 'Top corner', 'mixed_0.60': 'Zero net'}
ORDER = ['Broad', 'High-dose block', 'Top corner', 'Middle bump', 'Low dose', 'Saturating', 'Zero net']


def f(x):
    return f'{x:.3f}'


def tab_s12():
    s = pd.read_csv(os.path.join(ST.RESDIR, 'raw_benchmark_summary.csv'))
    L = [r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule', r' & & $\Tstd$ & Raw positive part & Raw maximum & Region scan\\', r'\midrule']
    for part, title in [('size', 'Rejection rate at exact Bliss (2,000 matrices per surface)'), ('power', 'Power (1,000 matrices per pattern)'),
                        ('missing', 'Power with two combination wells missing (1,000 per pattern)')]:
        L.append(r'\multicolumn{6}{@{}l}{\emph{' + title + r'}}\\')
        g = s[s.part == part].copy()
        g['lab'] = g.cond.map(LAB)
        for d in ['S', 'P']:
            gg = g[g.design == d]
            if part == 'size':
                for surf in ['weak', 'moderate', 'strong']:
                    r = gg[gg.surface == surf].iloc[0]
                    L.append(f'Design {d} & {surf} single agents & {f(r.tstd)} & {f(r.rawpos)} & {f(r.rawmax)} & {f(r.scan)}\\\\')
            else:
                for o in ORDER:
                    r = gg[gg.lab == o]
                    if len(r):
                        r = r.iloc[0]
                        L.append(f'Design {d} & {o} & {f(r.tstd)} & {f(r.rawpos)} & {f(r.rawmax)} & {f(r.scan)}\\\\')
    L += [r'\midrule', r'\multicolumn{6}{@{}l}{\emph{Largest cell of the excess map inside the synergistic cells (isotonic map; raw map)}}\\']
    g = s[s.part == 'power']
    for d in ['S', 'P']:
        for c, o in [('syn_hdblock_0.80', 'High-dose block'), ('syn_corner_1.00', 'Top corner')]:
            r = g[(g.design == d) & (g.cond == c)].iloc[0]
            L.append(f'Design {d} & {o} & \\multicolumn{{2}}{{r}}{{{f(r.loc_iso)}}} & \\multicolumn{{2}}{{r}}{{{f(r.loc_raw)}}}\\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    ST.write_table('tabS12_raw', '\n'.join(L))


def tab_s13():
    s = pd.read_csv(os.path.join(ST.RESDIR, 'offset_oneil_summary.csv'))
    s = s[s.alpha == 0.05]

    def cell(kind, size, test):
        return f"{s[(s.kind == kind) & (s['size'] == size) & (s.test == test)].iloc[0].rate:.3f}"
    L = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule', r'Offset of the combination wells & $\Tstd$ & Region scan & Block-level $t$\\', r'\midrule',
         r'\multicolumn{4}{@{}l}{\emph{Random offset, $\delta\sim N(0,\tau^2)$ per unit}}\\']
    for t in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        L.append(f"$\\tau={t:g}$ & {cell('random', t, 'p_tstd')} & {cell('random', t, 'p_scan')} & {cell('random', t, 'p_mean')}\\\\")
    L.append(r'\multicolumn{4}{@{}l}{\emph{Fixed offset $\delta$ toward synergy}}\\')
    for t in [0.01, 0.02, 0.04, 0.08]:
        L.append(f"$\\delta={t:g}$ & {cell('fixed', t, 'p_tstd')} & {cell('fixed', t, 'p_scan')} & {cell('fixed', t, 'p_mean')}\\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    ST.write_table('tabS13_offset', '\n'.join(L))


if __name__ == '__main__':
    tab_s12()
    tab_s13()
