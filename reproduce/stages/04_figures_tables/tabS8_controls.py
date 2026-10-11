"""tabS8_controls calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
RES = str(mc.paths.REAL)
ROWS = [('ONEIL', 'ON-A + A2 + B pooled', "O'Neil, same-line evidence (MK-8776 + gemcitabine; MK-2206 + chemotherapy; WEE1 + mTOR in KRAS-mutant lines)", True), ('ONEIL', 'ON-S ', "O'Neil, AZD1775 + ridaforolimus, all lines", False), ('ONEIL', 'ON-P ', "O'Neil, pair-level literature positives, all lines", False), ('ONEIL', 'ON-N1 ', "O'Neil, MK-8776 + 5-FU or SN-38 (negative)", False), ('ONEIL', 'ON-N3 ', "O'Neil, pairs of inactive single agents", False), ('ALMANAC', 'AL-XG+ ', 'ALMANAC, in vitro positive with xenograft $P<0.05$', True), ('ALMANAC', 'AL-ALL', 'ALMANAC, all source-reported in vitro positives', True), ('ALMANAC', 'AL-N1 ', 'ALMANAC, nilotinib + paclitaxel, negative ComboScore', False), ('ALMANAC', 'AL-N2 ', 'ALMANAC, pairs of inactive single agents', True)]

def fmt_p(p):
    if not np.isfinite(p):
        return ''
    if p == 0:
        return '$<10^{-300}$'
    if p < 0.001:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        if round(m) == 10:
            (m, e) = (1, e + 1)
        return f'${int(round(m))}\\times10^{{{e}}}$'
    if p < 0.01:
        return f'{p:.2g}'
    return f'{p:.2f}'

def main():
    S = pd.read_csv(os.path.join(RES, 'controls_summary.csv'))
    lines = ['\\begin{tabular}{@{}p{0.36\\linewidth}rlll@{}}', '\\toprule', 'Group & Clusters & Percentile & Wilcoxon $p$ & $T_{\\mathrm{SIR}}$ $p\\le0.05$ (background)\\\\', '\\midrule']
    for (st, g, lab, ci) in ROWS:
        r = S[(S.study == st) & S.group.str.startswith(g)]
        assert len(r) == 1, (st, g)
        r = r.iloc[0]
        n = int(r.n_clusters)
        if n < 5:
            pct = f'{r.mean_pct_mean_excess:.3f}'
            k = int(round(r.rate_tstd_p05 * n))
            det = f'{k} of {n} ({r.bg_rate_tstd_p05:.3f})'
        else:
            pct = f'{r.mean_pct_mean_excess:.3f} ({r.median_pct_mean_excess:.3f})'
            det = f'{r.rate_tstd_p05:.3f}'
            if ci:
                det += f' [{r.rate_tstd_p05_lo:.3f}, {r.rate_tstd_p05_hi:.3f}]'
            det += f' ({r.bg_rate_tstd_p05:.3f})'
        lines.append(f'{lab} & {n:,} & {pct} & {fmt_p(r.wilcoxon_p_pct_mean_excess)} & {det}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    body = '\n'.join(lines)
    print(body)
    ST.write_table('tabS8_controls', body)
if __name__ == '__main__':
    main()
