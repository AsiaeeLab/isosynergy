"""Table bodies for the real-noise, reproducibility and concentration-window results."""
import paths
import json
import pandas as pd
import style


def table(name, columns, header, rows):
    body = ['\\begin{tabular}{' + columns + '}', '\\toprule', header + r'\\', '\\midrule']
    body += [' & '.join(map(str, row)) + r'\\' for row in rows]
    body += ['\\bottomrule', '\\end{tabular}']
    style.write_table(name, '\n'.join(body))


def main():
    n = pd.read_csv(paths.REAL / 'null_summary.csv')
    rows = []
    for study, label in [('oneil', "O'Neil"), ('almanac', 'ALMANAC')]:
        for col, test in [('p_tstd', r'$T_{\mathrm{SIR}}$'), ('p_scan', 'Region scan'),
                          ('p_companion', r'Block-level $t$' if study == 'oneil' else 'V2Sp'),
                          ('p_fixed_edge', r'Fixed-edge $t$')]:
            g = n[(n.study == study) & (n.column == col)]
            if g.empty:
                continue
            values = []
            for level in [.05, .01, .005, .001]:
                r = g[g.level == level].iloc[0]
                values.append(f'{r.rate:.4f}' + (f' [{r.lo:.4f}, {r.hi:.4f}]' if level > .001 else ''))
            rows.append([label, test] + values)
    table('tabS9_null', '@{}llllll@{}', 'Study & Test & 0.05 & 0.01 & 0.005 & 0.001', rows)
    d = pd.read_csv(paths.REAL / 'repro_continuous.csv')
    rows = []
    comparisons = [("O'Neil batch 0 vs batch 1", 'all', "O'Neil, batch 1 vs batch 2"),
                   ("O'Neil batch 0 vs batch 1", 'edges identical across batches', 'Identical curves'),
                   ("O'Neil batch 0 vs batch 1", 'dose grids within 1%', 'Dose-matched groups'),
                   ('ALMANAC orientation pairs', 'all (identical copies dropped)', 'ALMANAC orientation pairs'),
                   ("O'Neil vs ALMANAC (cross-study)", 'all matches', "O'Neil vs ALMANAC")]
    for comparison, subset, label in comparisons:
        g = d[(d.comparison == comparison) & (d.subset == subset)]
        if g.empty:
            g = d[d.comparison == comparison]
        if g.empty:
            raise ValueError(comparison)
        values = []
        for score in ['activity', 'mean Bliss excess', 'T_std statistic']:
            r = g[g.score == score].iloc[0]
            values.append(f'{r.spearman:.3f} [{r.lo:.3f}, {r.hi:.3f}]')
        rows.append([label, f'{int(g.iloc[0].n):,}'] + values)
    table('tabS10_reproducibility', '@{}lrlll@{}', r'Comparison & Groups & Activity & Mean excess & $T_{\mathrm{SIR}}$', rows)
    c = pd.read_csv(paths.REAL / 'repro_calls.csv')
    rows = []
    almanac = next(x for x in comparisons if x[0].startswith('ALMANAC'))
    for comparison, _, label in [comparisons[0], almanac]:
        g = c[(c.comparison == comparison) & ~c.subset.str.contains('identical across') & ~c.subset.str.startswith('dose grids')]
        selected = g[g.call.str.contains('T_std|scan:', regex=True)]
        if comparison.startswith('ALMANAC'):
            selected = selected[~selected.call.str.contains('BH')]
        for r in selected.itertuples():
            test = 'SIR BH' if 'BH' in r.call else 'Region scan' if 'scan' in r.call else 'SIR, level 0.05'
            rows.append([label + ', ' + test, int(r.N), int(r.n_first), int(r.n_second), int(r.n_both), f'{r.expected_both:.1f}', f'{r.kappa:.2f}'])
    table('tabS10_calls', '@{}lrrrrrr@{}', 'Comparison & Groups & First & Second & Both & Expected & Kappa', rows)
    w = json.loads((paths.REAL / 'window_summary.json').read_text())
    def pair(key):
        return [f'{w[s][key]:,}' for s in ['oneil', 'almanac']]
    rows = [['Drugs with a concentration value'] +
            [f"{w[s]['drugs_with_cmax_present']} of {w[s]['drugs_in_cmax_table']}" for s in ['oneil','almanac']],
            ['Units with both drugs in the table'] + pair('units_both_drugs_with_cmax'),
            ['Untestable after restriction'] + pair('units_untestable_after_restriction'),
            ['Testable (keeping every dose)'] + [f"{w[s]['units_testable']:,} ({w[s]['units_window_keeps_all_doses']:,})" for s in ['oneil', 'almanac']],
            ['SIR BH: full / window / both', ' / '.join(f"{w['oneil']['tstd_bh_calls_'+k]:,}" for k in ['full','window','both']), 'not used'],
            ['SIR at level 0.05: full / window'] + [' / '.join(f"{w[s]['tstd_nominal_'+k]:,}" for k in ['full','window']) for s in ['oneil','almanac']],
            ['Companion BH: full / window / both', ' / '.join(f"{w['oneil']['companion_bh_calls_'+k]:,}" for k in ['full','window','both']), 'not used'],
            ['Mean excess, Spearman correlation'] + [f"{w[s]['spearman_mean_excess_full_vs_window']:.3f}" for s in ['oneil','almanac']]]
    table('tabS11_window', '@{}lll@{}', "Quantity & O'Neil & ALMANAC", rows)


if __name__ == '__main__':
    main()
