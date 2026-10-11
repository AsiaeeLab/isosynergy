"""tabS4_stress calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
from figS1_reference_components import cl_interval
R = mc.RESDIR
MS = [1.0, 4.0, 8.0]

def main():
    d = pd.read_parquet(os.path.join(R, 'stress_shared_null.parquet'))
    f = pd.read_csv(os.path.join(R, 'stress_shared_null_family.csv'), keep_default_na=False, na_values=[''])
    x = pd.read_parquet(os.path.join(R, 'stress_shared_mixed.parquet'))
    fx = pd.read_csv(os.path.join(R, 'stress_shared_mixed_family.csv'), keep_default_na=False, na_values=[''])
    s = pd.read_parquet(os.path.join(R, 'stress_shift.parquet'))
    num = []

    def add(item, col, v, lo=np.nan, hi=np.nan, n=np.nan):
        num.append(dict(item=item, col=col, value=v, lo=lo, hi=hi, n=n))
        return v
    for (test, c) in (('tstd', 'p_t1'), ('scan', 'p_s1'), ('mean', 'p_m')):
        for ref in (bytes([82, 48]).decode('ascii'), bytes([82, 49]).decode('ascii')):
            for m in MS + [np.inf]:
                if not np.isfinite(m) and ref == bytes([82, 49]).decode('ascii'):
                    continue
                g = d[(d.share == 'shared') & (d.m_s == m) & (d.ref == ref)]
                (p, lo, hi) = cl_interval(g[c] <= 0.05, g.screen)
                add(f'per_matrix_{test}_{ref}', m, p, lo, hi, len(g))
                if test == 'tstd':
                    (p01, lo1, hi1) = cl_interval(g[c] <= 0.01, g.screen)
                    add(f'per_matrix_{test}_{ref}_a01', m, p01, lo1, hi1, len(g))
        for ref in (bytes([82, 48]).decode('ascii'), bytes([82, 49]).decode('ascii')):
            for m in MS:
                g = d[(d.share == 'indep') & (d.m_s == m) & (d.ref == ref)]
                (p, lo, hi) = cl_interval(g[c] <= 0.05, g.screen)
                add(f'per_matrix_indep_{test}_{ref}', m, p, lo, hi, len(g))
    for ((m, sh, ref), g) in f.groupby(['m_s', 'share', 'ref']):
        (k, n) = (int(g.any_disc.sum()), len(g))
        (lo, hi) = mc.wilson(k, n)
        add(f'fwer_{sh}_{ref}', m, k / n, float(lo), float(hi), n)
        add(f'n_eval_{sh}_{ref}', m, g.n_eval.mean(), n=n)
    syn = x[x.cls == 'syn']
    for m in MS + [np.inf]:
        g = syn[syn.m_s == m]
        rates = g.groupby('cond').p_t1.apply(lambda p: (p <= 0.05).mean())
        add('power_per_matrix', m, rates.mean(), n=len(g))
        for (cnd, v) in rates.items():
            add(f'power_per_matrix_{cnd}', m, v, n=int((g.cond == cnd).sum()))
        nul = x[(x.cls == 'null') & (x.m_s == m)]
        (p, lo, hi) = cl_interval(nul.p_t1 <= 0.05, nul.screen)
        add('mixed_size_bliss_pairs', m, p, lo, hi, len(nul))
    for m in MS:
        g = fx[fx.m_s == m]
        bp = (g.n_true / g.n_syn).to_numpy()
        add('power_bh', m, bp.mean(), bp.mean() - 1.96 * bp.std(ddof=1) / np.sqrt(len(bp)), bp.mean() + 1.96 * bp.std(ddof=1) / np.sqrt(len(bp)), len(g))
        fdp = (g.n_false / np.maximum(g.n_disc, 1)).to_numpy()
        add('fdr_bh', m, fdp.mean(), n=len(g))
    for m in (np.inf, 4.0):
        for r in (0.0, 0.1, 0.25, 0.5):
            for ref in ('without', 'with'):
                g = s[(s.m_s == m) & (s.ratio == r) & (s.ref == (ref if r > 0 else 'without'))]
                for (test, c) in (('tstd', 'p_t'), ('scan', 'p_s'), ('mean', 'p_m')):
                    (k, n) = (int((g[c] <= 0.05).sum()), len(g))
                    (lo, hi) = mc.wilson(k, n)
                    add(f'shift_{test}_{ref}_ms{m}', r, k / n, float(lo), float(hi), n)
                (k, n) = (int((g.p_t <= 0.01).sum()), len(g))
                (lo, hi) = mc.wilson(k, n)
                add(f'shift_tstd_{ref}_ms{m}_a01', r, k / n, float(lo), float(hi), n)
    N = pd.DataFrame(num)
    N.to_csv(os.path.join(R, 'tabS4_stress_numbers.csv'), index=False)

    def v(item, col):
        return float(N[(N.item == item) & (N.col == col)].value.iloc[0])

    def f3(z):
        from decimal import Decimal, ROUND_HALF_UP
        return str(Decimal(repr(round(float(z), 12))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
    lines = ['\\begin{tabular}{@{}lrrrr@{}}', '\\toprule', ' & $m_s=1$ & $m_s=4$ & $m_s=8$ & exact\\\\', '\\midrule']
    lines.append('Per matrix, curves treated as exact & ' + ' & '.join((f3(v('per_matrix_tstd_R0', m)) for m in MS)) + f" & {f3(v('per_matrix_tstd_R0', np.inf))}" + '\\\\')
    lines.append('Per matrix, curves redrawn & ' + ' & '.join((f3(v('per_matrix_tstd_R1', m)) for m in MS)) + ' & \\\\')
    lines.append('BH FWER, curves treated as exact & ' + ' & '.join((f3(v('fwer_shared_R0', m)) for m in MS)) + f" & {f3(v('fwer_shared_R0', np.inf))}" + '\\\\')
    lines.append('BH FWER, curves redrawn & ' + ' & '.join((f3(v('fwer_shared_R1', m)) for m in MS)) + ' & \\\\')
    lines.append('Power per matrix, curves redrawn & ' + ' & '.join((f3(v('power_per_matrix', m)) for m in MS)) + f" & {f3(v('power_per_matrix', np.inf))}" + '\\\\')
    lines.append('Power after BH, curves redrawn & ' + ' & '.join((f3(v('power_bh', m)) for m in MS)) + ' & \\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{5}{@{}l}{Plate shift, exact curves: $\\tau/\\sigma=$ 0, 0.1, 0.25, 0.5}\\\\')
    lines.append('Shift omitted from the reference & ' + ' & '.join((f3(v('shift_tstd_without_msinf', r)) for r in (0.0, 0.1, 0.25, 0.5))) + '\\\\')
    lines.append('Shift in the reference & ' + ' & '.join((f3(v('shift_tstd_with_msinf' if r > 0 else 'shift_tstd_without_msinf', r)) for r in (0.0, 0.1, 0.25, 0.5))) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    ST.write_table('tabS4_stress', '\n'.join(lines))
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 400)
    print(N.round(4).to_string())
if __name__ == '__main__':
    main()
