"""tabS3_power calculations."""
import mc
import style as ST
import h2h_lib as L
ROWS = [('syn_broad_0.20', 'Broad 0.20'), ('syn_hdblock_0.80', 'High-dose block 0.80'), ('syn_corner_1.00', 'Top corner 1.00'), ('syn_bump_0.30', 'Middle bump 0.30'), ('syn_lowdose_0.10', 'Low dose 0.10'), ('sat_0.60', 'Saturating 0.60'), ('mixed_0.60', 'Zero net 0.60')]

def main():
    df = L.load('power')
    R = {d: L.rates(df[df.design == d], ['cond']).set_index('cond') for d in ('S', 'P')}
    lines = ['\\begin{tabular}{@{}lrrrrrr@{}}', '\\toprule', ' & \\multicolumn{3}{c}{Design S} & \\multicolumn{3}{c}{Design P}\\\\', '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}', 'Pattern & $T_{\\mathrm{SIR}}$ & Scan & Mean & $T_{\\mathrm{SIR}}$ & Scan & Mean\\\\', '\\midrule']
    for (c, lab) in ROWS:
        (s, p) = (R['S'].loc[c], R['P'].loc[c])
        em = abs(s.eff_mean) if abs(s.eff_mean) < 0.0005 else s.eff_mean
        lines.append(f"{lab} ({em:.3f}, {s.eff_max:.3f}) & {s.tstd:.3f} & {s.scan:.3f} & {s['mean']:.3f} & {p.tstd:.3f} & {p.scan:.3f} & {p['mean']:.3f}\\\\")
        print(lines[-1], 'n =', int(s.n), int(p.n))
    lines += ['\\bottomrule', '\\end{tabular}']
    ST.write_table('tabS3_power', '\n'.join(lines))
if __name__ == '__main__':
    main()
