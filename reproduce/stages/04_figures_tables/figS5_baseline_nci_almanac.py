"""Supplementary S5_baseline_nci_almanac calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import fig1_baseline_disagreement as F1
SCORES = os.path.join(str(mc.paths.INPUTS), 'nci_almanac_baseline_scores.parquet')
SUMMARY_R = os.path.join(str(mc.paths.INPUTS), 'nci_almanac_baseline_disagreement.parquet')
OUT_CSV = os.path.join(ST.RESDIR, 'baseline_disagreement_nci_almanac.csv')
SUPP_WIDTH = 6.45

def main():
    df = pd.read_parquet(SCORES)
    X = F1.complete(df)
    n = len(X)
    print(f'sampled experiments {len(df):,}; Loewe missing {int(df.Loewe.isna().sum()):,}; all four scores finite: {n:,}')
    (res, thr) = F1.metrics(X)
    worst = F1.check_against_r(res, SUMMARY_R)
    print(f'reproduces {os.path.relpath(SUMMARY_R, ST.REPO)} (max |diff| {worst:.2g})')
    print('Loewe range', float(X.Loewe.min()), float(X.Loewe.max()), '; |Loewe| >= 99:', int((X.Loewe.abs() >= 99).sum()))
    F1.long_table(res, n, 'NCI-ALMANAC sample').to_csv(OUT_CSV, index=False)
    for (name, M) in res.items():
        print(name)
        print(M.round(4).to_string())
    lo = [res['pos_disagree_rate'].loc['Loewe', b] for b in ('Bliss', 'HSA', 'ZIP')]
    jl = [res['top_jaccard'].loc['Loewe', b] for b in ('Bliss', 'HSA', 'ZIP')]
    print(f'Loewe sign disagreement {min(lo):.4f} to {max(lo):.4f}; Loewe Jaccard {min(jl):.4f} to {max(jl):.4f}')
    F1.figure(res, 'figS5_baseline_nci_almanac', width=SUPP_WIDTH)
if __name__ == '__main__':
    main()
