"""Pairwise score agreement for Figure 1 and Supplementary Figure S5."""
import paths
import pandas as pd
import fig1_baseline_disagreement as F
for dataset in ('drugcombdb', 'nci_almanac'):
    name = 'drugcombdb_block_scores' if dataset == 'drugcombdb' else 'nci_almanac_baseline_scores'
    data = F.complete(pd.read_parquet(paths.INPUTS / (name + '.parquet')))
    values, thresholds = F.metrics(data)
    F.long_table(values, len(data), dataset).to_parquet(paths.INPUTS / (dataset + '_baseline_disagreement.parquet'), index=False)
