"""Confidence-band and reference-size summaries for Supplement S7 and S8."""
import json
import paths
import pandas as pd

search = pd.read_parquet(paths.INPUTS / 'bb_search_check.parquet')
gain = (search.c_ext - search.c_default) / (search.B + 1)
summary = {'band_search': {'datasets': len(search),
                          'max_p_change': float(gain.max())}}

order = pd.read_csv(paths.INPUTS / 'band_order_summary.csv')
columns = [name for name in order if name.startswith('n_viol_')]
violations = order.groupby('noise')[columns].sum()
summary['band_order'] = {
    noise: {name: int(value) for name, value in row.items()}
    for noise, row in violations.iterrows()
}

thresholds = pd.read_csv(paths.SIMS / 'resampling_effect80.csv')
summary['reference_size'] = {}
for level in (0.05, 0.005):
    selected = thresholds[(thresholds.test == 'tstd') & (thresholds.alpha == level)]
    paired = selected.pivot(index=['design', 'I', 'surface', 'noise', 'pattern'],
                            columns='B', values='mean_eta80').dropna()
    if len(paired):
        excess = 100 * (paired[399] / paired[1999] - 1)
        reduction = 100 * (1 - paired[1999] / paired[399])
        summary['reference_size'][str(level)] = {
            'paired_settings': len(paired),
            'median_excess_399_relative_to_1999_percent': float(excess.median()),
            'median_reduction_1999_relative_to_399_percent': float(reduction.median())
        }
    else:
        summary['reference_size'][str(level)] = {'paired_settings': 0}

(paths.SIMS / 'supplement_numbers.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
