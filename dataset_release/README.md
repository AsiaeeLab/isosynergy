# SIR-derived DrugCombDB synergy calls

This release contains matrix-level SIR synergy calls for 319,533 DrugCombDB dose-response matrices for which complete grid metadata was available in the SIR processed pipeline. SIR (Synergy via Isotonic Regression) fits a monotone-additive null and a fully monotone isotonic response surface, summarizes their departure by interaction energy S^2, and assigns calibrated p-values using a degrees-of-freedom-corrected wild bootstrap with B = 200 resamples.

The larger DrugCombDB baseline score table used in the paper contains 391,653 baseline-only rows. The Zenodo SIR release is intentionally limited to the 319,533 matrices present in `data/processed/drugcomb_matrices.parquet`, because that is the universe with complete SIR-ready grid metadata.

## Files

- `sir_drugcombdb_synergy_calls.parquet`: main release table, one row per SIR-analyzed DrugCombDB matrix.
- `schema.json`: JSON Schema row definition plus `x-parquet-*` metadata for column order and physical table checks.
- `CITATION.cff`: citation metadata for the Zenodo record.
- `checksums.txt`: SHA-256 checksums for every file in this folder except `checksums.txt`.

An interaction-surface companion file is not included in this release. The existing pipeline did not persist per-cell `theta_iso`, `theta_add`, or `delta` surfaces for the full universe; producing that file would require an additional large compute/storage pass and is deferred to a future release.

`sir_mean_delta` is intentionally omitted from the release table. The weighted mean interaction is zero by construction for the translation-invariant isotonic projection; see the Supplementary Information of the SIR paper for details. Keeping it in the public schema would add a numerically tiny column with no practical signal.

## Provenance

- Method paper: Asiaee, Long, Pal, Pua, and Coombes, "A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing", bioRxiv doi:10.1101/2026.02.05.704019.
- Code repository: https://github.com/amir-as/synergy
- Code commit used for this assembly: `9dd17a21ce20ad2616434b2c02b93ccf90784f74`.
- SIR version: `0.1.0`.
- DrugCombDB snapshot: DrugCombDB download accessed 2026-02-01.
- Analysis date: 2026-05-08.
- Bootstrap resamples per matrix: 200.
- DrugCombDB-native drug IDs and tissue annotations are not included because they were not available in the processed SIR cache. A future release may join these fields from a raw DrugCombDB drug dictionary or metadata table.
- Two NIH matrices in the processed cache lacked cell-line metadata; their `cell_line` value is encoded as `not_available` rather than null so `cell_line` remains a required identifier column.

## Schema

| column | type | nullable | description |
| --- | --- | --- | --- |
| `matrix_id` | string | no | Stable SIR matrix identifier; equals `experiment_id` from the processed DrugCombDB cache. |
| `drug_a_name` | string | no | Drug row/name as represented in the processed DrugCombDB cache. |
| `drug_b_name` | string | no | Drug column/name as represented in the processed DrugCombDB cache. |
| `cell_line` | string | no | Cell line name from DrugCombDB; `not_available` marks the two NIH matrices with missing processed-cache cell-line metadata. |
| `source_study` | string | yes | DrugCombDB source tag, such as `ONEIL`; may be missing if unavailable upstream. |
| `n_doses_a` | integer | no | Number of unique dose levels for drug A. |
| `n_doses_b` | integer | no | Number of unique dose levels for drug B. |
| `n_replicates_total` | integer | no | Number of dose-response measurements in the processed matrix. |
| `sir_S2` | number | no | SIR interaction energy, the weighted squared departure between isotonic and monotone-additive fits. |
| `sir_p_value` | number | no | Wild-bootstrap p-value for global interaction; B = 200. |
| `sir_q_value` | number | no | Benjamini-Hochberg FDR-adjusted q-value over all 319,533 matrices. |
| `sir_hit_call` | string | no | `synergy`, `antagonism`, or `no_call`; calls require `sir_q_value <= 0.05` and use the dominant signed interaction energy direction. |
| `sir_failed` | boolean | no | TRUE if the SIR fit failed. This release should contain only FALSE values. |
| `bliss_score` | number | yes | DrugCombDB Bliss score, if available. |
| `bliss_failed` | boolean | no | TRUE when a DrugCombDB Bliss row was present but the score was non-finite. |
| `bliss_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |
| `hsa_score` | number | yes | DrugCombDB HSA score, if available. |
| `hsa_failed` | boolean | no | TRUE when a DrugCombDB HSA row was present but the score was non-finite. |
| `hsa_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |
| `loewe_score` | number | yes | DrugCombDB Loewe score, if available. |
| `loewe_failed` | boolean | no | TRUE when a DrugCombDB Loewe row was present but the score was non-finite. |
| `loewe_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |
| `zip_score` | number | yes | DrugCombDB ZIP score, if available. |
| `zip_failed` | boolean | no | TRUE when a DrugCombDB ZIP row was present but the score was non-finite. |
| `zip_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |
| `sir_version` | string | no | SIR release/software version string. |
| `drugcombdb_version` | string | no | DrugCombDB snapshot provenance string. |
| `analysis_date` | string | no | ISO-8601 date for the completed SIR universe checkpoint. |
| `bootstrap_B` | integer | no | Number of wild-bootstrap resamples. |

## Usage

R:

```r
library(arrow)
sir <- read_parquet("sir_drugcombdb_synergy_calls.parquet")
table(sir$sir_hit_call)
```

Python:

```python
import pandas as pd
sir = pd.read_parquet("sir_drugcombdb_synergy_calls.parquet", engine="pyarrow")
sir["sir_hit_call"].value_counts()
```

## License

This derived SIR label dataset is released under CC-BY 4.0. Original DrugCombDB data and constituent studies remain subject to their upstream licenses and terms; please cite and respect those sources when using this release.

## Citation

Zenodo DOI placeholder: `10.5281/zenodo.TBD`.

```bibtex
@dataset{asiaee_sir_drugcombdb_2026,
  title     = {SIR-derived synergy calls for DrugCombDB},
  author    = {Asiaee, Amir and Long, James P. and Pal, Samhita and Pua, Heather H. and Coombes, Kevin R.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.TBD},
  url       = {https://doi.org/10.5281/zenodo.TBD}
}

@article{asiaee_sir_2026,
  title   = {A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing},
  author  = {Asiaee, Amir and Long, James P. and Pal, Samhita and Pua, Heather H. and Coombes, Kevin R.},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.1101/2026.02.05.704019}
}

@article{liu_drugcombdb_2020,
  title   = {DrugCombDB: a comprehensive database of drug combinations toward the discovery of combinatorial therapy},
  author  = {Liu, Hui and Zhang, Wenhao and Zou, Bo and Wang, Jinxian and Deng, Yuanyuan and Deng, Lei},
  journal = {Nucleic Acids Research},
  year    = {2020},
  volume  = {48},
  number  = {D1},
  pages   = {D871--D881},
  doi     = {10.1093/nar/gkz1007}
}
```

## Upstream Sources

The processed DrugCombDB cache includes constituent source tags such as O'Neil and other DrugCombDB-provided studies. Users should cite DrugCombDB (Liu et al., Nucleic Acids Research 2020, doi:10.1093/nar/gkz1007) and the relevant original screens when focusing on a specific source study.
