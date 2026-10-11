# SIR per-unit synergy test results for DrugCombDB (O'Neil and NCI-ALMANAC)

Version 2.0.0. One row per analysis unit for the two largest dose-matrix studies in DrugCombDB:
23,052 O'Neil units and 227,323 NCI-ALMANAC plates (250,375 rows, 61 columns). Each row reports
the SIR analysis of that unit against Bliss independence on the viability scale: activity, the mean
Bliss excess with its standard error and companion p-value, the SIR statistic and its p-value, q-values
within the declared testing family, the region-scan result, fragility diagnostics for single plates,
the data-handling log and the random seed. For O'Neil, `mean_excess` is the
estimate of the block-level companion test (the average over the four plates of
each plate's mean excess); it equals the cell average of the replicate means,
which enters the identity activity = Bliss-expected activity + mean excess, except
in units where the replicate screen removed values (`n_screened` > 0).

The table contains no binary synergy labels. Use the p- and q-values within the declared family
(`family`), and read the tested null (`null_tested`) before comparing the two studies.

## Files

| File | Content |
| --- | --- |
| `sir_drugcombdb_unit_results.parquet` | the table (Apache Parquet) |
| `sir_drugcombdb_unit_results.csv` | the same table as CSV |
| `schema.json` | column names, types and descriptions; row counts; tested nulls |
| `CITATION.cff` | citation metadata |
| `checksums.txt` | SHA-256 of every other file |

## Method in brief

Viability `Y` (each plate divided by its control) is analyzed on its dose grid, with the zero-dose
row and column holding the single agents. The reference is Bliss independence,
`E Y_ij = E Y_i0 E Y_0j`. SIR fits nonincreasing isotonic curves `a_hat`, `b_hat` to the two single
agents and a 2D isotonic surface `mu_hat` to the matrix. The SIR statistic sums the squared positive
parts of the standardized excess `(a_hat_i b_hat_j - mu_hat_ij) / s(a_hat_i b_hat_j)` over the
combination cells; its p-value comes from a plug-in parametric bootstrap at exact Bliss (B = 399, then
9,999 fresh reference plates when a stage-1 p-value is at most 0.05). The companion test asks whether
the mean Bliss excess on raw wells is positive. The region scan localizes synergy with familywise
error control over high-dose rectangles and 2 x 2 blocks.

## The two studies

| | O'Neil | NCI-ALMANAC |
| --- | --- | --- |
| Unit | drug pair x cell line x batch, four replicate 5 x 5 blocks | one 4 x 4 plate |
| Design | P: reported single-agent curves held fixed | S: single agents measured on the same plate |
| Null of the SIR statistic | no synergy in any combination cell relative to the reported curves | no synergy in any combination cell |
| Companion test | block-level t (3 df) | V2Sp |
| Reference noise | residual bank from the replicate blocks | plate-difference bank from orientation pairs |
| Testing family | study x cell line x batch (54 families) | study x cell line (60 families) |

Both noise banks are cross-fitted (built from cell lines other than the target's) and sampled within
potency strata. ALMANAC p-values are limited by the resolution of the plate-difference bank, and its
lower tail was miscalibrated on real noise; `p_tstd_floored` and `q_tstd_floored` apply the resolution
floor, and level-0.05 rates rather than BH lists are the appropriate summary for single-plate units. Activity and the mean Bliss excess are continuous scores
that do not depend on the bank.

## Provenance

- Input: DrugCombDB `drugcombs_response.csv` (accessed 2026-02-01; SHA-256
  `3f67aab5efc4df3ae60fc95122b6daf5016f6fde40abc1be115a9683ad69bf73`), `Response / 100`, unclipped.
- Software: SIR R package 0.2.0 with its bundled Python implementation
  (https://github.com/AsiaeeLab/isosynergy). The repository's `reproduce/` pipeline rebuilds this table
  from the raw file and checks it against these files.
- Seeds: every row stores the seed entropy of its fit; a rerun reproduces the row exactly.

## License

CC-BY-4.0. DrugCombDB and NCI-ALMANAC remain subject to their own terms.

## Citation

Please cite this record (DOI [10.5281/zenodo.23302266](https://doi.org/10.5281/zenodo.23302266)) and the
SIR method paper (see `CITATION.cff`).
