# Reproduce the SIR paper

This directory runs the numerical analyses, Monte Carlo studies, figures, and table bodies for SIR 0.2.0. It imports the Python implementation bundled in `inst/python/sirsyn`. The result fields `t_std` and `p_tstd` contain the SIR statistic and its p-value.

Run from the repository root:

```sh
bash reproduce/run_all.sh --smoke --workers 16
bash reproduce/run_all.sh --workers 16
bash reproduce/run_all.sh 03 --workers 16
bash reproduce/run_all.sh 03 --item grid --workers 8
bash reproduce/run_all.sh 04 05 --real-only
```

The default is 16 worker processes, configurable with `--workers` or `SIR_WORKERS` (1 through 16). Stages and study groups run sequentially. OpenMP, OpenBLAS, MKL, BLIS, and Numba each use one thread. Individual stage commands require the outputs of earlier stages. A failed command stops the driver and prints the end of its log.

A Unix-like system with Bash is required; the R calculations use forked workers. Python 3.10.12 and the exact package versions in `requirements.txt` were used. Install them in a Python environment before running the driver. The NCI baseline-score calculation additionally needs R, `data.table`, and `arrow`. The BIGL comparison requires BIGL **1.9.3** and its R dependencies. The R interface benchmark also needs `reticulate`. R 4.6.1 was used. Dependencies are never installed by the driver; an existing R library can be selected with `R_LIBS_USER`.

Place the five public raw files in `data/raw/drugcombdb/` and `data/raw/nci_almanac/`, as described in [data/README.md](data/README.md). Stage 00 verifies their SHA-256 values. The two DrugCombDB CSVs contain the dose responses and the scored-block metadata. All matrices, replicate groups, reference banks, panels, and simulation datasets are rebuilt from these inputs. The small literature and concentration tables are included in `data/`.

| Stage | Contents | Worker limit | Wall time |
| --- | --- | ---: | --- |
| 00 | Verify the raw files | 1 | 1.1 s |
| 01 | Dose grids, replicate banks, orientation pairs, donors, panels, comparison datasets, NCI baseline scores | 16 | 2.6 min |
| 02 | Screens, null checks, family BH, release, reproducibility, controls, concentration window, missing-well prediction, counts | 16 | 11.2 min |
| 03 | Monte Carlo studies and comparator calculations | 16 | 4.0 h |
| 04 | Figures 1–7 and S1–S6; Tables S1–S11 | 1 | 83.0 s |
| 05 | Release structure, checksums, and 28,937 numerical comparisons | 1 | 26.2 s |

Wall times are observations on an Intel Xeon w9-3575X with the worker limits shown. The O'Neil screen took 227.2 s and the ALMANAC screen 242.6 s. The driver records every command, worker limit, exit status, and wall time in `runs.jsonl`.

Stage 03 item times, including collection steps and excluding figure rendering:

| Item | Wall time | Worker limit |
| --- | ---: | ---: |
| `bigl` | 34.5 s | 16 |
| `grid` | 27.6 min | 16 |
| `missing` | 11.9 min | 16 |
| `head-to-head` | 25.7 min | 16 |
| `references` | 19.4 min | 16 |
| `bands` | 87.0 min | 16 |
| `resampling` | 7.6 min | 16 |
| `runtime` | 23.3 s | 1 |
| `companions` | 70.4 s | 16 |
| `null-search` | 6.9 min | 16 |
| `band-search` | 4.2 min | 16 |
| `band-order` | 1.8 min | 16 |
| `propositions` | 5.7 min | 16 |
| `bigl-r` | 40.3 min | 16 |

Outputs are written exclusively under `reproduce/output/`. Smoke outputs are isolated under `reproduce/output/smoke/`. The smoke run retains the real input populations, reduces reference draws and simulation sample counts, and exercises every analysis and rendering path. Its final check validates structure and successful execution; it does not compare the reduced calculations with the paper's values.

Full runs use the same seeds, Monte Carlo sample sizes, reference sizes, and family definitions as the paper. Seed components are part of the numerical configuration; their literal byte values must not be changed when renaming a script. A screen uses 399 reference draws and 9,999 fresh second-stage draws, with 200 scan-standardization draws. The simulation scripts state their study-specific settings.

Stage 03 accepts these `--item` values: `bigl`, `grid`, `missing`, `head-to-head`, `references`, `bands`, `bigl-r`, `propositions`, `null-search`, `band-search`, `companions`, `resampling`, `runtime`, and `band-order`. The R BIGL calculation is separable using `--item bigl-r`; `--skip-bigl` omits it when its outputs are already available. Figure 5 and Table S5 require those outputs.

The BIGL R code runs both the distributed Bliss reference and a viability-product reference. For the latter, `set_null()` replaces `Blissindependence` in the package namespace with the product of the two marginal viabilities divided by the fitted baseline. Marginal fitting uses `nlslm`; surface inference uses `statistic="both"`, `B.CP=50`, and `B.B=100`. The comparison retains the distributed and product-reference results separately. Single-replicate normal errors and replicated residual errors are selected explicitly; the equal and model variance choices are both evaluated where specified.

The first row of Table S5 reports distributed BIGL's null rejection rates for weak and moderate agents with Gaussian noise (0.154–0.353). Across all six settings shown in Figure 5, that range is 0.021–0.353; all six rates are retained in `bigl_summary.json`. The 25–48 real-data reference failures refer to distributed BIGL with equal and model variance; the viability-product configuration has zero additional reference failures.

For the score-pooling comparison, the empirical noise curve at its measured scale gives a maximum size of 0.06965, printed as 0.070; the maximum across every tested noise scale is 0.0712. The analytic example in Supplement S5 uses the full-precision normal quantile, giving 0.060394 (printed as 0.0604); the rounded critical value 1.644 would give 0.060483. Both analytic calculations and both pooling ranges are emitted in `comparator_numbers.json`.

`output/realdata/release/` contains the 250,375-row, 61-column per-unit table in CSV and Parquet, its schema, and SHA-256 checksums. `output/simulations/` holds Monte Carlo results; `output/figures/` holds PDFs; `output/tables/` holds LaTeX table bodies; `output/logs/` holds per-command logs. Tables S9–S11 are the three tables embedded directly in the supplementary text. There are no numbered main-text tables. See [MANIFEST.md](MANIFEST.md) for the paper-to-output mapping. The [notebooks](notebooks/README.md) explain the method and redraw every figure from these outputs without rerunning anything.

`expected_numbers.csv` gives check IDs, values, tolerances, units, paper locations, and machine-readable calculation recipes. The final stage recomputes quantities from the released table independently, then checks stage outputs. It prints each mismatch with its ID, expected value, and observed value, writes `output/checks.json`, and exits with status 1 if any check fails. The Supplement S8 reference-size comparison reports the median reduction of the required effect relative to the 399-draw threshold (0.6% and 2.3%); `supplement_numbers.json` gives this and the excess relative to the 1,999-draw threshold. Runtime is measured afresh and is not a deterministic numerical expectation. A full run requires all listed numerical studies; smoke completion alone does not establish numerical agreement.

After a full numerical check passes, `output/simulation_cache.json` records source and content hashes for the simulation outputs. `bash reproduce/run_all.sh --reuse-simulations` verifies that record and reruns the data, real-data, rendering, and final-check stages while reusing the checked simulation files. It refuses reuse if any recorded source or output differs. The default driver always executes stage 03.
