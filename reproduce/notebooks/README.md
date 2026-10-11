# Notebooks

Six R Markdown notebooks explain the method and redraw every figure of the accompanying
article from results the pipeline has already computed. Nothing is recomputed: each
notebook loads results from `../output/` when the pipeline has been run, and otherwise
from the small copies in `data/` (about 13 MB in total), so the notebooks run from a fresh
clone. The rendered `.html` files can be read without running anything.

| Notebook | Contents | Paper items |
| --- | --- | --- |
| `01_method_walkthrough` | Bliss independence, isotonic fits, the excess map, the SIR statistic and its calibration, the companion test, the region scan and BH, worked on one O'Neil matrix; how to use `sir_fit()` and `sir_screen()` | Figure 2 |
| `02_baseline_disagreement` | Bliss, HSA, Loewe and ZIP scores and how much they disagree | Figures 1, S4, S5 |
| `03_calibration_and_power` | level and power in simulation, what the reference must contain, grid-size guidance, the Berger-Boos sensitivity analysis | Figures 3, 4, S1, S2; Tables S1-S4, S7 |
| `04_comparators_bigl` | BIGL, SynergyFinder Plus and the Demidenko-Miller test | Figure 5; Table S5 |
| `05_missing_wells_and_outliers` | missing wells, the replicate screen, gross errors, fragility checks, departures from monotonicity | Figure 6; Table S6 |
| `06_drugcombdb_application` | the O'Neil and NCI-ALMANAC analyses: scores, calls, reproducibility, controls, clinical window | Figures 7, S3, S6; Tables S8-S11 |

To render, install the SIR package (see the repository README) and the R packages
rmarkdown, knitr, ggplot2, patchwork, kableExtra, data.table, dplyr, tidyr, scales, jsonlite and reticulate, then
from this folder:

```sh
Rscript -e 'rmarkdown::render("01_method_walkthrough.Rmd")'
```

Each notebook takes well under a minute. `06_drugcombdb_application` reads the released
per-unit table from `../../dataset_release/` or, if it is absent, downloads it once from
Zenodo (10.5281/zenodo.23302266). After rerunning the pipeline, `data/make_aggregates.py`
rebuilds the count tables in `data/aggregates/` from the full simulation output.
