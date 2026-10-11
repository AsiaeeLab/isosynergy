# SIR: calibrated testing of Bliss synergy in dose-response matrices

SIR tests a drug-combination viability matrix against Bliss independence. It fits
nonincreasing curves to the two single agents and a two-dimensional isotonic surface
to the matrix, measures excess killing as the Bliss prediction minus the fitted
surface, and tests it with the SIR statistic (the standardized positive Bliss excess,
returned as `t_std`), calibrated by a plug-in parametric bootstrap at exact Bliss.
Each fit also reports a companion test of the mean Bliss excess, a region scan that
localizes synergy with familywise error control, activity (mean killing), and
fragility flags for single plates. Screens adjust p-values within declared families.

The R package wraps the bundled Python implementation (`inst/python/sirsyn`), so R
and Python give identical results.

## Install

R (>= 4.1) and Python (>= 3.9) with numpy, scipy and numba are required.

```sh
python3 -m venv ~/.venvs/sir
~/.venvs/sir/bin/python -m pip install 'numpy>=1.22' 'scipy>=1.8' 'numba>=0.56'
export RETICULATE_PYTHON="$HOME/.venvs/sir/bin/python"
```

```r
install.packages(c("reticulate", "Matrix", "osqp", "remotes"))
remotes::install_github("AsiaeeLab/isosynergy")
```

`RETICULATE_PYTHON` selects the interpreter; before Python starts,
`sir_python_config("/path/to/python")` does the same. `sir_python_config()` reports
the configuration and `sir_python_install_help()` prints setup instructions. The
bundled Python code needs no separate installation, and loading SIR does not start
Python.

## One matrix

```r
library(SIR)
set.seed(18)
a <- c(1, 0.9, 0.7, 0.5, 0.3)
Y <- outer(a, a) + matrix(rnorm(25, sd = 0.04), 5)
Y[1, 1] <- 1
Y[3:5, 3:5] <- Y[3:5, 3:5] - 0.15          # synergy in the high-dose block
noise <- sir_noise_parametric(sd = 0.04)
fit <- sir_fit(Y, doses = list(0:4, 0:4), design = "S", noise = noise, seed = 7)
summary(fit)
plot(fit)
```

`Y` is viability (each plate divided by its control), with the zero-dose row and
column holding the single agents. A replicate stack has dimensions
`c(replicates, rows, columns)`. Missing wells are `NA`; values are not clipped to
[0, 1]. The noise model should describe the assay; `sir_sd_from_replicates()` and
`sir_noise_bank()` build one from replicate plates.

Main outputs:

| Field | Meaning |
| --- | --- |
| `t_std`, `p_tstd` | SIR statistic and its p-value (null: no synergy in any combination cell) |
| `mean_excess`, `mean_excess_se`, `p_mean` | mean Bliss excess, its standard error and the companion p-value (null: no synergy on average) |
| `activity` | mean killing, 1 - viability, over the combination wells |
| `claims`, `p_scan` | regions claimed to contain at least one synergistic cell, and the scan p-value |
| `excess_map` | isotonic excess per cell (descriptive) |
| `fragility`, `log` | single-plate fragility checks and the data-handling log |

## Designs

| Design | Single agents | Companion test |
| --- | --- | --- |
| S | measured on the same plate | V2Sp on a complete grid |
| R | measured on each replicate plate | block-level t |
| P | external curves in the first row and column | fixed-edge t, or block-level t for replicate stacks |

S with replicate plates is analyzed as R. In design P, `sa_replicates = Inf` tests
relative to the reported curves, holding them fixed; a finite `sa_replicates`
propagates curve uncertainty into the reference. The default Monte Carlo budget is
399 reference plates, followed by 9,999 fresh plates when the primary or scan
p-value is at most 0.05.

## Screens

```r
panel <- data.frame(id = c("pair1", "pair2"), family = c("cellA", "cellA"))
panel$Y <- list(Y, outer(a, a))
screen <- sir_screen(panel, noise = noise, design = "S", seed = 7)
screen[c("id", "family", "p", "q", "call")]
```

Benjamini-Hochberg adjustment is applied within each declared family (for example a
cell line). Calls from a single experiment need independent confirmation.

## Scope and limits

- Exact Bliss is the least favorable null for "no synergy in any cell" when the true
  single-agent curves and noise law are used; the plug-in bootstrap substitutes
  estimates, so its p-values are approximate and their calibration was checked by
  simulation and on real replicate noise.
- That result covers replicate means and other replicate summaries that are
  nondecreasing in each replicate. The replicate screen (drop a replicate more than
  5 model SDs from the median of the others, applied identically to the reference)
  is outside this class; its effect on the level was evaluated by simulation.
- Missing combination wells are excluded; a missing single-agent dose drops its row
  or column. On an incomplete single plate the companion p-value comes from the
  exact-Bliss bootstrap and is exploratory.
- For replicate stacks the block-level companion treats a shared single-agent
  reference as fixed, and `mean_excess` is the average over plates of each plate's
  mean excess.
- Region-scan calibration was evaluated on complete grids with parametric noise in
  designs S and P; plate shifts were evaluated in design P.

Fits are ordinary R lists and can be saved with `saveRDS()`. Region, fragility and log
indices use zero-based positions on the original grid; add one to index R inputs. The
standalone scorers `bliss_score()`, `hsa_score()`, `loewe_score()`, `zip_score()` and
`isotonic_2d_fit()` are also exported.

## Tests and runtime

```sh
cd inst/python
python3 -m pytest -q           # fast invariants
python3 -m pytest -m slow -q   # Monte Carlo calibration checks
```

The R tests compare complete outputs with direct Python calls under fixed seeds. On one
core a 5 by 5 fit (SIR statistic, scan and companion test) takes about 3 ms at
B = 399 in Python and 5 ms through R; `inst/benchmarks/` has the scripts, hardware
record and timings by grid size. The first call in a session compiles the numerical
kernels (a few seconds).

## Reproducing the article

`reproduce/` regenerates every number, table and figure of the accompanying article
from the public DrugCombDB and NCI-ALMANAC downloads, in numbered stages:

```sh
bash reproduce/run_all.sh --smoke --workers 16   # reduced end-to-end run, about 10 minutes
bash reproduce/run_all.sh --workers 16           # full run
```

The last stage recomputes the reported numbers from the released per-unit table and
the stage outputs and exits with an error on any mismatch. `reproduce/README.md` lists
requirements, stage times and the map from paper items to scripts. The per-unit
DrugCombDB results (`dataset_release/`) are deposited on Zenodo.

## Citation

See `CITATION.cff`. Released versions are archived on Zenodo (concept DOI
[10.5281/zenodo.20604671](https://doi.org/10.5281/zenodo.20604671)).

## License

MIT (see `LICENSE`).
