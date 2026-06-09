# SIR: Synergy via Isotonic Regression

A nonparametric framework for drug combination synergy testing using shape-constrained regression and wild bootstrap inference.

SIR defines interaction as the deviation from a monotone-additive null within a shared monotone model class, fit by two-dimensional isotonic regression. A degrees-of-freedom-corrected wild bootstrap yields calibrated p-values for each dose-response matrix, enabling principled hit calling and error-rate control in large screens.

**Paper:** A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing.
Asiaee A, Long JP, Pal S, Pua HH, Coombes KR. *bioRxiv* (2026). [doi:10.1101/2026.02.05.704019](https://doi.org/10.1101/2026.02.05.704019)

## Installation as an R package

To install the methods exposed by this repository as a regular R package
(the recommended path for downstream users):

```r
# install.packages("devtools")
devtools::install_github("AsiaeeLab/isosynergy")
```

Once installed, the main entry point is `sir_test()`:

```r
library(SIR)
set.seed(1)
Z <- matrix(rnorm(36), 6, 6)
fit <- sir_test(Z, B = 100, direction = "decreasing")
fit$p_value
plot(fit)
```

A worked walk-through is in the `getting-started` vignette
(`vignette("getting-started", package = "SIR")`). For the full pipeline
that reproduces the figures in the paper, follow the steps below.

## Reproducing paper results: development install

SIR requires R >= 4.1. Clone the repository and restore the package environment:

```bash
git clone https://github.com/AsiaeeLab/isosynergy.git
cd SIR
```

In R:
```r
install.packages("renv")
renv::restore()
```

## Quick start

Run the demo analysis on a small example dataset:

```r
source("R/utils.R")
source("R/config.R")
source("R/transforms.R")
source("R/isotonic_2d.R")
source("R/additive_ordered.R")
source("R/interaction.R")
source("R/bootstrap.R")
source("R/metrics.R")

cfg <- read_config("configs/smoke.yaml")
# See experiments/ notebooks for worked examples
```

## Reproducing paper results

### 1. Download data

**DrugCombDB** (required for main results):
- Download `drugcombs_response.csv` and `drugcombs_scored.csv` from http://drugcombdb.denglab.org/
- Place in `data/raw/drugcombdb/`
- See `data/raw/drugcombdb/README.md` for checksums

**NCI-ALMANAC** (required for supplementary replication):
- Download `ComboDrugGrowth_Nov2017.csv` from https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC
- Place in `data/raw/nci_almanac/`
- See `data/raw/nci_almanac/README.md` for checksums

### 2. Prepare data

```bash
Rscript scripts/prepare_drugcombdb.R
Rscript scripts/prepare_public_data.R
```

### 3. Run analyses and generate figures

```bash
# Figure 1: Baseline disagreement
Rscript scripts/baseline_disagreement.R --config configs/default.yaml

# Figure 2: Method overview
Rscript scripts/method_overview_figure.R --config configs/default.yaml

# Figure 3: Simulation calibration and power
Rscript scripts/simulation_calibration_power.R --config configs/default.yaml

# Figure 4: Pseudo-null calibration
Rscript scripts/pseudonull_calibration.R --config configs/default.yaml

# Figures 5-6: Replicate concordance and missingness prediction
Rscript scripts/drugcombdb_realdata_figures.R --config configs/default.yaml --seed 1

# Supplementary: NCI-ALMANAC replication
Rscript scripts/nci_almanac_baseline_disagreement.R --config configs/default.yaml
Rscript scripts/nci_almanac_pseudonull_calibration.R --config configs/default.yaml
```

### 4. Compile manuscript

```bash
cd doc
pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
pdflatex supplement.tex && bibtex supplement && pdflatex supplement.tex && pdflatex supplement.tex
```

## Repository structure

```
R/                  Source code (isotonic regression, bootstrap, baselines, visualization)
scripts/            Analysis scripts for all figures and tables
experiments/        R Markdown notebooks documenting methodology development
configs/            YAML configuration files
data/raw/           Raw data (download separately; see above)
doc/                Manuscript, supplement, and figures
tests/              Unit tests
```

## Configuration

All pipeline parameters are specified in `configs/default.yaml`:
- `transform.name`: Response transform (logit, identity, asinh)
- `transform.eps`: Clamping constant for logit (default: 1e-6)
- `bootstrap.B`: Number of bootstrap resamples (default: 200)
- `weights.tau`: Variance floor for inverse-variance weights (default: 1e-6)

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{Asiaee2026SIR,
  title   = {A shape-constrained regression and wild bootstrap framework
             for reproducible drug synergy testing},
  author  = {Asiaee, Amir and Long, James P. and Pal, Samhita and
             Pua, Heather H. and Coombes, Kevin R.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2026.02.05.704019}
}
```
