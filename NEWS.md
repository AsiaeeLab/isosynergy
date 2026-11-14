# SIR 0.2.1

* `sir_fit()` and `sir_screen()` now warn in design P when the reported
  single-agent curves are treated as exact because `sa_replicates` was not
  given, and when a parametric reference has no plate shift. In both cases an
  offset between the combination plates and the curves reads as synergy. Pass
  `sa_replicates = Inf` to accept exact curves without the warning.
* Added `sir_check_noise()` (Python `check_sd_curve()`), which compares the
  replicate spread of wells with the SD that a curve predicts, by bins of mean
  viability.
* `reproduce/`: added the benchmark of the isotonic statistic against raw-well
  Bliss-excess statistics under the same reference
  (`stages/03_simulations/sim_raw_benchmark.py`), the level of the O'Neil
  configuration under an offset between combination plates and reported curves
  (`sim_offset_oneil.py`), the batch offset on O'Neil groups with identical
  reported curves (`stages/02_realdata/offset_batches.py`), checks of the new
  numbers in `expected_numbers.csv`, and Supplementary Tables S12 and S13; panel titles in
  Figure 3 and the Figure 6 legend wording; the figure notebooks.
* Numerical results of 0.2.0 are unchanged.

# SIR 0.2.0

* New method: isotonic Bliss excess with a plug-in parametric bootstrap at exact
  Bliss, two-stage Monte Carlo, design-matched companion mean tests, region
  scans, activity, missing-well handling and replicate screening.
* Added `sir_fit()`, `sir_screen()`, noise constructors, fit methods and Python
  setup helpers. The reference Python implementation is bundled via reticulate.
* Removed the old interaction test and its analysis pipeline. This is a breaking
  API change: callers should migrate to `sir_fit()` and provide a noise model.
* Added `reproduce/`, a staged pipeline that regenerates every number, table and
  figure of the accompanying article and checks them against the released results.
* Retained standalone Bliss, HSA, Loewe and ZIP scorers and weighted 2D isotonic
  regression. Baseline inhibition input is converted to viability before scoring.
