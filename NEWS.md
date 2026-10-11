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
