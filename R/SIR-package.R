#' SIR: Shape-Constrained Drug Synergy Testing via Isotonic Regression
#'
#' SIR (Synergy via Isotonic Regression) is a nonparametric framework for
#' drug combination synergy testing. The package fits two competing
#' shape-constrained surfaces to a dose-response matrix---a fully monotone
#' two-dimensional isotonic surface and a monotone-additive null---and
#' summarises their difference as an interaction surface. A
#' degrees-of-freedom-corrected wild bootstrap calibrates a p-value
#' against the additive null without parametric distributional assumptions.
#'
#' @section Main entry points:
#' * [sir_test()] runs the full pipeline (fits both surfaces, computes
#'   the interaction energy, and returns a wild-bootstrap p-value).
#' * [additive_ordered_fit()] and [isotonic_2d_fit()] expose the two
#'   underlying fits.
#' * [wild_bootstrap()] performs the bootstrap p-value computation.
#' * [interaction_summaries()] and [interaction_fit()] compute summary
#'   statistics from a fitted interaction surface.
#' * [bliss_score()], [hsa_score()], [loewe_score()], [zip_score()]
#'   provide classical baselines for comparison.
#'
#' @section Reference:
#' Asiaee A, Long JP, Pal S, Pua HH, Coombes KR (2026).
#' \emph{A shape-constrained regression and wild bootstrap framework for
#' reproducible drug synergy testing}. bioRxiv.
#' \doi{10.1101/2026.02.05.704019}
#'
#' @keywords internal
#' @importFrom data.table := .N .SD
#' @importFrom stats isoreg median optim plogis qlogis quantile rnorm rt sd var approx complete.cases
#' @importFrom utils read.csv write.csv
#' @importFrom grDevices dev.off pdf
"_PACKAGE"

# data.table NSE confuses R CMD check's static analyser. The names below
# are referenced via non-standard evaluation (e.g. `dt[, x := ...]`),
# never as true free symbols, so silencing the NOTE here is the
# canonical workaround.
utils::globalVariables(c(
  ".", "..cols", "..key_cols", ".data",
  # data.table aggregation columns
  "Z", "doseA", "doseB", "i", "j", "m", "s2",
  "experiment_id", "replicate", "response", "response_raw",
  "response_mode", "response_viability", "response_inhibition",
  "drugA", "drugB", "drug1", "drug2", "drugA_id", "drugB_id",
  "drug_name", "cell_line", "cell_line_clean", "cell_line_raw",
  "block_id", "row_index", "col_index", "study_source",
  "study", "plate", "test_date", "panel", "source_detail",
  "n_i", "n_j", "nsc", "concindex", "concindex1", "concindex2",
  "dose", "doseA_unit", "doseB_unit", "metric", "value",
  "value_base", "value_pert", "delta", "perturbation", "perturbed",
  "rep_idx", "correlation", "group_id", "N", "p_value",
  "pair_key", "power_0.05", "sign_flip", "sim_interaction_strength",
  "t_int",
  # NCI-ALMANAC raw columns (CSV header names)
  "VALID", "PERCENTGROWTH", "STUDY", "PLATE", "TESTDATE", "PANEL",
  "CELLNAME", "NSC1", "NSC2", "NSC", "CONC1", "CONC2",
  "CONCINDEX1", "CONCINDEX2",
  # DrugCombDB / SynergyFinder columns
  "Bliss", "BlockID", "ConcCol", "ConcColUnit", "ConcRow",
  "ConcRowUnit", "Drug1", "Drug2", "DrugCol", "DrugRow",
  "HSA", "ID", "Loewe", "Response", "ZIP", "Cell line",
  # Score / metric columns surfaced inside experiment helpers
  "S2", "S2_1", "S2_2", "Sminus", "Splus",
  "baseline_mean", "bliss_mean", "hsa_mean", "loewe_mean",
  "zip_mean", "clinical"
))
