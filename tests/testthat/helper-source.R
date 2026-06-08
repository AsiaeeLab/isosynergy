# When the SIR package is installed (the standard testthat / R CMD
# check / devtools::test() path), exported functions are reached via
# library(SIR) and internal helpers via the `SIR:::name` prefix in the
# test files.
#
# As a fallback, when these tests are run from a fresh clone *without*
# installing the package (e.g. by sourcing this helper from the repo
# root), expose the helpers directly so test files written against
# either layout keep working.
if (!requireNamespace("SIR", quietly = TRUE)) {
  root <- NULL
  if (requireNamespace("rprojroot", quietly = TRUE)) {
    root <- rprojroot::find_root(rprojroot::has_file("DESCRIPTION"))
  }
  if (is.null(root)) root <- getwd()
  files <- c(
    "R/utils.R", "R/transforms.R", "R/matrix_stats.R",
    "R/osqp_helpers.R", "R/isotonic_2d.R", "R/additive_ordered.R",
    "R/interaction.R", "R/bootstrap.R", "R/simulation.R",
    "R/baselines_common.R"
  )
  for (f in files) source(file.path(root, f))
}
