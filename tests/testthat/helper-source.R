root <- NULL
if (requireNamespace("rprojroot", quietly = TRUE)) {
  root <- rprojroot::find_root(rprojroot::has_file("renv.lock"))
}
if (is.null(root)) root <- getwd()

source(file.path(root, "R/utils.R"))
source(file.path(root, "R/transforms.R"))
source(file.path(root, "R/matrix_stats.R"))
source(file.path(root, "R/osqp_helpers.R"))
source(file.path(root, "R/isotonic_2d.R"))
source(file.path(root, "R/additive_ordered.R"))
source(file.path(root, "R/interaction.R"))
source(file.path(root, "R/bootstrap.R"))
source(file.path(root, "R/simulation.R"))
