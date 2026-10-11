skip_without_python <- function() {
  skip_if_not(reticulate::py_module_available("numpy"))
  skip_if_not(reticulate::py_module_available("scipy"))
  skip_if_not(reticulate::py_module_available("numba"))
}

reference_python <- function() {
  reticulate::import_from_path("sirsyn", system.file("python", package = "SIR"), convert = FALSE)
}

simulated_plate <- function(seed = 15L) {
  set.seed(seed)
  Y <- outer(c(1, .95, .8, .55, .3), c(1, .9, .7, .45, .25)) +
    matrix(rnorm(25, sd = .04), 5)
  Y[1, 1] <- 1
  Y
}

reference_list <- function(result) {
  reticulate::py_to_r(reticulate::import("dataclasses", convert = FALSE)$asdict(result))
}
