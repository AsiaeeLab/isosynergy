#' Test Synergy Relative to Bliss Independence
#'
#' Fits nonincreasing single-agent curves and a two-dimensional isotonic surface.
#' The standardized positive excess statistic tests no synergy in any combination
#' cell using a plug-in parametric bootstrap at exact Bliss. A companion test
#' assesses mean Bliss excess, and a maximum-statistic scan localizes regions.
#'
#' @param Y Numeric viability matrix, or array with dimensions
#'   `c(replicates, rows, columns)`. The first row and column hold single-agent
#'   measurements at increasing doses, including zero; the control is `Y[1, 1]`
#'   (one per replicate). Use `NA` or `NaN` for missing wells.
#' @param doses Optional list of two increasing numeric dose vectors, each
#'   starting at zero and matching the corresponding matrix dimension.
#' @param design `"S"`: single agents measured on the plate; `"R"`: at least two
#'   replicate plates with their own single agents; `"P"`: external single-agent
#'   curves in the first row and column. S with replicates is analyzed as R.
#' @param noise Required model from [sir_noise_parametric()] or [sir_noise_bank()].
#' @param B,B2 Positive integer bootstrap counts for stages one and two.
#' @param seed Optional nonnegative integer seed (up to R's maximum integer),
#'   or integer vector of seed components. Does not use or change R's RNG state.
#' @param alpha Significance level for matrix and region claims.
#' @param n_std Number of independent reference plates for scan standardization.
#' @param two_stage Run a fresh stage two when either primary or scan stage-one
#'   p-value is at most `stage2_threshold`.
#' @param stage2_threshold Stage-two trigger, default 0.05.
#' @param sa_replicates For P, number of replicates underlying reported curves.
#'   `Inf` keeps edges fixed and tests synergy relative to those reported curves.
#' @param sa_rho Correlated share of single-agent error, between zero and one.
#' @param screen_replicates Screen discrepant replicates when at least three
#'   values are available; the same rule is applied to reference plates.
#' @param screen_c Positive screen cutoff in single-replicate model SDs.
#' @param plate_var Optional nonnegative plate variance or [sir_plate_variance()]
#'   object for the design-P companion test. By default use the parametric
#'   shift variance, or zero for a bank.
#' @param group Target group label for noise-bank and plate-variance cross-fitting.
#' @param fragility `"auto"`: compute for significant single plates;
#'   `"always"`: compute for any significant fit; `"never"`: skip.
#' @param sd Optional [sir_sd_curve()] overriding statistic standardization.
#' @param normalize `"check"` requires finite controls to equal one;
#'   `"divide"` divides each plate by its positive control; `"ignore"` sets
#'   controls to one. Missing controls are set to one by the reference engine.
#' @param min_bank_pool Minimum eligible donor sets for a noise bank.
#'
#' @return An S3 `sir_fit` list containing `activity`, `activity_all_wells`,
#'   `mean_excess`, `mean_excess_se`, `p_mean`, `mean_test`, `mean_tests`,
#'   `p_mean_boot`, `t_std`, `p_tstd`, `p_tstd_stage1`, `p_scan`, `p_scan_stage1`,
#'   `regions`, `claims`, `call`, `stage`, `B_used`, `excess_map`,
#'   `raw_excess_map`, `fitted_surface`, `analyzed_matrix`, `a_hat`, `b_hat`,
#'   `doses_a`, `doses_b`, `fragility`, `log`, `settings`, `design`, and `alpha`.
#'   All returned fields are ordinary R values, so fits can be saved with saveRDS.
#'   **Indices inside regions, fragility and logs retain Python's zero-based
#'   original-grid convention**: add one to index the input R matrix. Maps use
#'   the reduced analyzed grid; excess maps exclude the zero-dose edges.
#'
#' @details Missing single-agent doses remove their entire row or column.
#'   Missing combination wells are excluded without imputation. On an incomplete
#'   single-plate S grid, the companion p-value uses bootstrap mean excess at
#'   exact Bliss and its SE is unavailable (`NaN`); treat that p-value as
#'   exploratory, because the mean null contains surfaces the primary theory does
#'   not cover. Complete S grids use V2Sp; replicate stacks use block-level t;
#'   single-plate P uses a fixed-edge t with plate/edge variance. The block-level
#'   t treats a shared single-agent reference as fixed: in design P with finite
#'   `sa_replicates` its SE omits the reference uncertainty, which the fixed-edge
#'   result in `mean_tests` includes. For replicate stacks `mean_excess` is the
#'   block-level estimate (the average over plates of each plate's mean excess),
#'   while `activity` uses the cell-wise replicate means; the two averages agree
#'   when no replicate value is screened.
#'   The default 399/9999 two-stage procedure preserves the Python method.
#'   The least-favorable-null result covers replicate means and other replicate
#'   summaries that are nondecreasing in each replicate; the replicate screen is
#'   not of this kind, and its effect on the level was evaluated by simulation.
#'   Region-scan calibration was evaluated by simulation on complete grids with
#'   parametric noise in designs S and P; it has not been evaluated with residual
#'   banks, missing wells, replicate screening or propagated single-agent
#'   uncertainty. Plate shifts were evaluated in design P, not in design S. Fragility flags describe sensitivity, not outliers.
#' @seealso [sir_screen()], [sir_python_config()]
#' @examples
#' if (reticulate::py_module_available("numpy") &&
#'     reticulate::py_module_available("numba") &&
#'     reticulate::py_module_available("scipy")) {
#'   a <- c(1, 0.9, 0.7, 0.5, 0.3)
#'   Y <- outer(a, a)
#'   fit <- sir_fit(Y, noise = sir_noise_parametric(sd = 0.08), seed = 1)
#'   print(fit)
#' }
#' @export
sir_fit <- function(Y, doses = NULL, design = c("S", "P", "R"), noise,
                    B = 399L, seed = NULL, alpha = 0.05, B2 = 9999L,
                    n_std = 200L, two_stage = TRUE, stage2_threshold = 0.05,
                    sa_replicates = Inf, sa_rho = 0.10, screen_replicates = TRUE,
                    screen_c = 5, plate_var = NULL, group = NULL,
                    fragility = c("auto", "always", "never"), sd = NULL,
                    normalize = c("check", "divide", "ignore"), min_bank_pool = 50L) {
  args <- .sir_fit_args(as.list(environment()))
  .sir_result(do.call(.sir_python()$fit, args))
}

.sir_fit_args <- function(args) {
  args$Y <- .sir_array(args$Y)
  dims <- utils::tail(dim(args$Y), 2L)
  if (any(dims < 3L)) stop("At least two nonzero doses per drug are required.", call. = FALSE)
  args$design <- match.arg(args$design, c("S", "P", "R"))
  args$fragility <- match.arg(args$fragility, c("auto", "always", "never"))
  args$normalize <- match.arg(args$normalize, c("check", "divide", "ignore"))
  if (!inherits(args$noise, "sirsyn.noise.ParametricNoise") &&
      !inherits(args$noise, "sirsyn.noise.NoiseBank"))
    stop("noise must be created by sir_noise_parametric() or sir_noise_bank().", call. = FALSE)
  for (nm in c("B", "B2", "n_std", "min_bank_pool"))
    args[[nm]] <- .sir_number(args[[nm]], nm, if (nm == "n_std") 2 else 1,
                             .Machine$integer.max, TRUE)
  for (nm in c("alpha", "stage2_threshold", "sa_rho"))
    args[[nm]] <- .sir_number(args[[nm]], nm, 0, 1)
  args$screen_c <- .sir_number(args$screen_c, "screen_c", .Machine$double.eps)
  if (!identical(args$sa_replicates, Inf))
    args$sa_replicates <- .sir_number(args$sa_replicates, "sa_replicates", 1)
  for (nm in c("two_stage", "screen_replicates"))
    if (!is.logical(args[[nm]]) || length(args[[nm]]) != 1L || is.na(args[[nm]]))
      stop(nm, " must be TRUE or FALSE.", call. = FALSE)
  if (!is.null(args$seed)) {
    if (!is.numeric(args$seed) || !length(args$seed) || any(!is.finite(args$seed)) ||
        any(args$seed < 0 | args$seed > .Machine$integer.max | args$seed != floor(args$seed)))
      stop("seed must contain nonnegative integers up to .Machine$integer.max.", call. = FALSE)
    args$seed <- if (length(args$seed) == 1L) as.integer(args$seed) else as.list(as.integer(args$seed))
  }
  if (!is.null(args$doses)) {
    if (!is.list(args$doses) || length(args$doses) != 2L)
      stop("doses must be a list of two dose vectors.", call. = FALSE)
    for (k in 1:2) {
      d <- args$doses[[k]]
      if (!is.numeric(d) || length(d) != dims[k] || any(!is.finite(d)) ||
          d[1L] != 0 || any(diff(d) <= 0))
        stop("doses must increase from zero and match the grid dimensions.", call. = FALSE)
    }
    args$doses <- lapply(args$doses, as.numeric)
  }
  if (!is.null(args$group)) {
    if (length(args$group) != 1L || is.na(args$group)) stop("group must be one nonmissing label.", call. = FALSE)
    args$group <- as.character(args$group)
  }
  if (is.numeric(args$plate_var)) args$plate_var <- .sir_number(args$plate_var, "plate_var")
  if (args$normalize == "divide") {
    controls <- if (length(dim(args$Y)) == 2L) args$Y[1L, 1L] else args$Y[, 1L, 1L]
    if (any(controls <= 0, na.rm = TRUE)) stop("Cannot divide by a nonpositive control.", call. = FALSE)
  }
  args
}

.sir_result <- function(result) {
  # dataclasses.asdict retains ndarray shapes until reticulate converts them.
  out <- reticulate::py_to_r(reticulate::import("dataclasses", convert = FALSE)$asdict(result))
  out$claims <- Filter(function(r) isTRUE(r$claimed), out$regions)
  out$call <- isTRUE(out$p_tstd <= out$alpha)
  class(out) <- "sir_fit"
  out
}
