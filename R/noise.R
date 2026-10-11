#' Single-Replicate Standard Deviation Curve
#'
#' @param kind Constant SD, proportional SD (with viability floor 0.02),
#'   linearly interpolated table, or the built-in O'Neil pooled SD curve.
#' @param value Positive constant SD or proportionality coefficient.
#' @param knots Increasing mean viability values for `kind = "table"`.
#' @param sds Positive SD values at the knots. Extrapolation is constant.
#' @return A Python `sirsyn.SDCurve` object for use in noise constructors.
#' @export
sir_sd_curve <- function(kind = c("constant", "proportional", "table", "oneil"),
                         value = 0.08, knots = NULL, sds = NULL) {
  kind <- match.arg(kind)
  py <- .sir_python()
  if (kind == "oneil") return(py$oneil_sd_curve())
  if (kind == "table") {
    if (!is.numeric(knots) || !is.numeric(sds) || length(knots) < 2L ||
        length(knots) != length(sds) || any(!is.finite(c(knots, sds))) ||
        any(diff(knots) <= 0) || any(sds <= 0))
      stop("A table needs increasing finite knots and equally many positive finite sds.", call. = FALSE)
    return(py$SDCurve(kind, kv = as.numeric(knots), ks = as.numeric(sds)))
  }
  py$SDCurve(kind, par = .sir_number(value, "value", .Machine$double.eps))
}

#' Parametric Noise Model
#'
#' Noise is added on the viability scale, except for the lognormal law.
#' The Gaussian lognormal law is mean preserving. The t4 option with lognormal
#' noise does not have a finite mean and should not be used for Bliss calibration.
#' Supply a noise model appropriate to the assay; it is not estimated by sir_fit.
#' @param model Noise law. `"table"` uses `curve`; `"oneil"` uses the built-in SD table.
#' @param sd Positive additive SD, log-scale SD, or constant CV, as appropriate.
#' @param curve An object from [sir_sd_curve()] for `model = "table"`.
#' @param tails Gaussian or variance-standardized Student t with four degrees of freedom.
#' @param plate_shift_ratio Nonnegative ratio of plate-shift SD to the model SD.
#' @param plate_shift_sd Optional nonnegative plate-shift SD overriding the ratio.
#' @return A Python `sirsyn.ParametricNoise` object accepted by [sir_fit()].
#' @export
sir_noise_parametric <- function(model = c("additive", "lognormal", "constant_cv", "table", "oneil"),
                                 sd = 0.08, curve = NULL, tails = c("gaussian", "t4"),
                                 plate_shift_ratio = 0, plate_shift_sd = NULL) {
  model <- match.arg(model)
  tails <- match.arg(tails)
  sd <- .sir_number(sd, "sd", .Machine$double.eps)
  plate_shift_ratio <- .sir_number(plate_shift_ratio, "plate_shift_ratio")
  if (!is.null(plate_shift_sd)) plate_shift_sd <- .sir_number(plate_shift_sd, "plate_shift_sd")
  py <- .sir_python()
  if (model == "table" && !inherits(curve, "sirsyn.noise.SDCurve"))
    stop("curve must be a sir_sd_curve() table.", call. = FALSE)
  if (model == "oneil") curve <- py$oneil_sd_curve()
  py$ParametricNoise(family = if (model %in% c("table", "oneil")) "additive_table" else model,
    par = if (model %in% c("table", "oneil")) 1 else sd, table = curve, tails = tails, plate_shift_ratio = plate_shift_ratio,
    plate_shift_sd = plate_shift_sd)
}

#' Estimate an SD Curve from Replicate Plates
#' @param Y Numeric array with dimensions `c(sets, replicates, rows, columns)`.
#' @param n_bins Number of quantile bins of mean viability.
#' @param min_wells Minimum number of wells per retained bin.
#' @param wells Use combination wells or all non-control wells with replicate spread.
#' @return A Python SD curve for [sir_noise_parametric()] or [sir_noise_bank()].
#' @export
sir_sd_from_replicates <- function(Y, n_bins = 13L, min_wells = 20L,
                                   wells = c("combination", "all")) {
  Y <- .sir_array(Y, 4L)
  .sir_python()$sd_curve_from_replicates(Y,
    n_bins = .sir_number(n_bins, "n_bins", 2, .Machine$integer.max, TRUE),
    min_wells = .sir_number(min_wells, "min_wells", 1, .Machine$integer.max, TRUE),
    wells = match.arg(wells))
}

#' Noise Bank from Replicate Residuals
#'
#' Donors are sampled within potency strata, from folds other than the target
#' group, without selection on realized noise. Cellwise SD ratios are clipped
#' to 0.5--2. Supply the target `group` to [sir_fit()] to enable cross-fitting.
#' @inheritParams sir_sd_from_replicates
#' @param groups One nonmissing group label per set (for example cell line).
#' @param curve Optional SD curve; by default it is estimated from `Y`.
#' @param include_edges Include single-agent residuals, required for designs S and R.
#' @param n_strata Number of potency strata.
#' @param n_folds Number of group folds for cross-fitting.
#' @return A Python `sirsyn.NoiseBank` object accepted by [sir_fit()].
#' @export
sir_noise_bank <- function(Y, groups, curve = NULL, include_edges = FALSE,
                           n_strata = 10L, n_folds = 2L) {
  Y <- .sir_array(Y, 4L)
  if (length(groups) != dim(Y)[1L] || anyNA(groups))
    stop("groups must have one nonmissing label per set.", call. = FALSE)
  .sir_python()$NoiseBank$from_replicates(Y, groups = as.list(as.character(groups)),
    sd = curve, include_edges = include_edges,
    n_strata = .sir_number(n_strata, "n_strata", 1, .Machine$integer.max, TRUE),
    n_folds = .sir_number(n_folds, "n_folds", 2, .Machine$integer.max, TRUE))
}

#' Estimate Plate Variance by Viability and Group Fold
#' @inheritParams sir_noise_bank
#' @return A Python `sirsyn.PlateVarianceTable` for the `plate_var` argument of
#'   [sir_fit()]. Use the same group labels as the noise bank.
#' @export
sir_plate_variance <- function(Y, groups, n_folds = 2L) {
  Y <- .sir_array(Y, 4L)
  if (dim(Y)[2L] < 2L || anyNA(Y) || length(groups) != dim(Y)[1L] || anyNA(groups))
    stop("Complete sets with at least two replicates and one group label per set are required.", call. = FALSE)
  .sir_python()$PlateVarianceTable(Y, groups = as.list(as.character(groups)),
    n_folds = .sir_number(n_folds, "n_folds", 2, .Machine$integer.max, TRUE))
}
