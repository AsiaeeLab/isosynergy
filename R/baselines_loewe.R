#' Loewe Additivity Synergy Surface
#'
#' Computes the Loewe additivity synergy surface using monotone fits to
#' the marginal single-agent curves. The combination index is computed
#' from the inverse monotone curves at each cell of the dose grid.
#'
#' @param df_long Long-format dose-response data frame with columns
#'   `doseA`, `doseB`, `response`.
#' @param response_mode Either `"viability"` or `"inhibition"`.
#' @param direction Direction of monotonicity (`"decreasing"` for
#'   viability, `"increasing"` for inhibition).
#'
#' @return A list with components `synergy`, `combo_index`, `observed`,
#'   `doseA_levels`, `doseB_levels`.
#'
#' @seealso [loewe_score()] (alias), [bliss_synergy()],
#'   [hsa_synergy()], [zip_synergy()].
#' @export
loewe_synergy <- function(df_long, response_mode, direction = c("decreasing", "increasing")) {
  direction <- match.arg(direction)
  grid <- matrix_mean_response(df_long, response_mode = response_mode)
  Y_norm <- normalize_to_control(grid$Y)
  dosesA <- grid$doseA_levels
  dosesB <- grid$doseB_levels

  edges <- extract_monotherapy_edges(Y_norm)
  fitA <- isoreg_monotone(dosesA, edges$Va, direction = direction)
  fitB <- isoreg_monotone(dosesB, edges$Vb, direction = direction)

  y_obs <- Y_norm
  da_eq <- invert_monotone_curve(fitA$x, fitA$yhat, as.numeric(y_obs))
  db_eq <- invert_monotone_curve(fitB$x, fitB$yhat, as.numeric(y_obs))

  doseA_mat <- matrix(dosesA, nrow = length(dosesA), ncol = length(dosesB))
  doseB_mat <- matrix(dosesB, nrow = length(dosesA), ncol = length(dosesB), byrow = TRUE)

  CI <- (as.numeric(doseA_mat) / da_eq) + (as.numeric(doseB_mat) / db_eq)
  CI[!is.finite(CI)] <- NA_real_
  S <- 1 - matrix(CI, nrow = nrow(Y_norm), ncol = ncol(Y_norm))

  list(
    synergy = S,
    combo_index = matrix(CI, nrow = nrow(Y_norm), ncol = ncol(Y_norm)),
    observed = Y_norm,
    doseA_levels = dosesA,
    doseB_levels = dosesB
  )
}
