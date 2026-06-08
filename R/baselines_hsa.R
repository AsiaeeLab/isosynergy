#' Highest Single Agent (HSA) Synergy Surface
#'
#' Computes the HSA synergy surface, where the expected viability is
#' the cell-wise minimum of the two marginal single-agent curves.
#'
#' @param df_long Long-format dose-response data frame with columns
#'   `doseA`, `doseB`, `response`.
#' @param response_mode Either `"viability"` or `"inhibition"`.
#'
#' @return A list with components `synergy`, `expected`, `observed`,
#'   `doseA_levels`, `doseB_levels`.
#'
#' @seealso [hsa_score()] (alias), [bliss_synergy()],
#'   [loewe_synergy()], [zip_synergy()].
#' @export
hsa_synergy <- function(df_long, response_mode) {
  grid <- matrix_mean_response(df_long, response_mode = response_mode)
  Y_norm <- normalize_to_control(grid$Y)
  edges <- extract_monotherapy_edges(Y_norm)
  E <- outer(edges$Va, edges$Vb, pmin)
  S <- E - Y_norm
  list(
    synergy = S,
    expected = E,
    observed = Y_norm,
    doseA_levels = grid$doseA_levels,
    doseB_levels = grid$doseB_levels
  )
}

