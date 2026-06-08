#' Bliss Independence Synergy Surface
#'
#' Computes the Bliss independence synergy surface
#' `S = E - Y`, where the expected viability `E` is the outer product of
#' the marginal monotherapy curves. The dose-response table is
#' aggregated to its mean response on the dose grid before scoring.
#'
#' @param df_long Long-format dose-response data frame with columns
#'   `doseA`, `doseB`, `response`. Replicates are pooled by mean.
#' @param response_mode Either `"viability"` or `"inhibition"`.
#'
#' @return A list with components `synergy` (the synergy surface),
#'   `expected` (independence baseline), `observed` (normalised mean
#'   response), `doseA_levels`, and `doseB_levels`.
#'
#' @seealso [bliss_score()] (alias), [hsa_synergy()],
#'   [loewe_synergy()], [zip_synergy()].
#' @export
bliss_synergy <- function(df_long, response_mode) {
  grid <- matrix_mean_response(df_long, response_mode = response_mode)
  Y_norm <- normalize_to_control(grid$Y)
  edges <- extract_monotherapy_edges(Y_norm)
  E <- outer(edges$Va, edges$Vb, "*")
  S <- E - Y_norm
  list(
    synergy = S,
    expected = E,
    observed = Y_norm,
    doseA_levels = grid$doseA_levels,
    doseB_levels = grid$doseB_levels
  )
}

