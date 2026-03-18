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

