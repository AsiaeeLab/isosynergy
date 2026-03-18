synergyfinder_available <- function() {
  requireNamespace("synergyfinder", quietly = TRUE)
}

as_synergyfinder_input <- function(df_long,
                                  response_mode = c("viability", "inhibition"),
                                  block_id = 1,
                                  drug_row = "DrugA",
                                  drug_col = "DrugB",
                                  conc_row_unit = "a.u.",
                                  conc_col_unit = "a.u.") {
  response_mode <- match.arg(response_mode)
  grid <- matrix_mean_response(df_long, response_mode = response_mode)
  Y <- normalize_to_control(grid$Y)
  I <- nrow(Y)
  J <- ncol(Y)

  df_sf <- expand.grid(Row = seq_len(I), Col = seq_len(J))
  df_sf$BlockID <- block_id
  df_sf$DrugRow <- drug_row
  df_sf$DrugCol <- drug_col

  df_sf$ConcRow <- grid$doseA_levels[df_sf$Row]
  df_sf$ConcCol <- grid$doseB_levels[df_sf$Col]
  df_sf$ConcRowUnit <- conc_row_unit
  df_sf$ConcColUnit <- conc_col_unit

  # SynergyFinder expects Response in percent units.
  df_sf$Response <- 100 * as.numeric(Y)

  # Keep the original grid metadata for downstream alignment.
  list(
    data = df_sf,
    doseA_levels = grid$doseA_levels,
    doseB_levels = grid$doseB_levels
  )
}

synergyfinder_synergy <- function(df_long,
                                  method = c("ZIP", "Bliss", "HSA", "Loewe"),
                                  response_mode = c("viability", "inhibition"),
                                  correction = FALSE,
                                  Emin = 0,
                                  Emax = 100,
                                  nan.handle = c("LL4", "L4")) {
  method <- match.arg(method)
  response_mode <- match.arg(response_mode)
  nan.handle <- match.arg(nan.handle)
  if (!synergyfinder_available()) {
    stop("Package 'synergyfinder' is not installed. Try: renv::install('hly89/synergyfinder')")
  }

  inp <- as_synergyfinder_input(df_long, response_mode = response_mode)
  data_type <- if (response_mode == "viability") "viability" else "inhibition"

  reshaped <- synergyfinder::ReshapeData(inp$data, data.type = data_type)
  scored <- synergyfinder::CalculateSynergy(
    reshaped,
    method = method,
    correction = correction,
    Emin = Emin,
    Emax = Emax,
    nan.handle = nan.handle
  )

  S <- scored$scores[[1]]
  I <- length(inp$doseA_levels)
  J <- length(inp$doseB_levels)

  list(
    # SynergyFinder scores are in the same percent-scale as the inhibition matrix.
    # For consistency with our internal baselines (which operate on [0,1] fractions),
    # we rescale to a unit-free fraction scale.
    synergy = matrix(as.numeric(S) / 100, nrow = I, ncol = J),
    doseA_levels = inp$doseA_levels,
    doseB_levels = inp$doseB_levels,
    method = method
  )
}

synergyfinder_na_stats <- function(df_long, response_mode) {
  stats <- list()
  for (m in c("ZIP", "Bliss", "HSA", "Loewe")) {
    res <- tryCatch(
      synergyfinder_synergy(df_long, method = m, response_mode = response_mode),
      error = function(e) NULL
    )
    if (is.null(res) || is.null(res$synergy)) {
      stats[[m]] <- list(finite_frac = NA_real_, na_reason = "error")
      next
    }
    syn <- res$synergy
    stats[[m]] <- list(
      finite_frac = mean(is.finite(as.numeric(syn))),
      na_reason = if (all(is.na(as.numeric(syn)))) "all_na" else NA_character_,
      doseA_levels = res$doseA_levels,
      doseB_levels = res$doseB_levels,
      synergy = syn
    )
  }
  stats
}
