suppressPackageStartupMessages({
  if (requireNamespace("data.table", quietly = TRUE)) {
    `%DT%` <- TRUE
  }
})

standardize_response <- function(df, mode = c("viability", "inhibition"), clamp_01 = TRUE) {
  mode <- match.arg(mode)
  if (!all(c("response") %in% names(df))) stop("Expected column 'response'")
  y <- df$response
  if (clamp_01) y <- clamp01(y, eps = 0)
  if (mode == "viability") {
    df$response_viability <- y
    df$response_inhibition <- 1 - y
  } else {
    df$response_inhibition <- y
    df$response_viability <- 1 - y
  }
  df
}

infer_monotone_direction <- function(response_mode, method_direction) {
  if (!identical(method_direction, "auto")) return(method_direction)
  if (identical(response_mode, "viability")) return("decreasing")
  if (identical(response_mode, "inhibition")) return("increasing")
  stop("Unknown response_mode: ", response_mode)
}

summarise_matrix <- function(df_long, transform, response_mode, method_direction,
                             tau = 1e-6, winsor_q = 0.99) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dt <- data.table::as.data.table(df_long)
  req <- c("doseA", "doseB", "response", "replicate")
  missing <- setdiff(req, names(dt))
  if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

  direction <- infer_monotone_direction(response_mode, method_direction)

  dt <- standardize_response(dt, mode = response_mode, clamp_01 = TRUE)
  y <- if (response_mode == "viability") dt$response_viability else dt$response_inhibition
  dt[, Z := transform$forward(y)]

  dt[, `:=`(
    doseA = as.numeric(doseA),
    doseB = as.numeric(doseB),
    replicate = as.character(replicate)
  )]

  doseA_levels <- sort(unique(dt$doseA))
  doseB_levels <- sort(unique(dt$doseB))

  dt[, i := match(doseA, doseA_levels)]
  dt[, j := match(doseB, doseB_levels)]

  agg <- dt[, .(
    m = .N,
    barZ = mean(Z),
    s2 = if (.N > 1) stats::var(Z) else NA_real_
  ), by = .(i, j)]

  I <- length(doseA_levels)
  J <- length(doseB_levels)

  barZ_mat <- matrix(NA_real_, nrow = I, ncol = J)
  m_mat <- matrix(0, nrow = I, ncol = J)
  s2_mat <- matrix(NA_real_, nrow = I, ncol = J)

  barZ_mat[cbind(agg$i, agg$j)] <- agg$barZ
  m_mat[cbind(agg$i, agg$j)] <- agg$m
  s2_mat[cbind(agg$i, agg$j)] <- agg$s2

  s2_filled <- s2_mat
  s2_filled[is.na(s2_filled)] <- tau

  w_mat <- m_mat / pmax(s2_filled, tau)
  w_vec <- as.numeric(w_mat)
  cap <- as.numeric(stats::quantile(w_vec[w_vec > 0], winsor_q, na.rm = TRUE, names = FALSE, type = 8))
  if (is.finite(cap)) w_mat <- pmin(w_mat, cap)

  list(
    I = I, J = J,
    doseA_levels = doseA_levels,
    doseB_levels = doseB_levels,
    barZ = barZ_mat,
    w = w_mat,
    m = m_mat,
    s2 = s2_mat,
    direction = direction
  )
}

