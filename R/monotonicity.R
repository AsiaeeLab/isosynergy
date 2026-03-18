summarise_monotonicity <- function(df_long, response_mode = c("viability", "inhibition"),
                                   clamp_01 = TRUE,
                                   use_transform = TRUE,
                                   transform = NULL,
                                   method_direction = "auto") {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  response_mode <- match.arg(response_mode)
  dt <- data.table::as.data.table(df_long)
  req <- c("doseA", "doseB", "response")
  missing <- setdiff(req, names(dt))
  if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

  dt <- standardize_response(dt, mode = response_mode, clamp_01 = clamp_01)
  y <- if (response_mode == "viability") dt$response_viability else dt$response_inhibition

  direction <- infer_monotone_direction(response_mode, method_direction)

  if (isTRUE(use_transform)) {
    if (is.null(transform)) stop("transform is required when use_transform=TRUE")
    z <- transform$forward(y)
  } else {
    z <- y
  }

  dt[, `:=`(doseA = as.numeric(doseA), doseB = as.numeric(doseB))]
  dt[, z := z]
  agg <- dt[, .(z = mean(z, na.rm = TRUE)), by = .(doseA, doseB)]
  doseA_levels <- sort(unique(agg$doseA))
  doseB_levels <- sort(unique(agg$doseB))
  agg[, i := match(doseA, doseA_levels)]
  agg[, j := match(doseB, doseB_levels)]

  I <- length(doseA_levels)
  J <- length(doseB_levels)
  Z <- matrix(NA_real_, nrow = I, ncol = J)
  Z[cbind(agg$i, agg$j)] <- agg$z

  dxA <- Z[-1, , drop = FALSE] - Z[-I, , drop = FALSE]
  dxB <- Z[, -1, drop = FALSE] - Z[, -J, drop = FALSE]

  okA <- is.finite(dxA)
  okB <- is.finite(dxB)
  n_comp_A <- sum(okA)
  n_comp_B <- sum(okB)

  if (direction == "decreasing") {
    vioA <- dxA[okA] > 0
    vioB <- dxB[okB] > 0
    magA <- pmax(dxA[okA], 0)
    magB <- pmax(dxB[okB], 0)
  } else {
    vioA <- dxA[okA] < 0
    vioB <- dxB[okB] < 0
    magA <- pmax(-dxA[okA], 0)
    magB <- pmax(-dxB[okB], 0)
  }

  n_vio_A <- sum(vioA)
  n_vio_B <- sum(vioB)
  frac_vio_A <- if (n_comp_A > 0) n_vio_A / n_comp_A else NA_real_
  frac_vio_B <- if (n_comp_B > 0) n_vio_B / n_comp_B else NA_real_

  list(
    direction = direction,
    scale = if (isTRUE(use_transform)) "transformed" else "response",
    n_i = I,
    n_j = J,
    n_comp_A = n_comp_A,
    n_comp_B = n_comp_B,
    n_vio_A = n_vio_A,
    n_vio_B = n_vio_B,
    frac_vio_A = frac_vio_A,
    frac_vio_B = frac_vio_B,
    mean_vio_mag_A = if (n_comp_A > 0) mean(magA, na.rm = TRUE) else NA_real_,
    mean_vio_mag_B = if (n_comp_B > 0) mean(magB, na.rm = TRUE) else NA_real_,
    max_vio_mag_A = if (n_comp_A > 0) max(magA, na.rm = TRUE) else NA_real_,
    max_vio_mag_B = if (n_comp_B > 0) max(magB, na.rm = TRUE) else NA_real_
  )
}

