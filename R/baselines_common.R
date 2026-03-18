matrix_mean_response <- function(df_long, response_mode = c("viability", "inhibition"), clamp_01 = TRUE) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  response_mode <- match.arg(response_mode)
  dt <- data.table::as.data.table(df_long)
  dt <- standardize_response(dt, mode = response_mode, clamp_01 = clamp_01)
  y <- if (response_mode == "viability") dt$response_viability else dt$response_inhibition
  dt[, y := y]
  dt[, `:=`(doseA = as.numeric(doseA), doseB = as.numeric(doseB))]
  agg <- dt[, .(y = mean(y)), by = .(doseA, doseB)]
  doseA_levels <- sort(unique(agg$doseA))
  doseB_levels <- sort(unique(agg$doseB))
  agg[, i := match(doseA, doseA_levels)]
  agg[, j := match(doseB, doseB_levels)]
  I <- length(doseA_levels)
  J <- length(doseB_levels)
  Y <- matrix(NA_real_, nrow = I, ncol = J)
  Y[cbind(agg$i, agg$j)] <- agg$y
  list(Y = Y, doseA_levels = doseA_levels, doseB_levels = doseB_levels)
}

normalize_to_control <- function(Y) {
  i0 <- 1L
  j0 <- 1L
  y00 <- Y[i0, j0]
  if (!is.finite(y00) || y00 <= 0) {
    y00 <- 1
  }
  Y / y00
}

extract_monotherapy_edges <- function(Y_norm) {
  Va <- Y_norm[, 1L]
  Vb <- Y_norm[1L, ]
  list(Va = Va, Vb = Vb)
}

isoreg_monotone <- function(x, y, direction = c("decreasing", "increasing")) {
  direction <- match.arg(direction)
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) == 0) return(list(x = numeric(0), yhat = numeric(0)))
  if (direction == "decreasing") {
    fit <- stats::isoreg(x, -y)
    yhat <- -fit$yf
  } else {
    fit <- stats::isoreg(x, y)
    yhat <- fit$yf
  }
  list(x = x, yhat = yhat)
}

invert_monotone_curve <- function(x, yhat, y_target) {
  ok <- is.finite(x) & is.finite(yhat)
  x <- x[ok]; yhat <- yhat[ok]
  if (length(x) < 2) return(rep(NA_real_, length(y_target)))
  ord <- order(yhat)
  yy <- yhat[ord]
  xx <- x[ord]
  # approx requires strictly increasing x; jitter ties
  if (any(diff(yy) == 0)) yy <- yy + seq_along(yy) * 1e-12
  stats::approx(x = yy, y = xx, xout = y_target, rule = 1, ties = "ordered")$y
}
