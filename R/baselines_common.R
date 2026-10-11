matrix_mean_response <- function(df_long, response_mode = c("viability", "inhibition")) {
  response_mode <- match.arg(response_mode)
  required <- c("doseA", "doseB", "response")
  if (!is.data.frame(df_long) || !all(required %in% names(df_long)) ||
      !all(vapply(df_long[required], is.numeric, logical(1))) || !nrow(df_long) ||
      any(!is.finite(as.matrix(df_long[required]))))
    stop("Supply finite numeric doseA, doseB and response columns.", call. = FALSE)
  da <- sort(unique(df_long$doseA))
  db <- sort(unique(df_long$doseB))
  if (da[1L] != 0 || db[1L] != 0) stop("Both dose axes must begin at zero.", call. = FALSE)
  values <- if (response_mode == "inhibition") 1 - df_long$response else df_long$response
  idx <- match(df_long$doseA, da) + (match(df_long$doseB, db) - 1L) * length(da)
  avg <- tapply(values, idx, mean)
  Y <- matrix(NA_real_, length(da), length(db))
  Y[as.integer(names(avg))] <- avg
  if (anyNA(Y)) stop("Standalone scorers need a complete dose grid.", call. = FALSE)
  list(Y = Y, doseA_levels = da, doseB_levels = db)
}

clamp01 <- function(x, eps = 0) pmin(1 - eps, pmax(eps, x))

normalize_to_control <- function(Y) {
  i0 <- 1L
  j0 <- 1L
  y00 <- Y[i0, j0]
  if (!is.finite(y00) || y00 <= 0) {
    stop("The zero-dose control must be positive.", call. = FALSE)
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
