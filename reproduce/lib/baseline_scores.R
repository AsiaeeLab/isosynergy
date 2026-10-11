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
  if (any(diff(yy) == 0)) yy <- yy + seq_along(yy) * 1e-12
  stats::approx(x = yy, y = xx, xout = y_target, rule = 1, ties = "ordered")$y
}

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

fit_4pl_inhibition <- function(dose, inhib, eps = 1e-6) {
  dose <- as.numeric(dose)
  inhib <- clamp01(as.numeric(inhib), eps = 0)
  ok <- is.finite(dose) & is.finite(inhib)
  dose <- dose[ok]
  inhib <- inhib[ok]
  if (length(dose) < 4) {
    return(function(x) rep(mean(inhib, na.rm = TRUE), length(x)))
  }

  dpos <- dose[dose > 0]
  if (length(dpos) == 0) dpos <- rep(1, length(dose))
  d0 <- min(dpos)
  dose2 <- pmax(dose, d0 * 1e-3)

  nll <- function(par) {
    E0 <- plogis(par[1])
    Emax <- plogis(par[2])
    ec50 <- exp(par[3])
    h <- exp(par[4])
    pred <- E0 + (Emax - E0) / (1 + (ec50 / dose2)^h)
    sum((inhib - pred)^2)
  }

  p0 <- c(
    qlogis(clamp01(min(inhib), eps = 1e-4)),
    qlogis(clamp01(max(inhib), eps = 1e-4)),
    log(stats::median(dpos)),
    log(1)
  )

  opt <- tryCatch(
    stats::optim(p0, nll, method = "BFGS", control = list(maxit = 2000)),
    error = function(e) NULL
  )

  if (is.null(opt) || !is.finite(opt$value)) {
    fit <- isoreg_monotone(dose2, inhib, direction = "increasing")
    return(function(x) {
      x <- pmax(as.numeric(x), d0 * 1e-3)
      stats::approx(fit$x, fit$yhat, xout = x, rule = 2)$y
    })
  }

  par <- opt$par
  E0 <- plogis(par[1])
  Emax <- plogis(par[2])
  ec50 <- exp(par[3])
  h <- exp(par[4])
  function(x) {
    x <- pmax(as.numeric(x), d0 * 1e-3)
    E0 + (Emax - E0) / (1 + (ec50 / x)^h)
  }
}

zip_synergy <- function(df_long, response_mode, use_curvefit = TRUE) {
  grid <- matrix_mean_response(df_long, response_mode = response_mode)
  Y_norm <- normalize_to_control(grid$Y)
  dosesA <- grid$doseA_levels
  dosesB <- grid$doseB_levels

  I_obs <- clamp01(1 - Y_norm, eps = 0)
  edges <- extract_monotherapy_edges(Y_norm)
  I_A <- clamp01(1 - edges$Va, eps = 0)
  I_B <- clamp01(1 - edges$Vb, eps = 0)

  if (use_curvefit) {
    fA <- fit_4pl_inhibition(dosesA, I_A)
    fB <- fit_4pl_inhibition(dosesB, I_B)
    IA_hat <- clamp01(fA(dosesA), eps = 0)
    IB_hat <- clamp01(fB(dosesB), eps = 0)
  } else {
    IA_hat <- I_A
    IB_hat <- I_B
  }

  E <- outer(IA_hat, IB_hat, function(a, b) a + b - a * b)
  S <- I_obs - E
  list(
    synergy = S,
    expected = E,
    observed = I_obs,
    doseA_levels = dosesA,
    doseB_levels = dosesB
  )
}



clamp01 <- function(x, eps=0) pmin(1-eps,pmax(eps,x))
standardize_response <- function(dt, mode, clamp_01=TRUE) {
  dt <- data.table::copy(dt)
  y <- as.numeric(dt$response)
  if (clamp_01) y <- clamp01(y)
  if (mode == "viability") {
    dt[, response_viability := y]; dt[, response_inhibition := 1-y]
  } else {
    dt[, response_inhibition := y]; dt[, response_viability := 1-y]
  }
  dt
}
