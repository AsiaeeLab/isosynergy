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
    # Fall back to isotonic regression on dose.
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

  # Work in inhibition fractions.
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

