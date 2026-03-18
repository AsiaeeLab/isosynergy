wild_bootstrap <- function(barZ, w, direction, B, seed = 1, osqp_pars = list(),
                           n_cores = 1, use_parallel = TRUE,
                           stat = c("t_int", "t_int_norm", "S2", "S2_norm", "Splus", "Sminus", "max_synergy",
                                    "S2_plus", "S2_minus",
                                    "area_synergy", "area_synergy_weighted"),
                           synergy_sign = c("negative", "positive"),
                           threshold = 0,
                           residual_scale = c("none", "df", "hc2"),
                           df_null = NULL,
                           df_tol = 1e-8) {
  stat <- match.arg(stat)
  synergy_sign <- match.arg(synergy_sign)
  residual_scale <- match.arg(residual_scale)
  fit0_add <- additive_ordered_fit(barZ, w, direction = direction, osqp_pars = osqp_pars)
  fit0_iso <- isotonic_2d_fit(barZ, w, direction = direction, osqp_pars = osqp_pars)

  theta_add <- fit0_add$theta
  y <- barZ
  r <- y - theta_add

  compute_additive_leverage <- function(w, I, J) {
    wv <- as.numeric(w)
    ok <- is.finite(wv) & (wv > 0)
    n <- I * J
    p <- 1L + (I - 1L) + (J - 1L)

    i_idx <- rep(seq_len(I), times = J)
    j_idx <- rep(seq_len(J), each = I)

    X <- matrix(0, nrow = n, ncol = p)
    X[, 1L] <- 1

    if (I > 1L) {
      for (i in seq_len(I - 1L)) {
        X[i_idx == i, 1L + i] <- 1
      }
      X[i_idx == I, 2L:I] <- -1
    }
    if (J > 1L) {
      v_start <- 1L + (I - 1L) + 1L
      for (j in seq_len(J - 1L)) {
        X[j_idx == j, v_start + (j - 1L)] <- 1
      }
      X[j_idx == J, v_start:(v_start + (J - 2L))] <- -1
    }

    X_ok <- X[ok, , drop = FALSE]
    ws <- sqrt(wv[ok])
    Xw <- X_ok * ws
    XtX <- crossprod(Xw)
    XtX_inv <- tryCatch(solve(XtX), error = function(e) NULL)
    if (is.null(XtX_inv)) {
      return(matrix(0, nrow = I, ncol = J))
    }
    h_ok <- rowSums((Xw %*% XtX_inv) * Xw)
    h <- numeric(n)
    h[ok] <- h_ok
    matrix(h, nrow = I, ncol = J)
  }

  if (residual_scale == "df") {
    n_distinct_runs <- function(x, tol) {
      if (length(x) == 0) return(0L)
      1L + sum(abs(diff(x)) > tol)
    }

    wv <- as.numeric(w)
    rv <- as.numeric(r)
    ok <- is.finite(wv) & is.finite(rv) & (wv > 0)
    n_eff <- sum(ok)
    if (is.null(df_null)) {
      # Approximate effective degrees of freedom under monotone-additive constraints:
      # alpha (1) + pooled levels in u and v, minus 2 sum-to-zero constraints.
      # If u has Ku distinct runs and v has Kv, then df ≈ (Ku - 1) + (Kv - 1) + 1 = Ku + Kv - 1.
      Ku <- n_distinct_runs(fit0_add$u, df_tol)
      Kv <- n_distinct_runs(fit0_add$v, df_tol)
      df_null <- Ku + Kv - 1L
    }
    df_null <- max(0, min(df_null, n_eff - 1))
    scale <- sqrt(n_eff / max(1, n_eff - df_null))
    r <- r * scale
  } else if (residual_scale == "hc2") {
    h <- compute_additive_leverage(w, nrow(barZ), ncol(barZ))
    r <- r / sqrt(pmax(1e-8, 1 - h))
  }
  sse_add_0 <- sse_weighted(as.numeric(y), as.numeric(theta_add), as.numeric(w))
  sse_iso_0 <- sse_weighted(as.numeric(y), as.numeric(fit0_iso$theta), as.numeric(w))
  delta_0 <- fit0_iso$theta - theta_add
  t0 <- interaction_stat(stat, delta_0, w, sse_add_0, sse_iso_0,
                         synergy_sign = synergy_sign, threshold = threshold)

  one_boot <- function(b) {
    set.seed(seed + b)
    xi <- matrix(sample(c(-1, 1), length(r), replace = TRUE), nrow = nrow(r), ncol = ncol(r))
    y_star <- theta_add + xi * r
    iso_b <- isotonic_2d_fit(y_star, w, direction = direction, osqp_pars = osqp_pars)
    add_b <- additive_ordered_fit(y_star, w, direction = direction, osqp_pars = osqp_pars)
    sse_iso_b <- sse_weighted(as.numeric(y_star), as.numeric(iso_b$theta), as.numeric(w))
    sse_add_b <- sse_weighted(as.numeric(y_star), as.numeric(add_b$theta), as.numeric(w))
    delta_b <- iso_b$theta - add_b$theta
    interaction_stat(stat, delta_b, w, sse_add_b, sse_iso_b,
                     synergy_sign = synergy_sign, threshold = threshold)
  }

  if (use_parallel && n_cores > 1) {
    if (!requireNamespace("future.apply", quietly = TRUE) || !requireNamespace("future", quietly = TRUE)) {
      stop("Packages 'future' and 'future.apply' are required for parallel bootstrap.")
    }
    oplan <- future::plan()
    on.exit(future::plan(oplan), add = TRUE)
    future::plan(future::multisession, workers = n_cores)
    t_star <- future.apply::future_sapply(seq_len(B), function(b) {
      tryCatch(one_boot(b), error = function(e) NA_real_)
    }, future.seed = TRUE)
  } else {
    t_star <- vapply(seq_len(B), function(b) {
      tryCatch(one_boot(b), error = function(e) NA_real_)
    }, numeric(1))
  }

  ok <- is.finite(t_star)
  t_star_ok <- t_star[ok]
  pval <- (1 + sum(t_star_ok >= t0)) / (length(t_star_ok) + 1)

  list(
    t0 = t0,
    t_star = t_star,
    p_value = pval,
    fit_add = fit0_add,
    fit_iso = fit0_iso,
    stat = stat,
    synergy_sign = synergy_sign,
    threshold = threshold,
    residual_scale = residual_scale,
    df_null = df_null
  )
}
