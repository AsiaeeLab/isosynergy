skip_if_not_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip(paste("Missing package", pkg))
  }
}

# Internal helpers — when running under the installed package we reach
# them via SIR:::; when running as a fresh-clone source-and-test fallback
# (see tests/testthat/helper-source.R) the same names are present in
# the global env, so a small dispatcher keeps both layouts working.
sir_internal <- function(name) {
  if (requireNamespace("SIR", quietly = TRUE)) {
    return(get(name, envir = asNamespace("SIR")))
  }
  get(name, envir = parent.frame())
}

testthat::test_that("isotonic_2d_fit enforces monotonicity", {
  skip_if_not_installed("osqp")
  skip_if_not_installed("Matrix")

  check_monotone_2d <- sir_internal("check_monotone_2d")

  set.seed(1)
  I <- 6; J <- 7
  y <- matrix(rnorm(I * J), I, J)
  w <- matrix(1, I, J)
  fit <- isotonic_2d_fit(y, w, direction = "decreasing", osqp_pars = list(eps_abs = 1e-7, eps_rel = 1e-7))
  testthat::expect_true(check_monotone_2d(fit$theta, direction = "decreasing", tol = 1e-6))
})

testthat::test_that("additive_ordered_fit enforces monotone u,v and identifiability", {
  skip_if_not_installed("osqp")
  skip_if_not_installed("Matrix")

  check_monotone_1d <- sir_internal("check_monotone_1d")

  set.seed(2)
  I <- 8; J <- 5
  y <- matrix(rnorm(I * J), I, J)
  w <- matrix(1, I, J)
  fit <- additive_ordered_fit(y, w, direction = "decreasing", osqp_pars = list(eps_abs = 1e-7, eps_rel = 1e-7))
  testthat::expect_true(check_monotone_1d(fit$u, direction = "decreasing", tol = 1e-6))
  testthat::expect_true(check_monotone_1d(fit$v, direction = "decreasing", tol = 1e-6))
  testthat::expect_lt(abs(sum(fit$u)), 1e-5)
  testthat::expect_lt(abs(sum(fit$v)), 1e-5)
})

testthat::test_that("wild bootstrap is roughly calibrated under additive null (smoke)", {
  skip_if_not_installed("osqp")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("data.table")

  make_transform     <- sir_internal("make_transform")
  simulate_one_matrix <- sir_internal("simulate_one_matrix")
  summarise_matrix   <- sir_internal("summarise_matrix")

  cfg <- list(
    project = list(seed = 1, n_cores = 1),
    responses = list(mode = "viability"),
    transform = list(name = "logit", eps = 1e-6),
    weights = list(tau = 1e-6, winsor_q = 0.99),
    method = list(monotone_direction = "auto", osqp = list(eps_abs = 1e-7, eps_rel = 1e-7, max_iter = 20000)),
    bootstrap = list(B = 50, seed = 1)
  )
  tr <- make_transform(cfg$transform)
  df <- simulate_one_matrix("sim_null", I = 6, J = 6, interaction_strength = 0,
                            noise_sigma = 0.05, heavy_tail_df = 10,
                            missing_rate = 0, outlier_rate = 0,
                            transform = tr, response_mode = cfg$responses$mode)
  ms <- summarise_matrix(df, transform = tr, response_mode = cfg$responses$mode,
                         method_direction = cfg$method$monotone_direction,
                         tau = cfg$weights$tau, winsor_q = cfg$weights$winsor_q)
  boot <- wild_bootstrap(ms$barZ, ms$w, direction = ms$direction, B = cfg$bootstrap$B,
                         seed = cfg$bootstrap$seed, osqp_pars = cfg$method$osqp,
                         n_cores = 1, use_parallel = FALSE)
  testthat::expect_true(is.finite(boot$p_value))
  testthat::expect_true(boot$p_value >= 0 && boot$p_value <= 1)
})

testthat::test_that("sir_test returns a valid sir_test object", {
  skip_if_not_installed("osqp")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("SIR")

  set.seed(3)
  I <- 6L; J <- 6L
  u <- -cumsum(abs(rnorm(I, 0.3, 0.1))); u <- u - mean(u)
  v <- -cumsum(abs(rnorm(J, 0.3, 0.1))); v <- v - mean(v)
  Z <- outer(u, v, "+") + 0.05 * matrix(rnorm(I * J), I, J)

  fit <- sir_test(Z, B = 30, direction = "decreasing", seed = 1)
  testthat::expect_s3_class(fit, "sir_test")
  testthat::expect_equal(dim(fit$delta), c(I, J))
  testthat::expect_true(is.finite(fit$S2))
  testthat::expect_true(fit$p_value >= 0 && fit$p_value <= 1)
  testthat::expect_length(fit$bootstrap_samples, 30)
})
