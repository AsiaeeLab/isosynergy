test_that("parametric and SD-curve constructors preserve the Python models", {
  skip_without_python()
  curve <- sir_sd_curve("table", knots = c(0, .5, 1), sds = c(.02, .05, .08))
  nz <- sir_noise_parametric("table", curve = curve, plate_shift_ratio = .25)
  direct <- reference_python()$ParametricNoise$sd_table(curve, plate_shift_ratio = .25)
  expect_identical(reticulate::py_to_r(nz$describe()), reticulate::py_to_r(direct$describe()))
  Y <- simulated_plate()
  r <- sir_fit(Y, design = "P", noise = nz, sa_replicates = 4, seed = 6)
  p <- reference_list(reference_python()$fit(Y, design = "P", noise = direct,
                                            sa_replicates = 4L, seed = 6L))
  expect_identical(unclass(r)[names(p)], p)
})

test_that("replicate banks, estimated curves and plate variance work end to end", {
  skip_without_python()
  set.seed(6)
  donor <- array(0, c(80, 4, 5, 5))
  mu <- outer(c(1, .9, .7, .5, .3), c(1, .9, .7, .5, .3))
  for (s in 1:80) for (k in 1:4) {
    donor[s, k, , ] <- mu + matrix(rnorm(25, sd = .04), 5)
    donor[s, k, 1, 1] <- 1
  }
  groups <- rep(paste0("g", 1:8), each = 10)
  curve <- sir_sd_from_replicates(donor, n_bins = 4)
  expect_s3_class(curve, "sirsyn.noise.SDCurve")
  bank <- sir_noise_bank(donor, groups, curve = curve, include_edges = TRUE, n_strata = 1)
  variance <- sir_plate_variance(donor, groups)
  r <- sir_fit(donor[1, 1, , ], design = "P", noise = bank, group = "g1",
               min_bank_pool = 5, plate_var = variance, seed = 2)
  p <- reference_list(reference_python()$fit(donor[1, 1, , ], design = "P", noise = bank,
           group = "g1", min_bank_pool = 5L, plate_var = variance, seed = 2L))
  expect_identical(unclass(r)[names(p)], p)
  expect_match(paste(unlist(r$log$notes), collapse = " "), "noise bank")
})

test_that("Python help is non-installing and respects the interpreter environment", {
  expect_output(sir_python_install_help("python3"), "pip install")
  skip_without_python()
  cfg <- sir_python_config()
  expect_true(file.exists(cfg$python))
  if (nzchar(Sys.getenv("RETICULATE_PYTHON")))
    expect_identical(normalizePath(cfg$python), normalizePath(Sys.getenv("RETICULATE_PYTHON")))
})

test_that("design P warns about exact curves and a missing plate shift", {
  skip_without_python()
  set.seed(4)
  a <- c(1, 0.9, 0.7, 0.5, 0.3)
  Y <- outer(a, a) + matrix(rnorm(25, sd = 0.03), 5)
  Y[1, ] <- a; Y[, 1] <- a; Y[1, 1] <- 1
  expect_warning(sir_fit(Y, design = "P", noise = sir_noise_parametric(sd = 0.03, plate_shift_ratio = 0.25),
                         seed = 1, B = 39L, n_std = 9L, two_stage = FALSE), "sa_replicates not given")
  expect_warning(sir_fit(Y, design = "P", noise = sir_noise_parametric(sd = 0.03), sa_replicates = Inf,
                         seed = 1, B = 39L, n_std = 9L, two_stage = FALSE), "no plate shift")
  expect_no_warning(sir_fit(Y, design = "P", noise = sir_noise_parametric(sd = 0.03, plate_shift_ratio = 0.25),
                            sa_replicates = Inf, seed = 1, B = 39L, n_std = 9L, two_stage = FALSE))
})

test_that("sir_check_noise compares replicate spread with the curve", {
  skip_without_python()
  set.seed(5)
  Y <- array(rnorm(200 * 3 * 5 * 5, 0.5, 0.05), c(200, 3, 5, 5))
  Y[, , 1, 1] <- 1
  chk <- sir_check_noise(Y, sir_noise_parametric(sd = 0.05))
  expect_s3_class(chk, "data.frame")
  expect_true(all(abs(chk$ratio - 1) < 0.1))
  chk2 <- sir_check_noise(Y, sir_noise_parametric(sd = 0.1))
  expect_true(all(chk2$ratio < 0.6))
})
