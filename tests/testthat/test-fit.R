test_that("R reproduces every Python result field for S, P, replicate and missing inputs", {
  skip_without_python()
  py <- reference_python()
  nz <- sir_noise_parametric(sd = .04)
  Y <- simulated_plate()
  stack <- array(0, c(4, 5, 5))
  for (k in 1:4) stack[k, , ] <- simulated_plate(20L + k)
  stack[2, 3, 3] <- stack[2, 3, 3] + 1
  stack[1, 4, 4] <- NA_real_
  missing <- Y
  missing[2, 3] <- NA_real_
  missing_edge <- Y
  missing_edge[3, 1] <- NA_real_
  for (case in list(list(Y, "S"), list(Y, "P"), list(stack, "R"),
                   list(stack, "P"), list(missing, "S"), list(missing_edge, "S"))) {
    input <- case[[1]]
    design <- case[[2]]
    r <- sir_fit(input, list(0:4, 0:4), design = design, noise = nz, seed = 71, B = 399)
    input[is.na(input)] <- NaN
    p <- reference_list(py$fit(input, doses = list(as.numeric(0:4), as.numeric(0:4)),
                              design = design, noise = nz, seed = 71L, B = 399L))
    expect_identical(unclass(r)[names(p)], p)
    expect_identical(r$claims, Filter(function(z) isTRUE(z$claimed), p$regions))
  }
})

test_that("seeds reproduce complete fits and preserve the R random state", {
  skip_without_python()
  Y <- simulated_plate()
  Y[3:5, 3:5] <- Y[3:5, 3:5] - .3
  nz <- sir_noise_parametric(sd = .04)
  before <- .Random.seed
  a <- sir_fit(Y, noise = nz, seed = 42)
  b <- sir_fit(Y, noise = nz, seed = 42L)
  expect_identical(a, b)
  expect_identical(.Random.seed, before)
  expect_identical(a$stage, 2L)
  expect_identical(a$B_used, 9999L)
  expect_true(a$fragility$computed)
  p <- reference_list(reference_python()$fit(Y, noise = nz, seed = 42L))
  expect_identical(unclass(a)[names(p)], p)
  path <- tempfile(fileext = ".rds")
  saveRDS(a, path)
  expect_identical(readRDS(path), a)
})

test_that("result methods and missing-dose labels are usable", {
  skip_without_python()
  Y <- simulated_plate()
  Y[3, 1] <- NA_real_
  fit <- sir_fit(Y, doses = list(c(0, .1, .3, 1, 3), 0:4),
                 noise = sir_noise_parametric(), seed = 2)
  expect_s3_class(summary(fit), "summary.sir_fit")
  expect_output(print(fit), "Mean Bliss excess")
  path <- tempfile(fileext = ".pdf")
  grDevices::pdf(path)
  surfaces <- plot(fit)
  grDevices::dev.off()
  expect_equal(dim(surfaces$observed), c(4L, 5L))
  expect_equal(dim(surfaces$excess), c(3L, 4L))
  expect_identical(fit$log$rows_kept, c(0L, 1L, 3L, 4L))
  expect_equal(as.vector(fit$doses_a), c(0, .1, 1, 3))
})

test_that("invalid R inputs are rejected before entering kernels", {
  skip_without_python()
  nz <- sir_noise_parametric()
  Y <- simulated_plate()
  expect_error(sir_fit(Y, noise = nz, B = 2.5), "B must")
  expect_error(sir_fit(Y, noise = nz, seed = -1), "seed must")
  expect_error(sir_fit(Y, noise = nz, doses = list(1:5, 0:4)), "doses must")
  expect_error(sir_fit(Y, noise = nz, design = "R"), "replicate")
  expect_error(sir_fit(Y, noise = nz, normalize = "invalid"), "arg")
  Y[1, 1] <- 0
  expect_error(sir_fit(Y, noise = nz, normalize = "divide"), "nonpositive")
  expect_error(sir_noise_parametric(sd = -1), "sd must")
})
