test_that("standalone baselines use equivalent viability and inhibition inputs", {
  a <- c(1, .9, .7, .5, .3)
  b <- c(1, .8, .6, .4, .2)
  df <- expand.grid(doseA = 0:4, doseB = 0:4)
  df$response <- as.vector(outer(a, b))
  inhib <- df
  inhib$response <- 1 - inhib$response
  for (f in list(bliss_score, hsa_score, loewe_score, zip_score))
    expect_equal(f(df, "viability"), f(inhib, "inhibition"), tolerance = 1e-8)
  expect_equal(bliss_score(df, "viability")$synergy, matrix(0, 5, 5))
  expect_equal(zip_score(df, "viability", use_curvefit = FALSE)$synergy, matrix(0, 5, 5),
               tolerance = 1e-15)
})

test_that("weighted 2D isotonic fit is standalone and monotone", {
  set.seed(3)
  Y <- matrix(rnorm(20), 4, 5)
  f <- isotonic_2d_fit(Y, matrix(1, 4, 5))
  expect_true(all(f$theta[-1, ] <= f$theta[-4, ] + .002))
  expect_true(all(f$theta[, -1] <= f$theta[, -5] + .002))
})
