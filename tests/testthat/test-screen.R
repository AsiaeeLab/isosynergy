test_that("screen uses Python seeds and BH separately within families", {
  skip_without_python()
  nz <- sir_noise_parametric(sd = .04)
  panel <- data.frame(id = 1:8, family = rep(c("a", "b"), each = 4))
  panel$Y <- lapply(1:8, function(i) {
    Y <- simulated_plate(i)
    if (i %in% c(1, 2, 5)) Y[3:5, 3:5] <- Y[3:5, 3:5] - .25
    Y
  })
  out <- sir_screen(panel, noise = nz, seed = 34)
  expect_s3_class(out, "data.frame")
  for (f in unique(out$family)) {
    k <- out$family == f
    expect_equal(out$q[k], stats::p.adjust(out$p[k], "BH"), tolerance = 1e-15)
  }
  expect_identical(out$call, out$q <= .05)
  direct <- reference_python()$screen(panel$Y, family = as.list(panel$family),
                                      fit_kwargs = list(noise = nz), seed = 34L)
  expect_identical(out$p, as.vector(reticulate::py_to_r(direct$p)))
  expect_identical(out$q, as.vector(reticulate::py_to_r(direct$q)))
  expect_identical(out, sir_screen(panel, noise = nz, seed = 34))
  expect_identical(out$id, panel$id)
  forced <- sir_screen(panel[1, ], noise = nz, seed = 1, two_stage = FALSE)
  expect_true(forced$fit[[1]]$settings$two_stage)
  expect_match(unlist(attr(forced, "notes"))[1], "overridden")
})

test_that("screen propagates per-row doses and groups and handles one-row frames", {
  skip_without_python()
  panel <- data.frame(family = "one", group = "cellA")
  panel$Y <- list(simulated_plate())
  panel$doses <- list(list(0:4, c(0, .1, .3, 1, 3)))
  out <- sir_screen(panel, noise = sir_noise_parametric(), seed = 1)
  expect_identical(out$p, out$q)
  expect_identical(out$fit[[1]]$settings$group, "cellA")
  expect_equal(as.vector(out$fit[[1]]$doses_b), c(0, .1, .3, 1, 3))
  expect_error(sir_screen(panel, relative_null = TRUE), "requires")
})
