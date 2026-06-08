#' Synergy via Isotonic Regression (SIR) Test
#'
#' Fits a monotone-additive null and a fully monotone two-dimensional
#' isotonic surface to a dose-response matrix, computes the interaction
#' surface as their difference, and returns a wild-bootstrap p-value
#' against the additive null.
#'
#' @param matrix Numeric matrix of mean responses on a transformed scale,
#'   with rows indexed by doses of drug A and columns by doses of drug B.
#'   Rows and columns should be ordered by increasing dose.
#' @param weights Numeric matrix of nonnegative weights of the same
#'   dimensions as `matrix`. Typical use is the inverse-variance weight
#'   `m_ij / s2_ij` produced by replicate-aggregation helpers. If
#'   `NULL`, all finite cells receive weight 1.
#' @param B Integer number of wild-bootstrap resamples. Defaults to 200.
#' @param direction Direction of monotonicity in dose: `"decreasing"` for
#'   viability data (default), `"increasing"` for inhibition data.
#' @param synergy_sign Direction of synergy (signed sense of `delta`):
#'   `"negative"` (default) when more synergy means a more negative
#'   `delta`, as is conventional for viability data; `"positive"` for
#'   inhibition data.
#' @param stat Test statistic used for the bootstrap p-value. See
#'   [wild_bootstrap()] for allowed values. Defaults to `"S2"`.
#' @param residual_scale Residual scaling used inside the wild bootstrap.
#'   See [wild_bootstrap()]. Defaults to `"none"`.
#' @param osqp_pars Named list of OSQP solver parameters (see
#'   [osqp::osqpSettings()]).
#' @param seed Integer seed for the bootstrap.
#' @param n_cores Integer number of worker processes to use for the
#'   bootstrap. `1` (default) runs serially.
#' @param use_parallel Logical: when `TRUE` and `n_cores > 1`, the
#'   bootstrap is parallelised via the future framework.
#' @param threshold Threshold for the `area_synergy` statistic.
#' @param ... Additional arguments forwarded to [wild_bootstrap()].
#'
#' @return An object of class `"sir_test"`: a list with components
#'   `delta` (interaction surface, isotonic minus additive),
#'   `theta_iso` (fully monotone fit), `theta_add` (monotone-additive
#'   fit), `S2` (interaction energy), `p_value` (wild-bootstrap p-value),
#'   `t0` (observed test statistic), `bootstrap_samples` (vector of
#'   bootstrap statistics), `direction`, `stat`, `synergy_sign`,
#'   and `B`.
#'
#' @seealso [additive_ordered_fit()], [isotonic_2d_fit()],
#'   [wild_bootstrap()], [interaction_summaries()].
#'
#' @examples
#' set.seed(1)
#' I <- 6; J <- 6
#' u <- -cumsum(abs(rnorm(I, 0.3, 0.1))); u <- u - mean(u)
#' v <- -cumsum(abs(rnorm(J, 0.3, 0.1))); v <- v - mean(v)
#' Z_add <- outer(u, v, "+")
#' Z <- Z_add + 0.05 * matrix(rnorm(I * J), I, J)
#' fit <- sir_test(Z, B = 50, direction = "decreasing")
#' fit$S2
#' fit$p_value
#'
#' @export
sir_test <- function(matrix,
                     weights = NULL,
                     B = 200,
                     direction = c("decreasing", "increasing"),
                     synergy_sign = c("negative", "positive"),
                     stat = "S2",
                     residual_scale = c("none", "df", "hc2"),
                     osqp_pars = list(),
                     seed = 1L,
                     n_cores = 1L,
                     use_parallel = FALSE,
                     threshold = 0,
                     ...) {
  if (!is.matrix(matrix) || !is.numeric(matrix)) {
    stop("`matrix` must be a numeric matrix.")
  }
  direction <- match.arg(direction)
  synergy_sign <- match.arg(synergy_sign)
  residual_scale <- match.arg(residual_scale)

  if (is.null(weights)) {
    weights <- matrix(1, nrow = nrow(matrix), ncol = ncol(matrix))
    weights[!is.finite(matrix)] <- 0
  }
  if (!identical(dim(weights), dim(matrix))) {
    stop("`weights` must have the same dimensions as `matrix`.")
  }

  fit <- interaction_fit(matrix, weights, direction = direction,
                         osqp_pars = osqp_pars,
                         synergy_sign = synergy_sign,
                         threshold = threshold)

  boot <- wild_bootstrap(matrix, weights, direction = direction, B = B,
                         seed = seed, osqp_pars = osqp_pars,
                         n_cores = n_cores, use_parallel = use_parallel,
                         stat = stat, synergy_sign = synergy_sign,
                         threshold = threshold,
                         residual_scale = residual_scale, ...)

  out <- list(
    delta = fit$delta,
    theta_iso = fit$theta_iso,
    theta_add = fit$theta_add,
    S2 = fit$S2,
    Splus = fit$Splus,
    Sminus = fit$Sminus,
    S2_plus = fit$S2_plus,
    S2_minus = fit$S2_minus,
    synergy_index = fit$synergy_index,
    max_synergy = fit$max_synergy,
    max_antagonism = fit$max_antagonism,
    t_int = fit$t_int,
    t0 = boot$t0,
    p_value = boot$p_value,
    bootstrap_samples = boot$t_star,
    direction = direction,
    synergy_sign = synergy_sign,
    stat = stat,
    residual_scale = residual_scale,
    B = B,
    threshold = threshold
  )
  class(out) <- "sir_test"
  out
}

#' @export
print.sir_test <- function(x, ...) {
  cat("SIR (Synergy via Isotonic Regression) test\n")
  cat(sprintf("  Grid:           %d x %d\n", nrow(x$delta), ncol(x$delta)))
  cat(sprintf("  Direction:      %s (synergy = %s)\n", x$direction, x$synergy_sign))
  cat(sprintf("  Statistic:      %s\n", x$stat))
  cat(sprintf("  S2 (energy):    %.4g\n", x$S2))
  cat(sprintf("  Observed stat:  %.4g\n", x$t0))
  cat(sprintf("  Bootstrap B:    %d\n", x$B))
  cat(sprintf("  p-value:        %.4f\n", x$p_value))
  invisible(x)
}

#' @export
summary.sir_test <- function(object, ...) {
  out <- list(
    grid = dim(object$delta),
    direction = object$direction,
    synergy_sign = object$synergy_sign,
    stat = object$stat,
    residual_scale = object$residual_scale,
    B = object$B,
    S2 = object$S2,
    Splus = object$Splus,
    Sminus = object$Sminus,
    S2_plus = object$S2_plus,
    S2_minus = object$S2_minus,
    synergy_index = object$synergy_index,
    max_synergy = object$max_synergy,
    max_antagonism = object$max_antagonism,
    t0 = object$t0,
    p_value = object$p_value
  )
  class(out) <- "summary.sir_test"
  out
}

#' @export
print.summary.sir_test <- function(x, ...) {
  cat("SIR test summary\n")
  cat(sprintf("  Grid:            %d x %d\n", x$grid[1], x$grid[2]))
  cat(sprintf("  Direction:       %s\n", x$direction))
  cat(sprintf("  Synergy sign:    %s\n", x$synergy_sign))
  cat(sprintf("  Test statistic:  %s (residual scale: %s)\n", x$stat, x$residual_scale))
  cat(sprintf("  Bootstrap B:     %d\n", x$B))
  cat("\nInteraction energies (weighted):\n")
  cat(sprintf("  S2:              %.4g\n", x$S2))
  cat(sprintf("  S2_plus:         %.4g\n", x$S2_plus))
  cat(sprintf("  S2_minus:        %.4g\n", x$S2_minus))
  cat(sprintf("  Synergy index:   %.3f\n", x$synergy_index))
  cat(sprintf("  Max synergy:     %.4g\n", x$max_synergy))
  cat(sprintf("  Max antagonism:  %.4g\n", x$max_antagonism))
  cat(sprintf("\nObserved stat:    %.4g\n", x$t0))
  cat(sprintf("p-value:           %.4f\n", x$p_value))
  invisible(x)
}

#' Plot a SIR Test Result
#'
#' Produces a heatmap of the interaction surface `delta = theta_iso -
#' theta_add`. By default uses base graphics (`graphics::image`); when
#' `use_ggplot = TRUE` and `ggplot2` is installed, returns a `ggplot`.
#'
#' @param x A `sir_test` object returned by [sir_test()].
#' @param use_ggplot Logical: use `ggplot2` if available.
#' @param ... Additional arguments passed to `graphics::image()` (when
#'   not using ggplot).
#' @return When `use_ggplot = TRUE` and ggplot2 is available, a ggplot
#'   object is returned invisibly. Otherwise the function is called
#'   for its side effect.
#' @export
plot.sir_test <- function(x, use_ggplot = FALSE, ...) {
  delta <- x$delta
  if (isTRUE(use_ggplot) && requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      i = rep(seq_len(nrow(delta)), times = ncol(delta)),
      j = rep(seq_len(ncol(delta)), each = nrow(delta)),
      delta = as.numeric(delta)
    )
    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$j, y = .data$i, fill = .data$delta)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2() +
      ggplot2::labs(x = "Dose B index", y = "Dose A index",
                    fill = expression(delta),
                    title = sprintf("SIR interaction surface (p = %.3f)", x$p_value)) +
      ggplot2::theme_minimal()
    print(p)
    return(invisible(p))
  }
  graphics::image(seq_len(nrow(delta)), seq_len(ncol(delta)), delta,
                  xlab = "Dose A index", ylab = "Dose B index",
                  main = sprintf("delta (p = %.3f)", x$p_value),
                  ...)
  invisible(NULL)
}
