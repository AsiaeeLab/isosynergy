#' Two-Dimensional Isotonic Regression Fit
#'
#' Fits a fully monotone (in both row and column dimensions) surface to a
#' weighted dose-response matrix, by solving a sparse quadratic program
#' via [osqp::osqp()].
#'
#' @param barZ Numeric matrix of (transformed) mean responses.
#' @param w Numeric matrix of nonnegative weights, same dimensions as
#'   `barZ`.
#' @param direction Direction of monotonicity: `"decreasing"` (default)
#'   for viability data, `"increasing"` for inhibition data.
#' @param osqp_pars Named list of OSQP solver parameters.
#' @param ridge Small ridge added to the diagonal of the QP for numerical
#'   stability.
#'
#' @return A list with components `theta` (the fitted I-by-J surface),
#'   `status` (OSQP solver status), and `objective` (final objective
#'   value).
#'
#' @seealso [additive_ordered_fit()], [sir_test()].
#' @export
isotonic_2d_fit <- function(barZ, w, direction = c("decreasing", "increasing"),
                            osqp_pars = list(), ridge = 1e-9) {
  direction <- match.arg(direction)
  require_osqp()

  I <- nrow(barZ)
  J <- ncol(barZ)
  n <- I * J
  idx <- function(i, j) (j - 1L) * I + i

  y <- as.numeric(barZ)
  wv <- as.numeric(w)
  wv[!is.finite(wv) | wv < 0] <- 0
  y[!is.finite(y)] <- 0

  P <- 2 * Matrix::Diagonal(n, wv + ridge)
  q <- -2 * wv * y

  n_row <- (I - 1L) * J
  n_col <- I * (J - 1L)
  m <- n_row + n_col

  ii <- integer(m * 2L)
  jj <- integer(m * 2L)
  xx <- numeric(m * 2L)
  row <- 0L
  k <- 0L

  for (j in seq_len(J)) {
    for (i in seq_len(I - 1L)) {
      row <- row + 1L
      k <- k + 1L
      ii[k] <- row; jj[k] <- idx(i + 1L, j); xx[k] <- 1
      k <- k + 1L
      ii[k] <- row; jj[k] <- idx(i, j); xx[k] <- -1
    }
  }
  for (j in seq_len(J - 1L)) {
    for (i in seq_len(I)) {
      row <- row + 1L
      k <- k + 1L
      ii[k] <- row; jj[k] <- idx(i, j + 1L); xx[k] <- 1
      k <- k + 1L
      ii[k] <- row; jj[k] <- idx(i, j); xx[k] <- -1
    }
  }

  A <- Matrix::sparseMatrix(i = ii, j = jj, x = xx, dims = c(m, n))
  if (direction == "decreasing") {
    l <- rep(-Inf, m)
    u <- rep(0, m)
  } else {
    l <- rep(0, m)
    u <- rep(Inf, m)
  }

  res <- solve_osqp(P, q, A, l, u, pars = osqp_pars)
  theta <- matrix(res$x, nrow = I, ncol = J)

  list(
    theta = theta,
    status = res$info$status,
    objective = res$info$obj_val
  )
}

check_monotone_2d <- function(theta, direction = c("decreasing", "increasing"), tol = 1e-7) {
  direction <- match.arg(direction)
  dx_i <- theta[-1, , drop = FALSE] - theta[-nrow(theta), , drop = FALSE]
  dx_j <- theta[, -1, drop = FALSE] - theta[, -ncol(theta), drop = FALSE]
  if (direction == "decreasing") {
    ok <- all(dx_i <= tol) && all(dx_j <= tol)
  } else {
    ok <- all(dx_i >= -tol) && all(dx_j >= -tol)
  }
  ok
}

