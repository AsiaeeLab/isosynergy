additive_ordered_fit <- function(barZ, w, direction = c("decreasing", "increasing"),
                                 osqp_pars = list(), ridge = 1e-9) {
  direction <- match.arg(direction)
  require_osqp()

  I <- nrow(barZ)
  J <- ncol(barZ)
  n <- I * J

  y <- as.numeric(barZ)
  wv <- as.numeric(w)
  wv[!is.finite(wv) | wv < 0] <- 0
  y[!is.finite(y)] <- 0

  p <- 1L + I + J
  # Sparse design matrix with 3 non-zeros per row: alpha, u_i, v_j
  row_idx <- seq_len(n)
  i_idx <- rep(seq_len(I), times = J)
  j_idx <- rep(seq_len(J), each = I)

  X <- Matrix::sparseMatrix(
    i = rep(row_idx, times = 3L),
    j = c(
      rep(1L, n),                # alpha
      1L + i_idx,                # u_i
      1L + I + j_idx             # v_j
    ),
    x = 1,
    dims = c(n, p)
  )

  W <- Matrix::Diagonal(n, wv)
  P <- 2 * (Matrix::t(X) %*% W %*% X) + 2 * ridge * Matrix::Diagonal(p, 1)
  q <- -2 * as.numeric(Matrix::t(X) %*% (wv * y))

  # Constraints: monotone u, monotone v, sum(u)=0, sum(v)=0
  m_u <- I - 1L
  m_v <- J - 1L
  m <- m_u + m_v + 2L

  ii <- integer((m_u + m_v) * 2L + I + J)
  jj <- integer((m_u + m_v) * 2L + I + J)
  xx <- numeric((m_u + m_v) * 2L + I + J)
  row <- 0L
  k <- 0L

  for (i in seq_len(I - 1L)) {
    row <- row + 1L
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + (i + 1L); xx[k] <- 1
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + i; xx[k] <- -1
  }
  for (j in seq_len(J - 1L)) {
    row <- row + 1L
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + I + (j + 1L); xx[k] <- 1
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + I + j; xx[k] <- -1
  }

  # sum(u)=0
  row <- row + 1L
  for (i in seq_len(I)) {
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + i; xx[k] <- 1
  }
  # sum(v)=0
  row <- row + 1L
  for (j in seq_len(J)) {
    k <- k + 1L
    ii[k] <- row; jj[k] <- 1L + I + j; xx[k] <- 1
  }

  A <- Matrix::sparseMatrix(i = ii, j = jj, x = xx, dims = c(m, p))

  l <- rep(-Inf, m)
  u <- rep(Inf, m)
  if (direction == "decreasing") {
    l[seq_len(m_u + m_v)] <- -Inf
    u[seq_len(m_u + m_v)] <- 0
  } else {
    l[seq_len(m_u + m_v)] <- 0
    u[seq_len(m_u + m_v)] <- Inf
  }
  # equalities for sums (last two rows)
  l[(m_u + m_v + 1L):m] <- 0
  u[(m_u + m_v + 1L):m] <- 0

  res <- solve_osqp(P, q, A, l, u, pars = osqp_pars)
  x <- res$x
  alpha <- x[1]
  u_vec <- x[2:(1 + I)]
  v_vec <- x[(2 + I):(1 + I + J)]

  theta <- matrix(alpha, nrow = I, ncol = J) +
    matrix(u_vec, nrow = I, ncol = J) +
    matrix(v_vec, nrow = I, ncol = J, byrow = TRUE)

  list(
    theta = theta,
    alpha = alpha,
    u = u_vec,
    v = v_vec,
    status = res$info$status,
    objective = res$info$obj_val
  )
}

check_monotone_1d <- function(x, direction = c("decreasing", "increasing"), tol = 1e-7) {
  direction <- match.arg(direction)
  d <- diff(x)
  if (direction == "decreasing") all(d <= tol) else all(d >= -tol)
}

