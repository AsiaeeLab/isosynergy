require_osqp <- function() {
  if (!requireNamespace("osqp", quietly = TRUE)) {
    stop("Package 'osqp' is required. Install with: install.packages('osqp')")
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required. Install with: install.packages('Matrix')")
  }
  invisible(TRUE)
}

as_sparse <- function(x) {
  if (inherits(x, "sparseMatrix")) return(x)
  Matrix::Matrix(x, sparse = TRUE)
}

solve_osqp <- function(P, q, A, l, u, pars = list()) {
  require_osqp()
  if (is.null(pars$verbose)) pars$verbose <- FALSE
  model <- osqp::osqp(
    P = as_sparse(P),
    q = as.numeric(q),
    A = as_sparse(A),
    l = as.numeric(l),
    u = as.numeric(u),
    pars = pars
  )
  res <- model$Solve()
  if (!is.null(res$info$status_val) && res$info$status_val %in% c(1L, 2L)) {
    return(res)
  }
  msg <- paste0("OSQP failed: status=", res$info$status, " (", res$info$status_val, ")")
  stop(msg)
}

sse_weighted <- function(y, theta, w) {
  ok <- is.finite(y) & is.finite(theta) & is.finite(w) & (w > 0)
  sum(w[ok] * (y[ok] - theta[ok])^2)
}
