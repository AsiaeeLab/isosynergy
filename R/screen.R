#' Screen Matrices with BH Adjustment within Declared Families
#'
#' Calls the bundled Python screen with independent per-row seeds and two-stage
#' Monte Carlo. Family labels define the multiple-testing families in advance.
#' @param data Data frame containing a list column of viability matrices or
#'   replicate arrays, and a family-label column. Optional list column `doses`
#'   and character column `group` supply per-row fit arguments.
#' @param matrix_col,family_col Names of the matrix and family columns.
#' @param q BH false discovery rate threshold.
#' @param test Adjust the primary SIR statistic (`"tstd"`), global scan (`"scan"`), or
#'   companion mean (`"mean"`) p-values.
#' @param relative_null For `test = "mean"` only, use the within-family robust
#'   empirical null before BH adjustment.
#' @param nan_policy Count missing p-values as one (default) or omit them;
#'   their own q-values remain missing and they are never called.
#' @param seed Nonnegative scalar integer. Python uses seeds `c(seed, row - 1)`;
#'   when NULL, the screen uses zero as its master seed.
#' @param ... Shared arguments to [sir_fit()], including the required `noise`.
#'   Per-row `doses` and `group` columns override shared arguments.
#'   `two_stage = FALSE` is overridden by the Python screen and logged.
#' @return A data frame preserving input columns, with `p`, `q`, `call`,
#'   `activity`, `mean_excess`, `mean_excess_se`, `p_mean`, `p_tstd`, `p_scan`,
#'   and a `fit` list column. Input columns with these output names are rejected.
#'   Attributes `notes`, `relative_null`, `test`, and `q_level` record screen
#'   settings. The `p` column contains unadjusted original p-values even when an
#'   empirical null is used. Calls need independent experimental confirmation.
#' @export
sir_screen <- function(data, matrix_col = "Y", family_col = "family", q = 0.05,
                       test = c("tstd", "scan", "mean"), relative_null = FALSE,
                       nan_policy = c("one", "omit"), seed = NULL, ...) {
  test <- match.arg(test)
  nan_policy <- match.arg(nan_policy)
  if (!is.data.frame(data) || !nrow(data) ||
      !all(c(matrix_col, family_col) %in% names(data)) || !is.list(data[[matrix_col]]))
    stop("data must have rows, a matrix list column and a family column.", call. = FALSE)
  if (anyNA(data[[family_col]])) stop("Family labels cannot be missing.", call. = FALSE)
  fields <- c("activity", "mean_excess", "mean_excess_se", "p_mean", "p_tstd", "p_scan")
  if (any(c("p", "q", "call", "fit", fields) %in% names(data)))
    stop("Input columns conflict with screen output names.", call. = FALSE)
  q <- .sir_number(q, "q", 0, 1)
  if (!is.null(seed)) seed <- .sir_number(seed, "seed", 0, .Machine$integer.max, TRUE)
  if (!is.logical(relative_null) || length(relative_null) != 1L || is.na(relative_null))
    stop("relative_null must be TRUE or FALSE.", call. = FALSE)
  if (relative_null && test != "mean") stop("relative_null requires test = 'mean'.", call. = FALSE)
  supplied <- list(...)
  if (length(supplied) && (is.null(names(supplied)) || any(!nzchar(names(supplied))) ||
      anyDuplicated(names(supplied)) || any(!names(supplied) %in% names(formals(sir_fit)))))
    stop("... must contain named sir_fit arguments.", call. = FALSE)
  defaults <- lapply(formals(sir_fit)[setdiff(names(formals(sir_fit)), c("Y", "noise"))], eval)
  kw <- utils::modifyList(defaults, supplied, keep.null = TRUE)
  items <- lapply(seq_len(nrow(data)), function(i) {
    a <- kw
    a$Y <- data[[matrix_col]][[i]]
    if ("doses" %in% names(data)) a$doses <- data$doses[[i]]
    if ("group" %in% names(data)) a$group <- data$group[i]
    a <- .sir_fit_args(a)
    a$seed <- NULL
    a$two_stage <- NULL
    a
  })
  result <- .sir_python()$screen(items, family = as.list(as.character(data[[family_col]])),
    q = q, test = test, relative_null = relative_null, nan_policy = nan_policy,
    fit_kwargs = list(two_stage = kw$two_stage), n_workers = 1L, seed = seed)
  fits <- lapply(reticulate::iterate(result$results, simplify = FALSE), .sir_result)
  out <- data
  for (nm in c("p", "q", "call")) out[[nm]] <- as.vector(reticulate::py_to_r(result[[nm]]))
  for (nm in fields) out[[nm]] <- vapply(fits, `[[`, numeric(1), nm)
  out$fit <- fits
  attr(out, "notes") <- reticulate::py_to_r(result$notes)
  attr(out, "relative_null") <- reticulate::py_to_r(result$relative_null)
  attr(out, "test") <- test
  attr(out, "q_level") <- q
  out
}
