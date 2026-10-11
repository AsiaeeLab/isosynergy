.sir_env <- new.env(parent = emptyenv())

.sir_python <- function() {
  if (is.null(.sir_env$module)) {
    path <- system.file("python", package = "SIR")
    if (!nzchar(path)) stop("SIR's bundled Python package is missing. Reinstall SIR.", call. = FALSE)
    module <- tryCatch(
      reticulate::import_from_path("sirsyn", path = path, convert = FALSE),
      error = function(e) stop("Cannot load sirsyn: ", conditionMessage(e),
        "\nUse sir_python_config() and sir_python_install_help().", call. = FALSE)
    )
    module_file <- reticulate::py_to_r(module$`__file__`)
    if (normalizePath(dirname(dirname(module_file))) != normalizePath(path))
      stop("A different sirsyn is already loaded in Python. Restart R and load SIR first.",
           call. = FALSE)
    .sir_env$module <- module
  }
  .sir_env$module
}

#' Configure the Python Interpreter
#'
#' Select Python before constructing a noise model or fitting data. Loading SIR
#' does not initialize Python. An existing `RETICULATE_PYTHON` setting takes
#' precedence; a conflicting explicit path is rejected. An initialized Python
#' cannot be changed without restarting R.
#' @param python Optional path to a Python executable.
#' @return The configuration returned by [reticulate::py_config()].
#' @export
sir_python_config <- function(python = NULL) {
  if (!is.null(python)) {
    if (!is.character(python) || length(python) != 1L || is.na(python) || !file.exists(python))
      stop("python must name an existing executable.", call. = FALSE)
    env <- Sys.getenv("RETICULATE_PYTHON", "")
    if (nzchar(env) && normalizePath(env, mustWork = FALSE) != normalizePath(python))
      stop("RETICULATE_PYTHON already selects a different Python; change it before starting R.", call. = FALSE)
    if (reticulate::py_available(initialize = FALSE) &&
        normalizePath(reticulate::py_config()$python) != normalizePath(python))
      stop("Python is already initialized; restart R to change it.", call. = FALSE)
    reticulate::use_python(python, required = TRUE)
  }
  reticulate::py_config()
}

#' Show Python Dependency Installation Instructions
#'
#' Prints commands only. No software is installed and Python is not initialized.
#' @param python Python command or executable path. Defaults to
#'   `RETICULATE_PYTHON`, or `"python3"` when unset.
#' @return The instructions, invisibly as a character vector.
#' @export
sir_python_install_help <- function(python = Sys.getenv("RETICULATE_PYTHON", "python3")) {
  lines <- c("SIR needs Python >= 3.9 with numpy, scipy and numba.",
    paste(shQuote(python), "-m pip install 'numpy>=1.22' 'scipy>=1.8' 'numba>=0.56'"),
    "Alternatively create a virtual environment and install those packages there.",
    "Set RETICULATE_PYTHON to its executable before starting R, or call",
    "sir_python_config('/path/to/python') before using SIR.",
    "sirsyn is bundled with SIR and does not need a separate pip install.")
  cat(paste(lines, collapse = "\n"), "\n")
  invisible(lines)
}

.sir_number <- function(x, name, lower = 0, upper = Inf, integer = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x < lower || x > upper || (integer && x != floor(x)))
    stop(name, " must be a finite ", if (integer) "integer " else "number ",
         "in [", lower, ", ", upper, "].", call. = FALSE)
  if (integer) as.integer(x) else as.numeric(x)
}

.sir_array <- function(Y, ndim = c(2L, 3L)) {
  if (!is.numeric(Y) || !length(dim(Y)) || !length(dim(Y)) %in% ndim ||
      any(dim(Y) < 1L) || any(is.infinite(Y)))
    stop("Y must be a numeric array with ", paste(ndim, collapse = " or "),
         " dimensions; use NA for missing wells.", call. = FALSE)
  storage.mode(Y) <- "double"
  Y[is.na(Y)] <- NaN
  Y
}
