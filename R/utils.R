dir_create <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

now_utc <- function() format(Sys.time(), tz = "UTC", usetz = TRUE)

clamp01 <- function(x, eps = 0) pmin(1 - eps, pmax(eps, x))

write_table <- function(df, path) {
  ext <- tools::file_ext(path)
  dir_create(dirname(path))
  if (ext %in% c("parquet", "pq")) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("Package 'arrow' required to write parquet. Install with: install.packages('arrow')")
    }
    arrow::write_parquet(df, path)
  } else if (ext %in% c("csv")) {
    utils::write.csv(df, path, row.names = FALSE)
  } else {
    stop("Unsupported output extension: ", ext)
  }
  invisible(path)
}

with_seed <- function(seed, expr) {
  old <- .Random.seed
  on.exit({
    if (exists("old", inherits = FALSE)) .Random.seed <<- old
  }, add = TRUE)
  set.seed(seed)
  force(expr)
}
