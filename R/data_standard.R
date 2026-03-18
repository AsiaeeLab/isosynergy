load_standard_long <- function(path) {
  ext <- tools::file_ext(path)
  if (ext %in% c("parquet", "pq")) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("Package 'arrow' required to read parquet. Install with: install.packages('arrow')")
    }
    df <- arrow::read_parquet(path)
  } else if (ext %in% c("csv")) {
    df <- utils::read.csv(path, stringsAsFactors = FALSE)
  } else {
    stop("Unsupported extension: ", ext)
  }
  df
}

split_long_by_experiment <- function(df) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dt <- data.table::as.data.table(df)
  req <- c("experiment_id", "drugA", "drugB", "cell_line", "doseA", "doseB", "response", "replicate")
  missing <- setdiff(req, names(dt))
  if (length(missing) > 0) stop("Standard long format missing: ", paste(missing, collapse = ", "))
  split(dt, by = "experiment_id", keep.by = TRUE)
}

load_matrices_from_sources <- function(cfg) {
  sources <- cfg$data$sources %||% character()
  out <- list()
  std <- cfg$data$standard_files %||% list()

  for (src in sources) {
    if (src == "simulation") next
    path <- std[[src]]
    if (is.null(path) || is.na(path) || identical(path, "")) next
    df <- load_standard_long(path)
    mats <- split_long_by_experiment(df)
    out <- c(out, mats)
  }

  out
}

reshape_long_to_matrix <- function(df_long, value_col = "response", fill = NA_real_) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dt <- data.table::as.data.table(df_long)
  if (!value_col %in% names(dt)) {
    stop("Column not found in long table: ", value_col)
  }
  dt[, `:=`(doseA = as.numeric(doseA), doseB = as.numeric(doseB))]
  doseA_levels <- sort(unique(dt$doseA))
  doseB_levels <- sort(unique(dt$doseB))
  dt[, i := match(doseA, doseA_levels)]
  dt[, j := match(doseB, doseB_levels)]
  mat <- matrix(fill, nrow = length(doseA_levels), ncol = length(doseB_levels))
  vals <- dt[is.finite(get(value_col))]
  mat[cbind(vals$i, vals$j)] <- vals[[value_col]]
  list(
    matrix = mat,
    doseA_levels = doseA_levels,
    doseB_levels = doseB_levels,
    has_monoA = any(dt$doseB == 0, na.rm = TRUE),
    has_monoB = any(dt$doseA == 0, na.rm = TRUE),
    n_obs = nrow(vals)
  )
}
