run_public_benchmarks <- function(cfg, transform, out_results, out_figures) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  mats <- load_matrices_from_sources(cfg)
  if (length(mats) == 0) {
    message("[", now_utc(), "] No public datasets configured; skipping public benchmarks.")
    return(invisible(NULL))
  }

  n_cores <- cfg$project$n_cores %||% 1

  compute_one <- function(df) {
    meta <- list(
      experiment_id = df$experiment_id[1],
      source = (df$source %||% NA_character_)[1],
      drugA = (df$drugA %||% NA_character_)[1],
      drugB = (df$drugB %||% NA_character_)[1],
      cell_line = (df$cell_line %||% NA_character_)[1]
    )
    key <- cache_key("public", meta$experiment_id, cfg$transform, cfg$weights, cfg$method, cfg$bootstrap, cfg$baselines)
    cached <- cache_get(cfg$project$cache_dir %||% "cache", key)
    if (!is.null(cached)) return(cached)

    prop <- compute_proposed_metrics(df, cfg, transform)
    base <- compute_baselines(df, cfg)

    interior_mean <- function(M) {
      if (!is.matrix(M)) return(NA_real_)
      if (nrow(M) < 2 || ncol(M) < 2) return(NA_real_)
      mean(M[-1, -1], na.rm = TRUE)
    }

    out <- data.table::data.table(
      experiment_id = meta$experiment_id,
      source = meta$source,
      drugA = meta$drugA,
      drugB = meta$drugB,
      cell_line = meta$cell_line,
      direction = prop$direction,
      t_int = prop$t_int,
      p_value = prop$p_value,
      S2 = prop$S2,
      Splus = prop$Splus,
      Sminus = prop$Sminus,
      max_synergy = prop$max_synergy,
      area_synergy = prop$area_synergy,
      area_synergy_weighted = prop$area_synergy_weighted,
      max_abs_delta = prop$max_abs_delta,
      mean_delta = prop$mean_delta,
      boot_stat = prop$boot_stat,
      synergy_sign = prop$synergy_sign,
      threshold = prop$threshold,
      bliss_mean = interior_mean(base$bliss$synergy),
      hsa_mean = interior_mean(base$hsa$synergy),
      loewe_mean = interior_mean(base$loewe$synergy),
      zip_mean = interior_mean(base$zip$synergy)
    )
    cache_set(cfg$project$cache_dir %||% "cache", key, out)
    out
  }

  if (n_cores > 1 && requireNamespace("future.apply", quietly = TRUE) && requireNamespace("future", quietly = TRUE)) {
    oplan <- future::plan()
    on.exit(future::plan(oplan), add = TRUE)
    future::plan(future::multisession, workers = n_cores)
    res <- future.apply::future_lapply(mats, compute_one, future.seed = TRUE)
  } else {
    res <- lapply(mats, compute_one)
  }

  dt <- data.table::rbindlist(res, fill = TRUE)
  write_table(dt, file.path(out_results, "public_metrics.parquet"))

  fig_paths <- plot_public_benchmarks(dt, out_dir = file.path(out_figures, "public"))

  invisible(list(results = dt, figures = fig_paths))
}
