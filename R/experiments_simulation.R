run_simulation_benchmark <- function(cfg, transform, out_results, out_figures) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  mats <- simulate_matrices(cfg, transform = transform)

  osqp_pars <- cfg$method$osqp %||% list()
  n_cores <- cfg$project$n_cores %||% 1

  compute_one <- function(df) {
    key <- cache_key("sim", df$experiment_id[1], cfg$transform, cfg$weights, cfg$method, cfg$bootstrap)
    cached <- cache_get(cfg$project$cache_dir %||% "cache", key)
    if (!is.null(cached)) return(cached)

    prop <- compute_proposed_metrics(df, cfg, transform)
    out <- data.table::data.table(
      experiment_id = df$experiment_id[1],
      sim_interaction_strength = df$sim_interaction_strength[1],
      sim_interaction_mode = df$sim_interaction_mode[1],
      sim_synergy_sign = df$sim_synergy_sign[1],
      direction = prop$direction,
      t_int = prop$t_int,
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
      p_value = prop$p_value
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
  write_table(dt, file.path(out_results, "simulation_results.parquet"))

  fig_paths <- plot_simulation_benchmark(dt, out_dir = file.path(out_figures, "simulation"))
  invisible(list(results = dt, figures = fig_paths))
}
