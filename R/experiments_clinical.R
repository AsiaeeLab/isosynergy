run_clinical_overlap <- function(cfg, out_results, out_figures) {
  clinical_csv <- cfg$clinical$drugpair_csv
  if (is.null(clinical_csv) || is.na(clinical_csv) || clinical_csv == "") {
    message("[", now_utc(), "] No clinical drug-pair CSV configured; skipping clinical overlap.")
    return(invisible(NULL))
  }
  metrics_path <- file.path(out_results, "public_metrics.parquet")
  if (!file.exists(metrics_path)) {
    message("[", now_utc(), "] Missing public metrics table; run with --steps public first.")
    return(invisible(NULL))
  }
  if (!requireNamespace("arrow", quietly = TRUE) || !requireNamespace("data.table", quietly = TRUE)) {
    stop("Packages 'arrow' and 'data.table' are required.")
  }

  metrics <- data.table::as.data.table(arrow::read_parquet(metrics_path))
  clin <- data.table::as.data.table(utils::read.csv(clinical_csv, stringsAsFactors = FALSE))
  if (!all(c("drugA", "drugB") %in% names(clin))) stop("Clinical CSV must contain columns: drugA, drugB")

  clin[, pair_key := paste(pmin(drugA, drugB), pmax(drugA, drugB), sep = "||")]
  metrics[, pair_key := paste(pmin(drugA, drugB), pmax(drugA, drugB), sep = "||")]
  metrics[, clinical := pair_key %in% clin$pair_key]

  write_table(metrics[, .(experiment_id, source, drugA, drugB, cell_line, clinical,
                          t_int, p_value, S2, Splus, Sminus,
                          bliss_mean, hsa_mean, loewe_mean, zip_mean)],
              file.path(out_results, "clinical_overlap_table.parquet"))

  fig_paths <- plot_clinical_overlap(metrics, out_dir = file.path(out_figures, "clinical"))
  invisible(list(table = metrics, figures = fig_paths))
}

