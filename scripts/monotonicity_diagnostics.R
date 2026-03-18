#!/usr/bin/env Rscript

get_script_path <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)
  if (length(file_arg) == 0) return(NULL)
  sub("^--file=", "", file_arg[1])
}

script_path <- get_script_path()
if (!is.null(script_path)) {
  root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
  if (file.exists(file.path(root, "renv.lock"))) setwd(root)
}

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(arrow)
  library(ggplot2)
  library(future)
  library(future.apply)
  library(patchwork)
})

source("R/utils.R")
source("R/config.R")
source("R/transforms.R")
source("R/matrix_stats.R")
source("R/monotonicity.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--source", type = "character", default = "drugcombdb",
              help = "Dataset source key in config.data.standard_files", metavar = "name"),
  make_option("--sample-n", type = "integer", default = 5000L,
              help = "Number of matrices to sample for diagnostics", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--n-cores", type = "integer", default = 8L,
              help = "Parallel workers across matrices", metavar = "int")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

standard_path <- (cfg$data$standard_files %||% list())[[opt$source]]
if (is.null(standard_path) || !file.exists(standard_path)) {
  stop("Missing standard file for source '", opt$source, "': ", standard_path %||% "<null>")
}

out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(out_results)
dir_create(fig_dir)

message("[", now_utc(), "] Reading standard long file: ", standard_path)
dt_all <- as.data.table(read_parquet(standard_path))
if (!"experiment_id" %in% names(dt_all)) stop("Expected column 'experiment_id' in: ", standard_path)

exp_ids <- unique(dt_all$experiment_id)
if (length(exp_ids) == 0) stop("No experiments found in: ", standard_path)
target_n <- min(opt$`sample-n`, length(exp_ids))
exp_ids <- if (length(exp_ids) <= target_n) exp_ids else sample(exp_ids, target_n)

message("[", now_utc(), "] Sampling ", length(exp_ids), " experiments (", opt$source, ")")

dt_list <- split(dt_all[experiment_id %in% exp_ids], by = "experiment_id", keep.by = TRUE)

one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
  meta <- df[1, .(experiment_id, source = (source %||% NA_character_), drugA, drugB, cell_line)]

  mt_z <- summarise_monotonicity(
    df_long = df,
    response_mode = cfg$responses$mode,
    clamp_01 = cfg$responses$clamp_01 %||% TRUE,
    use_transform = TRUE,
    transform = transform,
    method_direction = cfg$method$monotone_direction
  )
  mt_y <- summarise_monotonicity(
    df_long = df,
    response_mode = cfg$responses$mode,
    clamp_01 = cfg$responses$clamp_01 %||% TRUE,
    use_transform = FALSE,
    transform = NULL,
    method_direction = cfg$method$monotone_direction
  )

  data.table(
    meta,
    direction = mt_z$direction,
    n_i = mt_z$n_i,
    n_j = mt_z$n_j,
    frac_vio_A_Z = mt_z$frac_vio_A,
    frac_vio_B_Z = mt_z$frac_vio_B,
    max_vio_mag_A_Z = mt_z$max_vio_mag_A,
    max_vio_mag_B_Z = mt_z$max_vio_mag_B,
    frac_vio_A_Y = mt_y$frac_vio_A,
    frac_vio_B_Y = mt_y$frac_vio_B,
    max_vio_mag_A_Y = mt_y$max_vio_mag_A,
    max_vio_mag_B_Y = mt_y$max_vio_mag_B
  )
}

oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
workers <- max(1L, min(opt$`n-cores`, as.integer(future::availableCores(methods = "mc.cores"))))
future::plan(future::multisession, workers = workers)

res <- future_lapply(dt_list, function(df) tryCatch(one(df), error = function(e) NULL), future.seed = TRUE)
res <- res[!vapply(res, is.null, logical(1))]
dt <- rbindlist(res, fill = TRUE)

out_p <- file.path(out_results, paste0(opt$source, "_monotonicity.parquet"))
write_table(dt, out_p)
message("[", now_utc(), "] Wrote: ", out_p, " (n=", nrow(dt), ")")

if (nrow(dt) > 0) {
  plot_ecdf <- function(v, title) {
    ggplot(dt[is.finite(get(v))], aes(x = get(v))) +
      stat_ecdf(geom = "step", color = "#0C6291") +
      labs(title = title, x = "fraction of adjacent comparisons violating monotonicity", y = "ECDF") +
      theme_minimal(base_size = 10) +
      coord_cartesian(xlim = c(0, 1))
  }

  p1 <- plot_ecdf("frac_vio_A_Z", "Monotonicity violations along drug A (Z-scale)")
  p2 <- plot_ecdf("frac_vio_B_Z", "Monotonicity violations along drug B (Z-scale)")

  p3 <- ggplot(dt[is.finite(frac_vio_A_Z) & is.finite(frac_vio_B_Z)],
               aes(x = frac_vio_A_Z, y = frac_vio_B_Z)) +
    geom_bin2d(bins = 30) +
    scale_fill_viridis_c(option = "C") +
    labs(title = "Joint violation rates (Z-scale)", x = "A violations", y = "B violations", fill = "count") +
    theme_minimal(base_size = 10)

  # Panel 4: ECDF of max violation magnitude on viability scale
  dt[, max_vio_mag_Y := pmax(max_vio_mag_A_Y, max_vio_mag_B_Y, na.rm = TRUE)]
  p4 <- ggplot(dt[is.finite(max_vio_mag_Y)], aes(x = max_vio_mag_Y)) +
    stat_ecdf(geom = "step", color = "#0C6291") +
    geom_vline(xintercept = c(0.1, 0.2), linetype = "dashed", color = "grey50", linewidth = 0.4) +
    annotate("text", x = 0.1, y = 0.05, label = "0.1", hjust = -0.1, size = 2.8, color = "grey40") +
    annotate("text", x = 0.2, y = 0.05, label = "0.2", hjust = -0.1, size = 2.8, color = "grey40") +
    labs(title = "Max violation magnitude (viability scale)",
         x = "max adjacent violation (viability units)", y = "ECDF") +
    theme_minimal(base_size = 10) +
    coord_cartesian(xlim = c(0, 0.5))

  out_fig <- file.path(fig_dir, paste0("monotonicity_diagnostics_", opt$source, ".pdf"))
  ggsave(out_fig, (p1 + p2) / (p3 + p4), width = 10, height = 7)
  message("[", now_utc(), "] Wrote: ", out_fig)
}

message("[", now_utc(), "] Done.")
