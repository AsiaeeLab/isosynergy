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
  library(patchwork)
  library(future)
  library(future.apply)
})

source("R/pipeline.R")
source("R/config.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--data", type = "character", default = NA_character_,
              help = "NCI-ALMANAC standardized long parquet (defaults to config.data.standard_files.nci_almanac)", metavar = "path"),
  make_option("--grid-summary", type = "character", default = "results/nci_almanac_grid_summary.parquet",
              help = "Parquet with per-experiment grid stats", metavar = "path"),
  make_option("--n", type = "integer", default = 20000L,
              help = "Number of matrices to sample for baseline disagreement", metavar = "int"),
  make_option("--top-q", type = "double", default = 0.05,
              help = "Quantile for 'top synergy' overlap", metavar = "num"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--n-cores", type = "integer", default = 8L,
              help = "Parallel workers across matrices", metavar = "int"),
  make_option("--out-scores", type = "character", default = "results/nci_almanac_baseline_scores.parquet",
              help = "Output parquet with per-experiment baseline summary scores", metavar = "path"),
  make_option("--out-summary", type = "character", default = "results/nci_almanac_baseline_disagreement.parquet",
              help = "Output parquet with pairwise disagreement summary", metavar = "path"),
  make_option("--out-fig", type = "character", default = "figures/public/baseline_disagreement_nci_almanac.pdf",
              help = "Output figure PDF", metavar = "path"),
  make_option("--replot-only", action = "store_true", default = FALSE,
              help = "Skip recompute; read --out-summary parquet and only redraw the figure")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
methods <- c("Bliss", "HSA", "Loewe", "ZIP")
q <- opt$`top-q`

if (opt$`replot-only`) {
  # Fast path: read precomputed pairwise summary and only redraw (no recompute).
  if (!file.exists(opt$`out-summary`)) stop("Missing summary parquet: ", opt$`out-summary`)
  out_tbl <- as.data.table(read_parquet(opt$`out-summary`))
  df_cor <- out_tbl[metric == "pearson_corr"]
  df_dis <- out_tbl[metric == "pos_disagree_rate"]
  df_jac <- out_tbl[metric == "top_jaccard"]
  lev <- intersect(methods, unique(c(df_cor$method1, df_cor$method2)))
  for (d in list(df_cor, df_dis, df_jac)) {
    d[, method1 := factor(method1, levels = lev)]
    d[, method2 := factor(method2, levels = lev)]
  }
  message("[", now_utc(), "] Replot-only from: ", opt$`out-summary`)
} else {

data_path <- opt$data
if (is.na(data_path) || !nzchar(data_path)) {
  data_path <- (cfg$data$standard_files %||% list())$nci_almanac
}
if (is.null(data_path) || !file.exists(data_path)) stop("Missing NCI-ALMANAC data parquet: ", data_path %||% "<null>")
if (!file.exists(opt$`grid-summary`)) stop("Missing grid summary: ", opt$`grid-summary`)

dir_create(out_results)
dir_create(file.path(out_figures, "public"))

grid <- as.data.table(read_parquet(opt$`grid-summary`))
req <- c("experiment_id", "n_i", "n_j", "has_monoA", "has_monoB")
miss <- setdiff(req, names(grid))
if (length(miss) > 0) stop("grid-summary missing columns: ", paste(miss, collapse = ", "))

grid_ok <- grid[
  !is.na(has_monoA) & has_monoA &
    !is.na(has_monoB) & has_monoB &
    is.finite(n_i) & is.finite(n_j) & n_i >= 3 & n_j >= 3
]
if (nrow(grid_ok) == 0) stop("No eligible experiments in grid summary after filtering for mono edges and min grid.")

n_target <- min(opt$n, nrow(grid_ok))
ids <- grid_ok[sample.int(nrow(grid_ok), n_target)]$experiment_id
message("[", now_utc(), "] Sampling n=", length(ids), " experiments (eligible=", nrow(grid_ok), ")")

ds <- arrow::open_dataset(data_path)
dt_sel <- as.data.table(ds |>
  dplyr::filter(experiment_id %in% ids) |>
  dplyr::collect())
if (nrow(dt_sel) == 0) stop("No rows loaded for selected experiment_ids.")
dt_list <- split(dt_sel, by = "experiment_id", keep.by = TRUE)

interior_mean <- function(M) {
  if (!is.matrix(M)) return(NA_real_)
  if (nrow(M) < 2 || ncol(M) < 2) return(NA_real_)
  mean(M[-1, -1], na.rm = TRUE)
}

compute_one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
  base <- compute_baselines(df, cfg)
  data.table(
    experiment_id = exp_id,
    Bliss = interior_mean(base$bliss$synergy),
    HSA = interior_mean(base$hsa$synergy),
    Loewe = interior_mean(base$loewe$synergy),
    ZIP = interior_mean(base$zip$synergy)
  )
}

workers <- max(1L, min(opt$`n-cores`, as.integer(future::availableCores(methods = "mc.cores"))))
oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
if (workers > 1) future::plan(future::multisession, workers = workers) else future::plan(future::sequential)

res_list <- future_lapply(dt_list, compute_one, future.seed = TRUE)
scores <- rbindlist(res_list, fill = TRUE)

write_table(scores, opt$`out-scores`)
message("[", now_utc(), "] Wrote: ", opt$`out-scores`, " (n=", nrow(scores), ")")

methods <- c("Bliss", "HSA", "Loewe", "ZIP")
X <- as.data.frame(scores[, ..methods])
X <- X[stats::complete.cases(X), , drop = FALSE]
n_complete <- nrow(X)
if (n_complete == 0) stop("No complete rows after dropping NAs.")
message("[", now_utc(), "] Complete rows: ", n_complete)

cor_mat <- stats::cor(X, use = "pairwise.complete.obs", method = "pearson")
df_cor <- as.data.table(as.table(cor_mat))
setnames(df_cor, c("method1", "method2", "value"))
df_cor[, metric := "pearson_corr"]

pos_disagree <- matrix(NA_real_, nrow = length(methods), ncol = length(methods),
                       dimnames = list(methods, methods))
for (i in seq_along(methods)) {
  for (j in seq_along(methods)) {
    a <- X[[methods[i]]] > 0
    b <- X[[methods[j]]] > 0
    pos_disagree[i, j] <- mean(a != b)
  }
}
df_dis <- as.data.table(as.table(pos_disagree))
setnames(df_dis, c("method1", "method2", "value"))
df_dis[, metric := "pos_disagree_rate"]

q <- opt$`top-q`
thr <- vapply(methods, function(m) stats::quantile(X[[m]], probs = 1 - q, na.rm = TRUE, names = FALSE, type = 8), numeric(1))
top_mat <- lapply(methods, function(m) X[[m]] >= thr[[m]])
names(top_mat) <- methods

jacc <- matrix(NA_real_, nrow = length(methods), ncol = length(methods), dimnames = list(methods, methods))
for (i in seq_along(methods)) {
  for (j in seq_along(methods)) {
    a <- top_mat[[methods[i]]]
    b <- top_mat[[methods[j]]]
    jacc[i, j] <- sum(a & b) / sum(a | b)
  }
}
df_jac <- as.data.table(as.table(jacc))
setnames(df_jac, c("method1", "method2", "value"))
df_jac[, metric := "top_jaccard"]

out_tbl <- rbindlist(list(df_cor, df_dis, df_jac), use.names = TRUE, fill = TRUE)
write_table(out_tbl, opt$`out-summary`)
message("[", now_utc(), "] Wrote: ", opt$`out-summary`)

}  # end recompute branch

# Colorblind-safe palettes (ColorBrewer), matching the DrugCombDB figure:
# diverging RdBu for correlation, sequential YlGnBu for rate/overlap panels.
diverging_cols <- rev(RColorBrewer::brewer.pal(11, "RdBu"))  # blue=low, red=high
sequential_cols <- RColorBrewer::brewer.pal(9, "YlGnBu")

plot_heat <- function(df, title, subtitle = NULL, fmt = "%.2f", limits = NULL, mid = 0) {
  df <- copy(df)
  df[, txt_color := ifelse(abs(value - mid) > 0.6 * max(abs(limits - mid)), "white", "black")]
  ggplot(df, aes(x = method1, y = method2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf(fmt, value), color = txt_color), size = 4, show.legend = FALSE) +
    scale_color_identity() +
    scale_fill_gradientn(colours = diverging_cols, limits = limits,
                         values = scales::rescale(seq(limits[1], limits[2], length.out = length(diverging_cols)),
                                                  from = limits)) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

plot_seq <- function(df, title, subtitle = NULL) {
  df <- copy(df)
  df[, txt_color := ifelse(value > 0.65, "white", "black")]
  ggplot(df, aes(x = method1, y = method2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value), color = txt_color), size = 4, show.legend = FALSE) +
    scale_color_identity() +
    scale_fill_gradientn(colours = sequential_cols, limits = c(0, 1)) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL, fill = NULL) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

p_corr <- plot_heat(
  df_cor,
  title = "Baseline agreement depends on the null model",
  subtitle = "Pearson correlation of baseline synergy scores across NCI-ALMANAC matrices",
  fmt = "%.2f",
  limits = c(-1, 1),
  mid = 0
)

p_dis <- plot_seq(
  df_dis,
  title = "Frequent sign disagreements",
  subtitle = "Fraction where only one method reports positive synergy"
)

p_jac <- plot_seq(
  df_jac,
  title = paste0("Low overlap in top ", round(100 * q), "% synergy hits"),
  subtitle = "Jaccard index of top-q call sets"
)

dir_create(dirname(opt$`out-fig`))
ggsave(opt$`out-fig`, p_corr / p_dis / p_jac, width = 7.5, height = 12)
message("[", now_utc(), "] Wrote: ", opt$`out-fig`)
