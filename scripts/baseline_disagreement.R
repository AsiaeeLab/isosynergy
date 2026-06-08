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
})

source("R/utils.R")
source("R/config.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path (used for output dirs)", metavar = "path"),
  make_option("--scores", type = "character", default = "results/drugcombdb_block_scores.parquet",
              help = "Parquet with baseline synergy scores (DrugCombDB block scores)", metavar = "path"),
  make_option("--summary", type = "character", default = "results/drugcombdb_baseline_disagreement.parquet",
              help = "Precomputed pairwise disagreement summary parquet", metavar = "path"),
  make_option("--replot-only", action = "store_true", default = FALSE,
              help = "Skip recompute; read --summary parquet and only redraw the figure"),
  make_option("--out-fig", type = "character", default = NA_character_,
              help = "Output figure PDF (defaults to <figures>/public/baseline_disagreement_drugcombdb.pdf)", metavar = "path"),
  make_option("--top-q", type = "double", default = 0.05,
              help = "Quantile for 'top synergy' overlap", metavar = "num")
)

opt <- parse_args(OptionParser(option_list = option_list))

cfg <- read_config(opt$config)
out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(out_results)
dir_create(fig_dir)

methods <- c("Bliss", "HSA", "Loewe", "ZIP")
q <- opt$`top-q`
out_p <- file.path(out_results, "drugcombdb_baseline_disagreement.parquet")

if (opt$`replot-only`) {
  # Fast path: read precomputed summary and only redraw (no pipeline recompute).
  if (!file.exists(opt$summary)) stop("Missing summary parquet: ", opt$summary)
  out_tbl <- as.data.table(read_parquet(opt$summary))
  df_cor <- out_tbl[metric == "pearson_corr"]
  df_dis <- out_tbl[metric == "pos_disagree_rate"]
  df_jac <- out_tbl[metric == "top_jaccard"]
  lev <- intersect(methods, unique(c(df_cor$method1, df_cor$method2)))
  for (d in list(df_cor, df_dis, df_jac)) {
    d[, method1 := factor(method1, levels = lev)]
    d[, method2 := factor(method2, levels = lev)]
  }
  message("[", now_utc(), "] Replot-only from: ", opt$summary)
} else {
  if (!file.exists(opt$scores)) stop("Missing file: ", opt$scores)

  dt <- as.data.table(read_parquet(opt$scores))
  missing <- setdiff(methods, names(dt))
  if (length(missing) > 0) stop("Missing expected columns: ", paste(missing, collapse = ", "))

  X <- as.data.frame(dt[, ..methods])
  X <- X[stats::complete.cases(X), , drop = FALSE]
  n <- nrow(X)
  if (n == 0) stop("No complete rows after dropping NAs.")
  message("[", now_utc(), "] Complete rows: ", n)

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
  write_table(out_tbl, out_p)
  message("[", now_utc(), "] Wrote: ", out_p)
}

# Colorblind-safe palettes (ColorBrewer): diverging RdBu for correlation,
# sequential YlGnBu for the two rate/overlap panels.
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
  title = "Pearson correlation of synergy scores",
  fmt = "%.2f",
  limits = c(-1, 1),
  mid = 0
)

p_dis <- plot_seq(df_dis, title = "Sign disagreement rate")

p_jac <- plot_seq(df_jac, title = "Jaccard overlap of top-5% hits")

out_fig <- if (is.na(opt$`out-fig`)) file.path(fig_dir, "baseline_disagreement_drugcombdb.pdf") else opt$`out-fig`
dir_create(dirname(out_fig))
# One row, three panels side by side.
ggsave(out_fig, p_corr | p_dis | p_jac, width = 15, height = 5.2)
message("[", now_utc(), "] Wrote: ", out_fig)

# Pairwise scatterplot matrix (Supplementary Figure)
if (!opt$`replot-only`) {
set.seed(42)
n_scatter <- min(nrow(X), 5000L)  # subsample for readability
idx <- sample.int(nrow(X), n_scatter)
X_sub <- as.data.table(X[idx, ])
X_long <- melt(X_sub, measure.vars = methods, variable.name = "method", value.name = "score")

pairs_list <- list()
for (i in seq_along(methods)) {
  for (j in seq_along(methods)) {
    if (i == j) next
    pairs_list[[length(pairs_list) + 1L]] <- data.table(
      x = X_sub[[methods[i]]],
      y = X_sub[[methods[j]]],
      xmethod = methods[i],
      ymethod = methods[j]
    )
  }
}
pairs_dt <- rbindlist(pairs_list)
pairs_dt[, xmethod := factor(xmethod, levels = methods)]
pairs_dt[, ymethod := factor(ymethod, levels = methods)]

p_scatter <- ggplot(pairs_dt, aes(x = x, y = y)) +
  geom_point(alpha = 0.05, size = 0.3, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey40") +
  facet_grid(ymethod ~ xmethod, scales = "free") +
  labs(x = "Synergy score (column method)", y = "Synergy score (row method)") +
  theme_minimal(base_size = 9) +
  theme(strip.text = element_text(size = 8))

out_scatter <- file.path(fig_dir, "baseline_pairwise_scatter.pdf")
ggsave(out_scatter, p_scatter, width = 8, height = 8)
message("[", now_utc(), "] Wrote: ", out_scatter)
}
