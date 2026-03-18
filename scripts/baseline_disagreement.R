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

if (!file.exists(opt$scores)) stop("Missing file: ", opt$scores)

dt <- as.data.table(read_parquet(opt$scores))
methods <- c("Bliss", "HSA", "Loewe", "ZIP")
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
out_p <- file.path(out_results, "drugcombdb_baseline_disagreement.parquet")
write_table(out_tbl, out_p)
message("[", now_utc(), "] Wrote: ", out_p)

plot_heat <- function(df, title, subtitle = NULL, fmt = "%.2f", limits = NULL, mid = 0) {
  ggplot(df, aes(x = method1, y = method2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf(fmt, value)), size = 4) +
    scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                         midpoint = mid, limits = limits) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
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

p_dis <- ggplot(df_dis, aes(x = method1, y = method2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", value)), size = 4) +
  scale_fill_viridis_c(option = "C", limits = c(0, 1)) +
  labs(
    title = "Sign disagreement rate",
    x = NULL, y = NULL, fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

p_jac <- ggplot(df_jac, aes(x = method1, y = method2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", value)), size = 4) +
  scale_fill_viridis_c(option = "C", limits = c(0, 1)) +
  labs(
    title = "Jaccard overlap of top-5% hits",
    x = NULL, y = NULL, fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

out_fig <- file.path(fig_dir, "baseline_disagreement_drugcombdb.pdf")
ggsave(out_fig, p_corr / p_dis / p_jac, width = 7.5, height = 12)
message("[", now_utc(), "] Wrote: ", out_fig)
