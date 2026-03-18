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
source("R/transforms.R")
source("R/matrix_stats.R")
source("R/osqp_helpers.R")
source("R/isotonic_2d.R")
source("R/additive_ordered.R")
source("R/interaction.R")
source("R/bootstrap.R")
source("R/metrics.R")
source("R/data_standard.R")

plot_mat <- function(mat, title, limits = NULL, midpoint = NULL, palette = "viridis") {
  dt <- data.table(
    i = rep(seq_len(nrow(mat)), times = ncol(mat)),
    j = rep(seq_len(ncol(mat)), each = nrow(mat)),
    value = as.numeric(mat)
  )
  p <- ggplot(dt, aes(x = j, y = i, fill = value)) +
    geom_raster() +
    scale_y_reverse() +
    coord_fixed(ratio = 1) +
    labs(title = title, x = "Dose B index", y = "Dose A index", fill = NULL) +
    theme_minimal(base_size = 10) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = 10))

  if (palette == "diverging") {
    if (is.null(midpoint)) midpoint <- 0
    p <- p + scale_fill_gradient2(low = "#2B83BA", mid = "white", high = "#D7191C", midpoint = midpoint, limits = limits)
  } else {
    p <- p + scale_fill_viridis_c(option = "C", limits = limits)
  }
  p
}

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--source", type = "character", default = "drugcombdb",
              help = "Dataset source key in config.data.standard_files", metavar = "name"),
  make_option("--experiment-id", type = "character", default = "drugcombdb__block0016939",
              help = "Example experiment_id for the method overview figure", metavar = "id"),
  make_option("--boot-B", type = "integer", default = 200L,
              help = "Bootstrap iterations for the example p-value", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

standard_path <- (cfg$data$standard_files %||% list())[[opt$source]]
if (is.null(standard_path) || !file.exists(standard_path)) {
  stop("Missing standard file for source '", opt$source, "': ", standard_path %||% "<null>")
}

out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(fig_dir)

dt_all <- as.data.table(read_parquet(standard_path))
df <- dt_all[experiment_id == opt$`experiment-id`]
if (nrow(df) == 0) stop("No rows for experiment_id: ", opt$`experiment-id`)

prop <- compute_proposed_metrics(df, cfg, transform, boot_override = opt$`boot-B`)

Y_obs <- reshape_long_to_matrix(df, value_col = "response")$matrix
V_add <- transform$inverse(prop$theta_add)
V_iso <- transform$inverse(prop$theta_iso)
delta_Z <- prop$delta
S_prop <- prop$S_proposed
W <- prop$w

p1 <- plot_mat(Y_obs, "Observed viability", limits = c(0, 1))
p2 <- plot_mat(V_add, expression(hat(theta)[add]~"(viability)"), limits = c(0, 1))
p3 <- plot_mat(V_iso, expression(hat(theta)[iso]~"(viability)"), limits = c(0, 1))
p4 <- plot_mat(delta_Z, expression(delta~"(logit scale)"), palette = "diverging")
p5 <- plot_mat(S_prop, expression(S[SIR]~"(viability)"), palette = "diverging", midpoint = 0)

# Panel F: contour plot of interaction surface
dt_contour <- data.table(
  x = rep(seq_len(ncol(delta_Z)), each = nrow(delta_Z)),
  y = rep(seq_len(nrow(delta_Z)), times = ncol(delta_Z)),
  z = as.numeric(delta_Z)
)
p6 <- ggplot(dt_contour, aes(x = x, y = y, z = z)) +
  geom_contour_filled() +
  scale_y_reverse() +
  coord_fixed(ratio = 1) +
  labs(title = "Interaction contour", x = "Dose B index", y = "Dose A index", fill = expression(delta)) +
  theme_minimal(base_size = 10) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(size = 10))

fig <- (p1 | p2 | p3) / (p4 | p5 | p6) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag = element_text(size = 14, face = "bold"))

out_path <- file.path(fig_dir, "method_overview_example.pdf")
ggsave(out_path, fig, width = 11, height = 7)
message("[", now_utc(), "] Wrote: ", out_path)
