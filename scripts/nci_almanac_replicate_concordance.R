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
  make_option("--n-groups", type = "integer", default = 800L,
              help = "Number of replicate groups (unordered drug pair + cell line) to sample", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--n-cores", type = "integer", default = 8L,
              help = "Parallel workers across groups", metavar = "int"),
  make_option("--out", type = "character", default = "results/nci_almanac_replicate_concordance.parquet",
              help = "Output parquet with replicate concordance rows", metavar = "path"),
  make_option("--out-fig", type = "character", default = "figures/public/replicate_concordance_nci_almanac.pdf",
              help = "Output figure PDF", metavar = "path")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)
data_path <- opt$data
if (is.na(data_path) || !nzchar(data_path)) {
  data_path <- (cfg$data$standard_files %||% list())$nci_almanac
}
if (is.null(data_path) || !file.exists(data_path)) stop("Missing NCI-ALMANAC data parquet: ", data_path %||% "<null>")
if (!file.exists(opt$`grid-summary`)) stop("Missing grid summary: ", opt$`grid-summary`)

out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
dir_create(out_results)
dir_create(file.path(out_figures, "public"))

grid <- as.data.table(read_parquet(opt$`grid-summary`))
req <- c("experiment_id", "drugA", "drugB", "cell_line", "has_monoA", "has_monoB")
miss <- setdiff(req, names(grid))
if (length(miss) > 0) stop("grid-summary missing columns: ", paste(miss, collapse = ", "))

grid_ok <- grid[!is.na(has_monoA) & has_monoA & !is.na(has_monoB) & has_monoB]
grid_ok[, canonA := pmin(drugA, drugB)]
grid_ok[, canonB := pmax(drugA, drugB)]

grp <- grid_ok[, .(n = .N), by = .(canonA, canonB, cell_line)]
grp <- grp[n >= 2]
if (nrow(grp) == 0) {
  message("[", now_utc(), "] No replicate groups found on NCI-ALMANAC after filtering; nothing to do.")
  quit(status = 0)
}

n_groups <- min(opt$`n-groups`, nrow(grp))
grp <- grp[sample.int(nrow(grp), n_groups)]
grp[, group_id := paste(canonA, canonB, cell_line, sep = "__")]

pick_pair <- function(cA, cB, cl) {
  sub <- grid_ok[canonA == cA & canonB == cB & cell_line == cl]
  sub <- sub[order(experiment_id)]
  if (nrow(sub) < 2) return(NULL)
  sub[1:2, .(experiment_id, drugA, drugB, canonA, canonB, cell_line)]
}

pairs <- rbindlist(lapply(seq_len(nrow(grp)), function(i) {
  pick_pair(grp$canonA[i], grp$canonB[i], grp$cell_line[i])
}), fill = TRUE)
if (nrow(pairs) == 0) stop("No experiment pairs selected.")
pairs[, group_id := paste(canonA, canonB, cell_line, sep = "__")]

ids <- unique(pairs$experiment_id)
message("[", now_utc(), "] Groups=", uniqueN(pairs$group_id), " experiments=", length(ids))

ds <- arrow::open_dataset(data_path)
dt_sel <- as.data.table(ds |>
  dplyr::filter(experiment_id %in% ids) |>
  dplyr::collect())
if (nrow(dt_sel) == 0) stop("No rows loaded for selected experiment_ids.")
dt_list <- split(dt_sel, by = "experiment_id", keep.by = TRUE)

align_surfaces <- function(s1, s2) {
  a <- intersect(s1$doseA, s2$doseA)
  b <- intersect(s1$doseB, s2$doseB)
  if (length(a) < 2 || length(b) < 2) return(NULL)
  i1 <- match(a, s1$doseA); j1 <- match(b, s1$doseB)
  i2 <- match(a, s2$doseA); j2 <- match(b, s2$doseB)
  list(
    v1 = as.numeric(s1$mat[i1, j1]),
    v2 = as.numeric(s2$mat[i2, j2])
  )
}

surface_compare <- function(s1, s2, eps = 1e-8) {
  aligned <- align_surfaces(s1, s2)
  if (is.null(aligned)) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_, rms1 = NA_real_, rms2 = NA_real_))
  v1 <- aligned$v1; v2 <- aligned$v2
  ok <- is.finite(v1) & is.finite(v2)
  v1 <- v1[ok]; v2 <- v2[ok]
  if (length(v1) == 0) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_, rms1 = NA_real_, rms2 = NA_real_, v1 = numeric(0), v2 = numeric(0)))
  rms1 <- sqrt(mean(v1^2)); rms2 <- sqrt(mean(v2^2))
  corr <- if (rms1 > eps && rms2 > eps && length(v1) >= 5) suppressWarnings(stats::cor(v1, v2)) else NA_real_
  rmse <- sqrt(mean((v1 - v2)^2))
  rel_rmse <- rmse / max(eps, rms1)
  list(corr = corr, rmse = rmse, rel_rmse = rel_rmse, rms1 = rms1, rms2 = rms2, v1 = v1, v2 = v2)
}

topk_overlap_vec <- function(v1, v2, k = 5) {
  v1 <- as.numeric(v1)
  v2 <- as.numeric(v2)
  ok <- is.finite(v1) & is.finite(v2)
  v1 <- v1[ok]
  v2 <- v2[ok]
  if (length(v1) == 0) return(NA_real_)
  ord1 <- order(-abs(v1), na.last = NA)
  ord2 <- order(-abs(v2), na.last = NA)
  idx1 <- head(ord1, k); idx2 <- head(ord2, k)
  if (length(idx1) == 0 || length(idx2) == 0) return(NA_real_)
  length(intersect(idx1, idx2)) / min(length(idx1), length(idx2))
}

canonicalize_surface <- function(surface, drugA, drugB, canonA, canonB) {
  if (is.null(surface) || !is.matrix(surface$mat)) return(surface)
  if (is.na(drugA) || is.na(drugB) || is.na(canonA) || is.na(canonB)) return(surface)
  if (identical(drugA, canonA) && identical(drugB, canonB)) return(surface)
  if (identical(drugA, canonB) && identical(drugB, canonA)) {
    return(list(mat = t(surface$mat), doseA = surface$doseB, doseB = surface$doseA))
  }
  surface
}

compute_surfaces <- function(df, canonA, canonB, drugA, drugB) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  prop <- compute_proposed_metrics(df, cfg, transform, boot_override = 0, skip_boot = TRUE)
  base <- compute_baselines(df, cfg)

  surfaces <- list(
    delta_Z = list(mat = prop$delta, doseA = prop$doseA_levels, doseB = prop$doseB_levels),
    S_proposed = list(mat = prop$S_proposed, doseA = prop$doseA_levels, doseB = prop$doseB_levels),
    Bliss = list(mat = base$bliss$synergy, doseA = base$bliss$doseA_levels, doseB = base$bliss$doseB_levels),
    HSA = list(mat = base$hsa$synergy, doseA = base$hsa$doseA_levels, doseB = base$hsa$doseB_levels),
    Loewe = list(mat = base$loewe$synergy, doseA = base$loewe$doseA_levels, doseB = base$loewe$doseB_levels),
    ZIP = list(mat = base$zip$synergy, doseA = base$zip$doseA_levels, doseB = base$zip$doseB_levels)
  )

  lapply(surfaces, canonicalize_surface, drugA = drugA, drugB = drugB, canonA = canonA, canonB = canonB)
}

pair_rows_one <- function(gid) {
  sub <- pairs[group_id == gid][order(experiment_id)]
  if (nrow(sub) < 2) return(NULL)
  a <- sub$experiment_id[1]
  b <- sub$experiment_id[2]
  df_a <- dt_list[[a]]
  df_b <- dt_list[[b]]
  if (is.null(df_a) || is.null(df_b)) return(NULL)

  canonA <- sub$canonA[1]
  canonB <- sub$canonB[1]
  sa <- compute_surfaces(df_a, canonA = canonA, canonB = canonB, drugA = sub$drugA[1], drugB = sub$drugB[1])
  sb <- compute_surfaces(df_b, canonA = canonA, canonB = canonB, drugA = sub$drugA[2], drugB = sub$drugB[2])

  methods <- names(sa)
  out <- vector("list", length(methods))
  for (i in seq_along(methods)) {
    m <- methods[i]
    comp <- surface_compare(
      list(mat = sa[[m]]$mat, doseA = sa[[m]]$doseA, doseB = sa[[m]]$doseB),
      list(mat = sb[[m]]$mat, doseA = sb[[m]]$doseA, doseB = sb[[m]]$doseB)
    )
    out[[i]] <- data.table(
      group_id = gid,
      experiment_a = a,
      experiment_b = b,
      method = m,
      corr = comp$corr,
      rel_rmse = comp$rel_rmse,
      top5 = topk_overlap_vec(comp$v1, comp$v2, k = 5),
      top10 = topk_overlap_vec(comp$v1, comp$v2, k = 10),
      rms_a = comp$rms1,
      rms_b = comp$rms2
    )
  }
  rbindlist(out, fill = TRUE)
}

workers <- max(1L, min(opt$`n-cores`, as.integer(future::availableCores(methods = "mc.cores"))))
oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
if (workers > 1) future::plan(future::multisession, workers = workers) else future::plan(future::sequential)

gids <- unique(pairs$group_id)
res_list <- future_lapply(gids, pair_rows_one, future.seed = TRUE)
res <- rbindlist(res_list, fill = TRUE)
if (nrow(res) == 0) stop("No concordance rows produced.")

write_table(res, opt$out)
message("[", now_utc(), "] Wrote: ", opt$out, " (rows=", nrow(res), ")")

summary <- res[, .(
  n_pairs = .N,
  n_valid = sum(is.finite(corr)),
  na_rate = mean(!is.finite(corr)),
  median_corr = stats::median(corr, na.rm = TRUE)
), by = method][order(-median_corr)]
print(summary)

method_order <- summary$method
res[, method := factor(method, levels = method_order)]

p <- ggplot(res[is.finite(corr)], aes(x = method, y = corr, fill = method)) +
  geom_violin(alpha = 0.65, draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    x = "Method",
    y = "Correlation between replicate surfaces",
    title = "NCI-ALMANAC replicate concordance",
    subtitle = "Replicates defined as same unordered drug pair + cell line (includes A/B-swapped assays)"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

dir_create(dirname(opt$`out-fig`))
ggsave(opt$`out-fig`, p, width = 8.5, height = 4.8)
message("[", now_utc(), "] Wrote: ", opt$`out-fig`)
