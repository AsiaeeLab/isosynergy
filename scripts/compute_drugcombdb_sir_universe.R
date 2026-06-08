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

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(jsonlite)
  library(parallel)
})

source("R/pipeline.R")
source("R/config.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

opts <- list(
  config = "configs/default.yaml",
  data = "data/processed/drugcomb_matrices.parquet",
  out = "results/drugcombdb_sir_universe.parquet",
  checkpoint = "results/drugcombdb_sir_universe.checkpoint.json",
  order = "results/drugcombdb_sir_universe.universe_order.parquet",
  progress = "dataset_release/PROGRESS.md",
  chunk_size = 1000L,
  B = 200L,
  seed = 42L,
  worker_cap = 12L,
  wall_clock_budget_hours = NA_real_,
  wall_clock_cap_s = Inf,
  stop_on_projected_runtime = FALSE,
  stop_on_wall_clock_budget = FALSE
)

iso_now <- function() {
  format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}

fmt_duration <- function(seconds) {
  seconds <- as.numeric(seconds)
  if (!is.finite(seconds)) return("NA")
  if (seconds < 60) return(sprintf("%.1fs", seconds))
  if (seconds < 3600) return(sprintf("%.1fmin", seconds / 60))
  sprintf("%.2fh", seconds / 3600)
}

sync_path <- function(path) {
  sync_bin <- Sys.which("sync")
  if (!nzchar(sync_bin)) return(invisible(FALSE))
  ok <- suppressWarnings(system2(sync_bin, c("-f", path), stdout = FALSE, stderr = FALSE))
  if (!identical(ok, 0L)) {
    suppressWarnings(system2(sync_bin, stdout = FALSE, stderr = FALSE))
  }
  invisible(TRUE)
}

ensure_progress <- function(path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (!file.exists(path)) {
    con <- file(path, open = "a")
    on.exit(close(con), add = TRUE)
    writeLines(c(
      "# DrugCombDB SIR Universe Release Progress",
      "",
      "| timestamp | phase | event | details |",
      "| --- | --- | --- | --- |"
    ), con)
    flush(con)
    sync_path(path)
  }
}

escape_md_cell <- function(x) {
  x <- gsub("\n", "<br>", as.character(x), fixed = TRUE)
  gsub("|", "\\\\|", x, fixed = TRUE)
}

append_progress <- function(phase, event, details, path = opts$progress) {
  ensure_progress(path)
  line <- sprintf(
    "| %s | %s | %s | %s |",
    iso_now(), escape_md_cell(phase), escape_md_cell(event), escape_md_cell(details)
  )
  con <- file(path, open = "a")
  on.exit(close(con), add = TRUE)
  writeLines(line, con)
  flush(con)
  sync_path(path)
  invisible(TRUE)
}

atomic_write_json <- function(x, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), sprintf(".%s.tmp.%s", basename(path), Sys.getpid()))
  jsonlite::write_json(x, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null")
  sync_path(tmp)
  if (!file.rename(tmp, path)) stop("Failed to atomically rename JSON checkpoint: ", path)
  sync_path(path)
  sync_path(dirname(path))
  invisible(path)
}

atomic_write_parquet <- function(dt, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), sprintf(".%s.tmp.%s", basename(path), Sys.getpid()))
  arrow::write_parquet(dt, tmp, compression = "zstd")
  sync_path(tmp)
  if (!file.rename(tmp, path)) stop("Failed to atomically rename parquet: ", path)
  sync_path(path)
  sync_path(dirname(path))
  invisible(path)
}

part_files <- function(out_dir) {
  if (!dir.exists(out_dir)) return(character(0))
  sort(list.files(out_dir, pattern = "^chunk_[0-9]{6}\\.parquet$", full.names = TRUE))
}

read_existing_results <- function(out_dir, col_select = NULL) {
  files <- part_files(out_dir)
  if (length(files) == 0) return(data.table())
  pieces <- lapply(files, function(f) {
    as.data.table(arrow::read_parquet(f, col_select = col_select, as_data_frame = TRUE))
  })
  rbindlist(pieces, use.names = TRUE, fill = TRUE)
}

stable_seed <- function(experiment_id, global_seed = 42L) {
  vapply(experiment_id, function(id) {
    h <- as.numeric(global_seed)
    ints <- utf8ToInt(as.character(id))
    for (ii in ints) h <- (h * 131 + ii) %% 2147483646
    as.integer(h + 1L)
  }, integer(1))
}

write_chunk_atomic <- function(dt, out_dir, chunk_id) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  final <- file.path(out_dir, sprintf("chunk_%06d.parquet", as.integer(chunk_id)))
  if (file.exists(final)) stop("Refusing to overwrite existing chunk file: ", final)
  tmp <- file.path(dirname(out_dir), sprintf(".%s.chunk_%06d.tmp.%s.parquet",
                                            basename(out_dir), as.integer(chunk_id), Sys.getpid()))
  arrow::write_parquet(dt, tmp, compression = "zstd")
  sync_path(tmp)
  if (!file.rename(tmp, final)) stop("Failed to atomically rename chunk file: ", final)
  sync_path(final)
  sync_path(out_dir)
  invisible(final)
}

compute_one_matrix <- function(payload) {
  df <- data.table::as.data.table(payload$df)
  exp_id <- as.character(payload$experiment_id)
  started <- proc.time()[["elapsed"]]

  out <- tryCatch({
    df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
    if (nrow(df) == 0) stop("no finite dose-response observations after filtering")

    cfg_local <- cfg_worker
    cfg_local$project$n_cores <- 1L
    cfg_local$bootstrap$B <- as.integer(payload$B)
    cfg_local$bootstrap$seed <- as.integer(payload$seed)

    prop <- compute_proposed_metrics(
      df_long = df,
      cfg = cfg_local,
      transform = transform_worker,
      boot_override = as.integer(payload$B),
      skip_boot = FALSE
    )

    direction <- if (is.finite(prop$synergy_index) && prop$synergy_index > 0) {
      "synergy"
    } else if (is.finite(prop$synergy_index) && prop$synergy_index < 0) {
      "antagonism"
    } else {
      "balanced"
    }

    data.table::data.table(
      experiment_id = exp_id,
      sir_S2 = as.numeric(prop$S2),
      sir_p_value = as.numeric(prop$p_value),
      sir_mean_delta = as.numeric(prop$mean_delta_w),
      sir_direction = direction,
      sir_failed = FALSE,
      n_i = as.integer(length(prop$doseA_levels)),
      n_j = as.integer(length(prop$doseB_levels)),
      B = as.integer(payload$B),
      runtime_s = as.numeric(proc.time()[["elapsed"]] - started),
      chunk_id = as.integer(payload$chunk_id),
      completed_at = format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      .error_message = NA_character_
    )
  }, error = function(e) {
    data.table::data.table(
      experiment_id = exp_id,
      sir_S2 = NA_real_,
      sir_p_value = NA_real_,
      sir_mean_delta = NA_real_,
      sir_direction = NA_character_,
      sir_failed = TRUE,
      n_i = NA_integer_,
      n_j = NA_integer_,
      B = as.integer(payload$B),
      runtime_s = as.numeric(proc.time()[["elapsed"]] - started),
      chunk_id = as.integer(payload$chunk_id),
      completed_at = format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      .error_message = conditionMessage(e)
    )
  })

  out
}

total_cores <- parallel::detectCores(logical = TRUE)
if (!is.finite(total_cores) || is.na(total_cores)) total_cores <- 1L
max_workers <- max(1L, min(as.integer(total_cores) - 1L, as.integer(opts$worker_cap)))

dir.create("results", showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(opts$progress), showWarnings = FALSE, recursive = TRUE)
ensure_progress(opts$progress)

append_progress(
  "compute", "started",
  sprintf("started SIR universe compute; max_workers=min(detectCores()-1,%d)=%d; B=%d; seed=%d; projected_runtime_stop=%s; wall_clock_stop=%s",
          opts$worker_cap, max_workers, opts$B, opts$seed,
          opts$stop_on_projected_runtime, opts$stop_on_wall_clock_budget)
)
append_progress(
  "compute", "decision",
  sprintf("DECISION: latest user instruction supersedes the prior 10-core/48h policy; using up to %d workers and continuing regardless of projected full runtime.",
          opts$worker_cap)
)
append_progress(
  "compute", "decision",
  "DECISION: using an Arrow Parquet dataset directory at results/drugcombdb_sir_universe.parquet with one atomically renamed part file per chunk, because single Parquet files are not safely appendable."
)
append_progress(
  "compute", "decision",
  "DECISION: limiting BLAS/OpenMP thread environment variables to 1 per R process so the outer PSOCK worker count is the only source of CPU parallelism."
)
append_progress(
  "compute", "decision",
  "DECISION: sir_mean_delta is retained in checkpoint chunks as weighted mean_delta_w for auditability, but the final Zenodo release omits it because it is zero by construction."
)
append_progress(
  "compute", "decision",
  "DECISION: sir_direction is derived from the sign of the SIR synergy index, i.e. dominant synergy-aligned versus antagonism-aligned interaction energy, because the weighted signed mean delta is near zero by construction."
)
append_progress(
  "compute", "decision",
  "DECISION: per-matrix bootstrap seeds are deterministic hashes of experiment_id and the global seed 42, so resumed runs reproduce an uninterrupted run while workers are also initialized with clusterSetRNGStream(42)."
)
append_progress(
  "compute", "decision",
  "DECISION: processing order is a persisted deterministic shuffle generated by set.seed(42); sample(experiment_ids), so runtime extrapolations see a representative grid-size mix."
)

cfg <- read_config(opts$config)
cfg$project$n_cores <- 1L
cfg$bootstrap$B <- opts$B
cfg$bootstrap$seed <- opts$seed
transform <- make_transform(cfg$transform)

if (!file.exists(opts$data)) stop("Missing processed matrix parquet: ", opts$data)

id_dt <- as.data.table(arrow::read_parquet(opts$data, col_select = "experiment_id", as_data_frame = TRUE))
experiment_ids <- unique(as.character(id_dt$experiment_id))
order_existed_on_start <- file.exists(opts$order)
if (order_existed_on_start) {
  order_dt <- as.data.table(arrow::read_parquet(opts$order, as_data_frame = TRUE))
  if (!all(c("order_index", "experiment_id") %in% names(order_dt))) {
    stop("Universe order file is missing required columns: ", opts$order)
  }
  setorder(order_dt, order_index)
  universe_ids <- as.character(order_dt$experiment_id)
  if (length(universe_ids) != length(experiment_ids) || !setequal(universe_ids, experiment_ids)) {
    append_progress("compute", "error", "persisted universe order does not match current processed DrugCombDB universe")
    stop("Universe order mismatch")
  }
  append_progress("compute", "started", sprintf("loaded persisted shuffled universe order from %s", opts$order))
} else {
  set.seed(opts$seed)
  universe_ids <- sample(experiment_ids)
  order_dt <- data.table(order_index = seq_along(universe_ids), experiment_id = universe_ids)
  atomic_write_parquet(order_dt, opts$order)
  append_progress(
    "compute", "decision",
    sprintf("DECISION: wrote deterministic shuffled universe order to %s using set.seed(%d); sample(experiment_ids).",
            opts$order, opts$seed)
  )
}
total_n <- length(universe_ids)
rm(id_dt)
gc()

existing <- read_existing_results(opts$out, col_select = c("experiment_id", "runtime_s", "sir_failed"))
completed_ids <- if (nrow(existing) > 0) unique(as.character(existing$experiment_id)) else character(0)
completed_count <- length(completed_ids)
existing_runtime_s <- if ("runtime_s" %in% names(existing)) sum(existing$runtime_s, na.rm = TRUE) else 0

sidecar <- NULL
if (file.exists(opts$checkpoint)) {
  sidecar <- tryCatch(jsonlite::read_json(opts$checkpoint, simplifyVector = TRUE), error = function(e) NULL)
}
cumulative_wall_clock_s <- as.numeric(sidecar$cumulative_wall_clock_s %||% 0)
if (!is.finite(cumulative_wall_clock_s)) cumulative_wall_clock_s <- 0
cumulative_core_runtime_s <- as.numeric(sidecar$cumulative_core_runtime_s %||% existing_runtime_s)
if (!is.finite(cumulative_core_runtime_s)) cumulative_core_runtime_s <- existing_runtime_s
last_chunk_id_done <- as.integer(sidecar$last_chunk_id %||% NA_integer_)

if (completed_count > 0) {
  expected_done <- head(universe_ids, completed_count)
  done_matches_order_prefix <- setequal(completed_ids, expected_done)
  if (!done_matches_order_prefix) {
    old_wall_s <- as.numeric(sidecar$cumulative_wall_clock_s %||% cumulative_wall_clock_s)
    if (!is.finite(old_wall_s)) old_wall_s <- cumulative_wall_clock_s
    if (old_wall_s <= 30 * 60) {
      append_progress(
        "compute", "decision",
        sprintf("DECISION: Option A selected; discarding %d old unshuffled checkpoint rows because they are not the prefix of the shuffled universe order and rerunning cost was %s (<30min). PROGRESS.md is retained.",
                completed_count, fmt_duration(old_wall_s))
      )
      old_parts <- part_files(opts$out)
      if (length(old_parts) > 0) unlink(old_parts, force = TRUE)
      unlink(list.files("results", pattern = "^\\.drugcombdb_sir_universe\\.parquet.*\\.tmp\\..*\\.parquet$", full.names = TRUE), force = TRUE)
      completed_ids <- character(0)
      completed_count <- 0L
      existing <- data.table()
      existing_runtime_s <- 0
      cumulative_wall_clock_s <- 0
      cumulative_core_runtime_s <- 0
      last_chunk_id_done <- NA_integer_
      sidecar <- list()
      atomic_write_json(list(
        total_universe_size = total_n,
      completed_count = 0L,
      last_chunk_id = NA_integer_,
      last_completed_at = iso_now(),
      seed = opts$seed,
      max_workers = max_workers,
      worker_cap = opts$worker_cap,
      chunk_size = opts$chunk_size,
      B = opts$B,
      output_path = opts$out,
      universe_order_path = opts$order,
      wall_clock_budget_hours = opts$wall_clock_budget_hours,
      stop_on_projected_runtime = opts$stop_on_projected_runtime,
      stop_on_wall_clock_budget = opts$stop_on_wall_clock_budget,
      cumulative_wall_clock_s = 0,
      cumulative_core_runtime_s = 0,
      extrapolation_after_3000_done = FALSE,
      stopped_for_runtime_budget = FALSE
      ), opts$checkpoint)
    } else {
      append_progress(
        "compute", "decision",
        sprintf("DECISION: Option B selected; keeping %d existing checkpoint rows even though they are not the shuffled-order prefix because rerunning cost was %s (>30min). Remaining matrices will be processed in shuffled order.",
                completed_count, fmt_duration(old_wall_s))
      )
    }
  }
}

if (completed_count > 0) {
  append_progress(
    "compute", "started",
    sprintf("RESUMING from %d/%d; existing_part_files=%d; cumulative_wall_clock=%s",
            completed_count, total_n, length(part_files(opts$out)), fmt_duration(cumulative_wall_clock_s))
  )
} else {
  append_progress("compute", "started", sprintf("fresh run over %d matrices from %s", total_n, opts$data))
}

if (!is.null(sidecar) && !is.null(sidecar$completed_count) &&
    as.integer(sidecar$completed_count) != completed_count) {
  append_progress(
    "compute", "warning",
    sprintf("checkpoint sidecar completed_count=%s but parquet parts contain %d unique experiment_id values; using parquet parts as authoritative",
            as.character(sidecar$completed_count), completed_count)
  )
}

if (any(duplicated(existing$experiment_id))) {
  dup_n <- sum(duplicated(existing$experiment_id))
  append_progress("compute", "error", sprintf("found %d duplicate experiment_id rows in existing checkpoint; stopping", dup_n))
  stop("Duplicate experiment_id rows in checkpoint")
}

if (completed_count >= total_n) {
  append_progress("compute", "completed", sprintf("checkpoint already complete: %d/%d", completed_count, total_n))
  quit(status = 0)
}

if (isTRUE(sidecar$stopped_for_runtime_budget %||% FALSE) && !isTRUE(opts$stop_on_projected_runtime) && !isTRUE(opts$stop_on_wall_clock_budget)) {
  sidecar$stopped_for_runtime_budget <- FALSE
  sidecar$runtime_budget_recommendation <- NULL
  sidecar$max_workers <- max_workers
  sidecar$worker_cap <- opts$worker_cap
  sidecar$wall_clock_budget_hours <- opts$wall_clock_budget_hours
  sidecar$stop_on_projected_runtime <- opts$stop_on_projected_runtime
  sidecar$stop_on_wall_clock_budget <- opts$stop_on_wall_clock_budget
  atomic_write_json(sidecar, opts$checkpoint)
  append_progress(
    "compute", "decision",
    sprintf("DECISION: cleared prior stopped_for_runtime_budget checkpoint flag at %d/%d because latest instruction says to continue regardless of projected full runtime.",
            completed_count, total_n)
  )
}

if (isTRUE(sidecar$stopped_for_runtime_budget %||% FALSE)) {
  append_progress(
    "compute", "decision",
    sprintf("DECISION: not resuming additional chunks because the checkpoint sidecar records stopped_for_runtime_budget=true after %d/%d matrices. Recommendation remains: %s",
            completed_count, total_n,
            sidecar$runtime_budget_recommendation %||% "reduce B, use a larger dedicated run, or release a documented subset")
  )
  quit(status = 2)
}

cols_needed <- c("experiment_id", "doseA", "doseB", "response", "replicate")
matrix_dt <- as.data.table(arrow::read_parquet(opts$data, col_select = cols_needed, as_data_frame = TRUE))
matrix_dt[, experiment_id := as.character(experiment_id)]
setkey(matrix_dt, experiment_id)

seed_dt <- data.table(experiment_id = universe_ids, sir_seed = stable_seed(universe_ids, opts$seed))
setkey(seed_dt, experiment_id)

cl <- parallel::makeCluster(max_workers, type = "PSOCK")
on.exit({
  try(parallel::stopCluster(cl), silent = TRUE)
}, add = TRUE)
parallel::clusterCall(cl, function(wd) {
  setwd(wd)
  Sys.setenv(
    OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    VECLIB_MAXIMUM_THREADS = "1",
    NUMEXPR_NUM_THREADS = "1"
  )
  NULL
}, getwd())
parallel::clusterSetRNGStream(cl, iseed = opts$seed)
parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages(library(data.table))
  source("R/pipeline.R")
  NULL
})
cfg_worker <- cfg
transform_worker <- transform
parallel::clusterExport(
  cl,
  varlist = c("cfg_worker", "transform_worker", "compute_one_matrix"),
  envir = environment()
)

run_started <- proc.time()[["elapsed"]]
extrapolation_after_3000_done <- isTRUE(sidecar$extrapolation_after_3000_done %||% FALSE)

repeat {
  remaining_ids <- universe_ids[!(universe_ids %chin% completed_ids)]
  if (length(remaining_ids) == 0) break

  if (isTRUE(opts$stop_on_wall_clock_budget) && cumulative_wall_clock_s >= opts$wall_clock_cap_s) {
    checkpoint <- list(
      total_universe_size = total_n,
      completed_count = completed_count,
      last_chunk_id = last_chunk_id_done,
      last_completed_at = iso_now(),
      seed = opts$seed,
      max_workers = max_workers,
      worker_cap = opts$worker_cap,
      chunk_size = opts$chunk_size,
      B = opts$B,
      output_path = opts$out,
      universe_order_path = opts$order,
      wall_clock_budget_hours = opts$wall_clock_budget_hours,
      stop_on_projected_runtime = opts$stop_on_projected_runtime,
      stop_on_wall_clock_budget = opts$stop_on_wall_clock_budget,
      cumulative_wall_clock_s = cumulative_wall_clock_s,
      cumulative_core_runtime_s = cumulative_core_runtime_s,
      extrapolation_after_3000_done = extrapolation_after_3000_done,
      stopped_for_runtime_budget = TRUE,
      runtime_budget_recommendation = "Cumulative wall clock reached the authorized 48h cap; request an extension or accept a partial checkpoint."
    )
    atomic_write_json(checkpoint, opts$checkpoint)
    append_progress(
      "compute", "decision:stopped_for_runtime_budget",
      sprintf("DECISION: stopping before next chunk because cumulative wall-clock %s reached the %dh cap.",
              fmt_duration(cumulative_wall_clock_s), opts$wall_clock_budget_hours)
    )
    break
  }

  chunk_ids <- head(remaining_ids, opts$chunk_size)
  first_idx <- match(chunk_ids[1], universe_ids)
  chunk_id <- if (is.finite(last_chunk_id_done)) {
    as.integer(last_chunk_id_done + 1L)
  } else {
    as.integer(ceiling(first_idx / opts$chunk_size))
  }
  chunk_start <- proc.time()[["elapsed"]]

  chunk_dt <- matrix_dt[J(chunk_ids), nomatch = 0]
  chunk_list <- split(chunk_dt, by = "experiment_id", keep.by = TRUE, sorted = FALSE)
  chunk_list <- chunk_list[chunk_ids]
  missing_ids <- chunk_ids[!vapply(chunk_list, function(x) is.data.frame(x) && nrow(x) > 0, logical(1))]
  if (length(missing_ids) > 0) {
    append_progress(
      "compute", "error",
      sprintf("chunk_id=%d has %d experiment_id values with no matrix rows; first_missing=%s",
              chunk_id, length(missing_ids), missing_ids[1])
    )
    stop("Missing matrix rows in chunk")
  }

  seeds <- seed_dt[J(chunk_ids)]$sir_seed
  payload <- Map(function(df, id, seed) {
    list(
      df = df,
      experiment_id = id,
      seed = seed,
      B = opts$B,
      chunk_id = chunk_id
    )
  }, chunk_list, chunk_ids, seeds)

  append_progress(
    "compute", "started",
    sprintf("chunk_id=%d started; rows=%d; completed_before=%d/%d",
            chunk_id, length(payload), completed_count, total_n)
  )

  result_list <- parallel::parLapply(cl, payload, compute_one_matrix)
  chunk_res <- rbindlist(result_list, use.names = TRUE, fill = TRUE)
  setorder(chunk_res, experiment_id)

  failed <- chunk_res[sir_failed == TRUE]
  if (nrow(failed) > 0) {
    for (ii in seq_len(nrow(failed))) {
      append_progress(
        "compute", "error",
        sprintf("sir_failed experiment_id=%s cause=%s",
                failed$experiment_id[ii], failed$.error_message[ii] %||% "unknown")
      )
    }
  }

  invalid <- chunk_res[
    !sir_failed &
      (!is.finite(sir_S2) | !is.finite(sir_p_value) | sir_p_value < 0 | sir_p_value > 1)
  ]
  if (nrow(invalid) > 0) {
    append_progress(
      "compute", "error",
      sprintf("invalid SIR numeric outputs in chunk_id=%d; n=%d; first=%s",
              chunk_id, nrow(invalid), invalid$experiment_id[1])
    )
    stop("Invalid finite/range check for SIR outputs")
  }

  chunk_res[, .error_message := NULL]
  write_chunk_atomic(chunk_res, opts$out, chunk_id)

  chunk_wall_s <- proc.time()[["elapsed"]] - chunk_start
  cumulative_wall_clock_s <- cumulative_wall_clock_s + chunk_wall_s
  cumulative_core_runtime_s <- cumulative_core_runtime_s + sum(chunk_res$runtime_s, na.rm = TRUE)
  completed_ids <- c(completed_ids, chunk_res$experiment_id)
  completed_count <- length(completed_ids)
  last_chunk_id_done <- chunk_id

  elapsed_rate <- completed_count / max(cumulative_wall_clock_s, 1e-9)
  eta_s <- (total_n - completed_count) / max(elapsed_rate, 1e-9)

  checkpoint <- list(
    total_universe_size = total_n,
    completed_count = completed_count,
    last_chunk_id = chunk_id,
    last_completed_at = iso_now(),
    seed = opts$seed,
    max_workers = max_workers,
    worker_cap = opts$worker_cap,
    chunk_size = opts$chunk_size,
    B = opts$B,
    output_path = opts$out,
    universe_order_path = opts$order,
    wall_clock_budget_hours = opts$wall_clock_budget_hours,
    stop_on_projected_runtime = opts$stop_on_projected_runtime,
    stop_on_wall_clock_budget = opts$stop_on_wall_clock_budget,
    cumulative_wall_clock_s = cumulative_wall_clock_s,
    cumulative_core_runtime_s = cumulative_core_runtime_s,
    extrapolation_after_3000_done = extrapolation_after_3000_done,
    stopped_for_runtime_budget = FALSE
  )

  append_progress(
    "compute", "chunk_done",
    sprintf("chunk_id=%d completed; chunk_rows=%d; completed=%d/%d; failed=%d; chunk_wall=%s; cumulative_wall=%s; ETA=%s",
            chunk_id, nrow(chunk_res), completed_count, total_n, nrow(failed),
            fmt_duration(chunk_wall_s), fmt_duration(cumulative_wall_clock_s), fmt_duration(eta_s))
  )

  if (completed_count >= 3000L) {
    est_total_s <- cumulative_wall_clock_s * total_n / completed_count
    checkpoint$latest_projected_total_wall_clock_s <- est_total_s
    if (isTRUE(opts$stop_on_projected_runtime) && est_total_s > opts$wall_clock_cap_s) {
      checkpoint$stopped_for_runtime_budget <- TRUE
      checkpoint$runtime_budget_recommendation <- "Full B=200 SIR universe run extrapolates beyond the authorized 48h budget; request an extension or accept a partial checkpoint."
      atomic_write_json(checkpoint, opts$checkpoint)
      append_progress(
        "compute", "decision:stopped_for_runtime_budget",
        sprintf("DECISION: stopping after %d matrices; extrapolated total runtime is %s, exceeding the %dh cap at max_workers=%d.",
                completed_count, fmt_duration(est_total_s), opts$wall_clock_budget_hours, max_workers)
      )
      quit(status = 2)
    }
    if (!extrapolation_after_3000_done) {
      extrapolation_after_3000_done <- TRUE
      checkpoint$extrapolation_after_3000_done <- TRUE
      append_progress(
        "compute", "decision:extrapolation_recorded",
        sprintf("DECISION: recorded extrapolation after %d shuffled matrices; projected total runtime is %s at max_workers=%d, and run continues because projected-runtime stopping is disabled.",
                completed_count, fmt_duration(est_total_s), max_workers)
      )
    }
  }

  if (isTRUE(opts$stop_on_wall_clock_budget) && cumulative_wall_clock_s >= opts$wall_clock_cap_s) {
    checkpoint$stopped_for_runtime_budget <- TRUE
    checkpoint$runtime_budget_recommendation <- "Cumulative wall clock reached the authorized 48h cap; request an extension or accept a partial checkpoint."
    atomic_write_json(checkpoint, opts$checkpoint)
    append_progress(
      "compute", "decision:stopped_for_runtime_budget",
      sprintf("DECISION: stopping after chunk_id=%d because cumulative wall-clock %s reached the %dh cap.",
              chunk_id, fmt_duration(cumulative_wall_clock_s), opts$wall_clock_budget_hours)
    )
    quit(status = 2)
  }

  atomic_write_json(checkpoint, opts$checkpoint)

  if (nrow(failed) > 0) {
    append_progress("compute", "error", sprintf("stopping after chunk_id=%d because SIR failures were observed", chunk_id))
    quit(status = 3)
  }
}

final_failed <- read_existing_results(opts$out, col_select = c("experiment_id", "sir_failed"))[sir_failed == TRUE]
if (length(unique(completed_ids)) >= total_n && nrow(final_failed) == 0) {
  append_progress(
    "compute", "completed",
    sprintf("completed SIR universe checkpoint: %d/%d; cumulative_wall=%s",
            length(unique(completed_ids)), total_n, fmt_duration(cumulative_wall_clock_s))
  )
} else {
  append_progress(
    "compute", "warning",
    sprintf("compute stopped incomplete: %d/%d; failures=%d; cumulative_wall=%s",
            length(unique(completed_ids)), total_n, nrow(final_failed), fmt_duration(cumulative_wall_clock_s))
  )
}
