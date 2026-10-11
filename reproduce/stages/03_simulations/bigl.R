
suppressMessages({
  library(BIGL)
  library(parallel)
  library(data.table)
})

script <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
ROOT <- normalizePath(file.path(dirname(script), "..", ".."))
OUTPUT <- file.path(ROOT, "output")
if (Sys.getenv("SIR_SMOKE", "0") == "1") OUTPUT <- file.path(OUTPUT, "smoke")
DATA <- file.path(OUTPUT, "inputs", "bigl_comparison_data")
OUT <- file.path(OUTPUT, "inputs", "bigl_raw")
MAXT <- 0L
NMAX <- as.integer(Sys.getenv("BIGL_NMAX", "0"))
stopifnot(as.character(packageVersion("BIGL")) == "1.9.3")
data.table::setDTthreads(1L)
args <- commandArgs(trailingOnly = TRUE)
configs <- strsplit(args[1], ",")[[1]]
pattern <- args[2]
cores <- max(1L, min(16L, as.integer(args[3])))
BB <- if (length(args) >= 4) as.integer(args[4]) else 200L
CHUNK <- if (length(args) >= 5) as.integer(args[5]) else 50L
ALPHAS <- c(0.10, 0.05, 0.01)

read_doses <- function(p) {
  txt <- paste(readLines(p, warn = FALSE), collapse = " ")
  m <- regmatches(txt, regexpr('"doses": \\[[^]]*\\]', txt))
  as.numeric(strsplit(gsub('"doses": \\[|\\]', "", m), ",")[[1]])
}
DOSES_SYN <- read_doses(file.path(DATA, "meta.json"))

ship_bliss <- get("Blissindependence", envir = asNamespace("BIGL"))
bliss_viab <- function(doseInput, parmInput, ...) {
  pars <- parmInput
  L4 <- get("L4", envir = asNamespace("BIGL"))
  f1 <- L4(doseInput[["d1"]], b = pars["h1"], logEC50 = pars["e1"], L = 0, U = 1)
  f2 <- L4(doseInput[["d2"]], b = pars["h2"], logEC50 = pars["e2"], L = 0, U = 1)
  y1 <- pars["b"] + (pars["m1"] - pars["b"]) * f1
  y2 <- pars["b"] + (pars["m2"] - pars["b"]) * f2
  unname(y1 * y2 / pars["b"])
}
set_null <- function(v) {
  f <- if (v == "vb") bliss_viab else ship_bliss
  environment(f) <- asNamespace("BIGL")
  assignInNamespace("Blissindependence", f, ns = "BIGL")
}

cfg_of <- function(name, r) {
  switch(name,
    ship_eq = list(null = "ship", method = "equal", error = if (r >= 2) 4 else 1),
    ship_eq_residual = list(null = "ship", method = "equal", error = 4),
    vb_eq = list(null = "vb", method = "equal", error = if (r >= 2) 4 else 1),
    ship_model = list(null = "ship", method = "model", error = 4),
    vb_model = list(null = "vb", method = "model", error = 4),
    stop("unknown config ", name))
}

read_file <- function(fn) {
  base <- sub("\\.f64$", "", fn)
  info <- file.info(file.path(DATA, fn))$size / 8
  seeds <- readBin(file.path(DATA, paste0(base, ".seeds.i32")), "integer", n = 1e7, size = 4, endian = "little")
  n <- length(seeds)
  r <- info / (n * 25)
  Y <- readBin(file.path(DATA, fn), "double", n = info, size = 8, endian = "little")
  dim(Y) <- c(5, 5, r, n)
  dp <- file.path(DATA, paste0(base, ".doses.f64"))
  doses <- NULL
  if (file.exists(dp)) {
    dd <- readBin(dp, "double", n = n * 10, size = 8, endian = "little")
    dim(dd) <- c(5, 2, n)
    doses <- dd
  }
  list(Y = Y, n = n, r = r, doses = doses, seeds = seeds)
}

make_df <- function(Y, t, r, d1v, d2v) {
  g <- expand.grid(j = 1:5, i = 1:5, k = 1:r)
  eff <- Y[cbind(g$j, g$i, g$k, t)]
  df <- data.frame(effect = eff, d1 = d1v[g$i], d2 = d2v[g$j])
  key <- apply(df[, c("d1", "d2")], 1, paste, collapse = "_")
  df <- df[order(key), , drop = FALSE]
  rownames(df) <- NULL
  df
}

ecdf_values <- function(f) {
  e <- environment(f)
  cnt <- round(diff(c(0, e$y)) * e$nobs)
  rep(e$x, cnt)
}

capture <- function(expr) {
  w <- character(0)
  val <- withCallingHandlers(
    tryCatch(expr, error = function(e) e),
    warning = function(x) {
      w <<- c(w, conditionMessage(x))
      invokeRestart("muffleWarning")
    })
  list(val = val, warn = w)
}

one <- function(df, cfg, seed) {
  row <- list(fit_err = 0L, fit_msg = "", fit_nwarn = 0L, fit_wmsg = "", surf_err = 0L, surf_msg = "",
              surf_nwarn = 0L, surf_wmsg = "", t_fit = NA_real_, t_surf = NA_real_)
  set.seed(seed)
  t0 <- proc.time()[["elapsed"]]
  cf <- capture(fitMarginals(df, method = "nlslm"))
  row$t_fit <- proc.time()[["elapsed"]] - t0
  row$fit_nwarn <- length(cf$warn)
  if (length(cf$warn)) row$fit_wmsg <- substr(cf$warn[1], 1, 120)
  if (inherits(cf$val, "error")) {
    row$fit_err <- 1L
    row$fit_msg <- substr(conditionMessage(cf$val), 1, 120)
    return(row)
  }
  fit <- cf$val
  co <- coef(fit)
  for (nm in c("h1", "h2", "b", "m1", "m2", "e1", "e2")) row[[paste0("c_", nm)]] <- unname(co[nm])
  row$sigma0 <- fit$sigma
  row$df0 <- fit$df
  set.seed(seed)
  t0 <- proc.time()[["elapsed"]]
  cs <- capture(fitSurface(df, fit, null_model = "bliss", statistic = "both", B.CP = 50, B.B = BB,
                           method = cfg$method, error = cfg$error, confInt = FALSE, cutoff = 0.95,
                           progressBar = FALSE, parallel = FALSE))
  row$t_surf <- proc.time()[["elapsed"]] - t0
  row$surf_nwarn <- length(cs$warn)
  if (length(cs$warn)) row$surf_wmsg <- substr(cs$warn[1], 1, 120)
  if (inherits(cs$val, "error")) {
    row$surf_err <- 1L
    row$surf_msg <- substr(conditionMessage(cs$val), 1, 120)
    return(row)
  }
  rs <- cs$val
  mR <- rs$meanR
  Fb <- ecdf_values(mR$FDist)
  row$meanR_F <- mR$FStat
  row$meanR_n1 <- mR$n1
  row$meanR_B <- length(Fb)
  row$meanR_nge <- sum(Fb >= mR$FStat)
  row$meanR_p <- mR$p.value
  row$meanR_pasym <- pf(mR$FStat, mR$n1, mR$df0, lower.tail = FALSE)
  Ym <- rs$maxR$Ymean
  M <- ecdf_values(attr(Ym, "distr"))
  R <- Ym$R
  absR <- Ym$absR
  raw <- Ym[[5]]
  inc <- co["m1"] >= co["b"] && co["m2"] >= co["b"]
  dec <- co["m1"] <= co["b"] && co["m2"] <= co["b"]
  eqq <- co["m1"] == co["b"] && co["m2"] == co["b"]
  syn <- if (eqq) rep(FALSE, length(R)) else if (inc) R > 0 else if (dec) R < 0 else rep(FALSE, length(R))
  ant <- if (eqq) rep(FALSE, length(R)) else if (inc) R < 0 else if (dec) R > 0 else rep(FALSE, length(R))
  row$dir <- if (eqq) "flat" else if (inc) "inc" else if (dec) "dec" else "diverging"
  row$maxR_call05 <- rs$maxR$Call
  row$maxR_B <- length(M)
  row$maxR_maxabs <- max(absR)
  row$maxR_maxsyn <- if (any(syn)) max(absR[syn]) else 0
  row$maxR_maxant <- if (any(ant)) max(absR[ant]) else 0
  row$maxR_nge_abs <- sum(M >= row$maxR_maxabs)
  row$maxR_nge_syn <- if (any(syn)) sum(M >= row$maxR_maxsyn) else length(M)
  n1 <- length(R)
  pmx <- function(x) 1 - (2 * pt(x, fit$df) - 1)^n1
  row$maxR_pasym_abs <- pmx(row$maxR_maxabs)
  row$maxR_pasym_syn <- if (any(syn)) pmx(row$maxR_maxsyn) else 1
  for (a in ALPHAS) {
    q <- quantile(M, 1 - a)
    tag <- sprintf("%03d", round(100 * a))
    row[[paste0("maxR_rej_any_", tag)]] <- as.integer(row$maxR_maxabs > q)
    row[[paste0("maxR_rej_syn_", tag)]] <- as.integer(any(syn) && row$maxR_maxsyn > q)
    row[[paste0("maxR_rej_ant_", tag)]] <- as.integer(any(ant) && row$maxR_maxant > q)
    row[[paste0("maxR_nsyn_", tag)]] <- sum(syn & absR > q)
  }
  row$meanR_rawmean <- mean(raw)
  row
}

run_task <- function(task) {
  outf <- task$out
  if (file.exists(outf)) return(invisible(NULL))
  cfgname <- task$cfg
  dat <- read_file(task$file)
  cfg <- cfg_of(cfgname, dat$r)
  set_null(cfg$null)
  rows <- vector("list", 0)
  for (t in task$idx) {
    if (is.null(dat$doses)) {
      d1v <- DOSES_SYN
      d2v <- DOSES_SYN
    } else {
      d1v <- dat$doses[, 1, t]
      d2v <- dat$doses[, 2, t]
    }
    df <- make_df(dat$Y, t, dat$r, d1v, d2v)
    t0 <- proc.time()[["elapsed"]]
    row <- one(df, cfg, dat$seeds[t])
    row$t_total <- proc.time()[["elapsed"]] - t0
    row$idx <- t - 1L
    row$file <- task$file
    row$config <- cfgname
    row$error_type <- cfg$error
    row$BB <- BB
    rows[[length(rows) + 1]] <- row
  }
  res <- rbindlist(rows, fill = TRUE)
  tmp <- paste0(outf, ".tmp")
  fwrite(res, tmp)
  file.rename(tmp, outf)
  invisible(NULL)
}

files <- sort(list.files(DATA, pattern = "\\.f64$"))
files <- files[!grepl("\\.doses\\.f64$", files)]
files <- files[grepl(pattern, files)]
tasks <- list()
for (cfgname in configs) {
  dir.create(file.path(OUT, cfgname), recursive = TRUE, showWarnings = FALSE)
  for (fn in files) {
    base <- sub("\\.f64$", "", fn)
    n <- length(readBin(file.path(DATA, paste0(base, ".seeds.i32")), "integer", n = 1e7, size = 4))
    r <- file.info(file.path(DATA, fn))$size / 8 / (n * 25)
    if (cfgname %in% c("ship_model", "vb_model") && r < 2) next
    if (NMAX > 0) n <- min(n, NMAX)
    for (s in seq(1, n, by = CHUNK)) {
      idx <- s:min(n, s + CHUNK - 1)
      tasks[[length(tasks) + 1]] <- list(cfg = cfgname, file = fn, idx = idx,
                                         out = file.path(OUT, cfgname, sprintf("%s__%05d.csv", base, s - 1)))
    }
  }
}
todo <- Filter(function(tk) !file.exists(tk$out), tasks)
if (MAXT > 0) todo <- head(todo, MAXT)
cat(sprintf("%s: %d tasks (%d to do), cores %d, B.B %d\n", format(Sys.time()), length(tasks), length(todo), cores, BB))
t0 <- proc.time()[["elapsed"]]
res <- mclapply(todo, function(tk) {
  r <- try(run_task(tk), silent = TRUE)
  if (inherits(r, "try-error")) cat("TASK FAILED", tk$out, as.character(r), "\n")
  !inherits(r, "try-error")
}, mc.cores = cores, mc.preschedule = FALSE)
if (length(res) && !all(unlist(res))) stop("One or more BIGL tasks failed")
cat(sprintf("%s: done in %.0f s\n", format(Sys.time()), proc.time()[["elapsed"]] - t0))
