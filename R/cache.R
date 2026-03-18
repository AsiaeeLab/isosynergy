cache_key <- function(...) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("Package 'digest' is required for caching. Install with: install.packages('digest')")
  }
  digest::digest(list(...), algo = "xxhash64")
}

cache_get <- function(cache_dir, key) {
  path <- file.path(cache_dir, paste0(key, ".rds"))
  if (file.exists(path)) return(readRDS(path))
  NULL
}

cache_set <- function(cache_dir, key, value) {
  dir_create(cache_dir)
  path <- file.path(cache_dir, paste0(key, ".rds"))
  saveRDS(value, path)
  invisible(path)
}

