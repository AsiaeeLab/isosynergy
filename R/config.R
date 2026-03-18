read_config <- function(path) {
  if (!file.exists(path)) stop("Config not found: ", path)
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required. Install with: install.packages('yaml')")
  }
  cfg <- yaml::read_yaml(path)
  validate_config(cfg)
  cfg
}

validate_config <- function(cfg) {
  must_have <- function(x, key) {
    if (is.null(x[[key]])) stop("Missing config key: ", key)
  }
  must_have(cfg, "project")
  must_have(cfg, "data")
  must_have(cfg, "responses")
  must_have(cfg, "transform")
  must_have(cfg, "weights")
  must_have(cfg, "method")
  must_have(cfg, "bootstrap")
  invisible(TRUE)
}

