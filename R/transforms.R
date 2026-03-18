make_transform <- function(cfg_transform) {
  name <- cfg_transform$name
  eps <- cfg_transform$eps %||% 1e-6
  asinh_scale <- cfg_transform$asinh_scale %||% 1

  if (identical(name, "identity")) {
    list(
      name = "identity",
      forward = function(y) y,
      inverse = function(z) z
    )
  } else if (identical(name, "log")) {
    list(
      name = "log",
      forward = function(y) log(pmax(y, eps)),
      inverse = function(z) exp(z)
    )
  } else if (identical(name, "logit")) {
    list(
      name = "logit",
      forward = function(y) {
        y <- clamp01(y, eps = eps)
        log(y / (1 - y))
      },
      inverse = function(z) 1 / (1 + exp(-z))
    )
  } else if (identical(name, "asinh")) {
    list(
      name = "asinh",
      forward = function(y) asinh(y / asinh_scale),
      inverse = function(z) asinh_scale * sinh(z)
    )
  } else {
    stop("Unknown transform: ", name)
  }
}
