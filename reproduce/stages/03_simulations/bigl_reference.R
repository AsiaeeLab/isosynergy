suppressPackageStartupMessages(library(BIGL))
stopifnot(as.character(packageVersion("BIGL")) == "1.9.3")
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), "..", ".."))
output <- file.path(root, "output")
if (Sys.getenv("SIR_SMOKE", "0") == "1") output <- file.path(output, "smoke")

# Supplement S5: the two Bliss predictions at known marginal parameters.
doses <- c(0.1, 0.464158883361278, 2.15443469003188, 10)
grid <- expand.grid(d1=doses, d2=doses)
rows <- list()
for (E in c(0.3, 0.7, 0.95)) {
  parameters <- c(h1=1.5, h2=1.5, b=1, m1=1-E, m2=1-E, e1=0, e2=0)
  distributed <- BIGL:::Blissindependence(grid, parameters)
  viability <- function(x) 1-E*x^1.5/(1+x^1.5)
  product <- viability(grid$d1)*viability(grid$d2)
  rows[[length(rows)+1L]] <- data.frame(E=E, d1=grid$d1, d2=grid$d2,
    bigl_bliss=unname(distributed), viability_bliss=product,
    gap=unname(distributed)-product)
}
result <- do.call(rbind, rows)
write.csv(result, file.path(output, "simulations", "bigl_reference.csv"), row.names=FALSE)
print(aggregate(gap ~ E, result, function(x) c(mean=mean(x), max=max(x))))
