# Persistent R worker for the R-package baselines in src/baselines.py (riskscores, riskscores_cd,
# l0learn_seqround). Python starts one per worker process (so R's start-up and package loading fall in the
# harness's untimed warm-up) and sends one request per line on stdin:
#     <prefix> <method> <k> <time_limit> <seed>
# where <prefix>.X (n x d float64, column-major), <prefix>.y (n float64 in {0, 1}) and <prefix>.dims ("n d")
# were written by Python. It writes <prefix>.out, one row per candidate model ("b0 beta_1 ... beta_d") with
# 1..k nonzero coefficients, then a last line "OK", and prints DONE on stdout.
#   method annealscore | riskcd: riskscores::risk_mod (integer points in [-5, 5]) along a lambda0 path;
#                                b0 is lambda0 (unused).
#   method l0learn:              L0Learn::L0Learn.fit, logistic L0L2 path with CDPSI swaps (unbounded: L0Learn
#                                has no box constraints for CDPSI), the last solution of each support per gamma;
#                                b0 is the intercept, beta the real-valued coefficients (rounded in Python).
suppressPackageStartupMessages({ library(riskscores); library(L0Learn) })
con <- file("stdin", "r")

risk_one <- function(X1, y, l0, method, seed) {
  set.seed(seed)
  m <- tryCatch(suppressWarnings(risk_mod(X1, y, lambda0 = l0, a = -5, b = 5, method = method, seed = seed)),
                error = function(e) NULL)
  if (is.null(m)) return(NULL)
  b <- unname(m$beta[-1])
  b[is.na(b)] <- 0
  b
}

riskscores_path <- function(X, y, method, k, tl, seed, t0) {
  n <- nrow(X); d <- ncol(X)
  X1 <- cbind(rep(1, n), X)   # explicit intercept column, so a constant feature is never taken for one
  colnames(X1) <- c("Intercept", paste0("x", seq_len(d)))
  # cv_risk_mod's default lambda0 grid: 25 values from lambda_max down to 1e-4 lambda_max
  sdv <- sqrt(apply(X, 2, stats::var))
  Xs <- sweep(sweep(X, 2, colMeans(X)), 2, sdv, "/")
  yw <- ifelse(y == 0, -mean(y == 1), mean(y == 0))
  lmax <- max(abs(colSums(Xs * yw)), na.rm = TRUE) / n
  grid <- exp(seq(log(lmax), log(lmax * 1e-4), length.out = 25))
  rows <- list(); seen <- character(0)
  out_of_time <- function() proc.time()[["elapsed"]] - t0 > 0.9 * tl
  keep <- function(l0, b) {
    nz <- sum(b != 0)
    if (nz >= 1 && nz <= k) {
      key <- paste(b, collapse = ",")
      if (!(key %in% seen)) { seen <<- c(seen, key); rows[[length(rows) + 1]] <<- c(l0, b) }
    }
    nz
  }
  # walk the grid from the largest lambda0 (sparsest) down; stop after two models with more than k points
  feas_lo <- NA; infeas_hi <- NA; best_nz <- 0; over <- 0
  for (l0 in grid) {
    if (out_of_time()) break
    b <- risk_one(X1, y, l0, method, seed)
    if (is.null(b)) next
    nz <- keep(l0, b)
    if (nz <= k) { feas_lo <- l0; best_nz <- max(best_nz, nz); over <- 0 }
    else { if (is.na(infeas_hi)) infeas_hi <- l0; over <- over + 1; if (over >= 2) break }
  }
  # bisection on log lambda0 between the smallest feasible and the first infeasible grid value, to reach
  # a model with exactly k points when the grid jumps past k
  if (!is.na(feas_lo) && !is.na(infeas_hi) && best_nz < k && infeas_hi < feas_lo) {
    lo <- log(infeas_hi); hi <- log(feas_lo)
    for (it in 1:8) {
      if (out_of_time()) break
      mid <- 0.5 * (lo + hi)
      b <- risk_one(X1, y, exp(mid), method, seed)
      if (is.null(b)) break
      nz <- keep(exp(mid), b)
      if (nz <= k) { hi <- mid; if (nz == k) break } else lo <- mid
    }
  }
  rows
}

l0learn_path <- function(X, y, k, d) {
  keepc <- which(apply(X, 2, stats::var) > 0)   # L0Learn rejects constant columns
  if (length(keepc) == 0) return(list())
  fit <- tryCatch(L0Learn.fit(X[, keepc, drop = FALSE], ifelse(y > 0, 1, -1), loss = "Logistic",
                              penalty = "L0L2", algorithm = "CDPSI", maxSuppSize = min(k, length(keepc)),
                              nGamma = 5, gammaMin = 1e-4, gammaMax = 10),
                  error = function(e) NULL)
  if (is.null(fit)) return(list())
  rows <- list()
  for (g in seq_along(fit$gamma)) {
    B <- as.matrix(fit$beta[[g]])
    supp <- apply(B != 0, 2, function(z) paste(which(z), collapse = ","))
    for (j in seq_len(ncol(B))) {
      nz <- sum(B[, j] != 0)
      last <- j == ncol(B) || supp[j + 1] != supp[j]   # the smallest lambda of each support run
      if (nz >= 1 && nz <= k && last) {
        b <- numeric(d); b[keepc] <- B[, j]
        rows[[length(rows) + 1]] <- c(fit$a0[[g]][j], b)
      }
    }
  }
  rows
}

while (length(line <- readLines(con, n = 1)) > 0) {
  t0 <- proc.time()[["elapsed"]]
  a <- strsplit(trimws(line), " ")[[1]]
  prefix <- a[1]; method <- a[2]; k <- as.integer(a[3]); tl <- as.numeric(a[4]); seed <- as.integer(a[5])
  dims <- scan(paste0(prefix, ".dims"), quiet = TRUE)
  n <- dims[1]; d <- dims[2]
  X <- matrix(readBin(paste0(prefix, ".X"), "double", n * d), nrow = n, ncol = d)
  y <- readBin(paste0(prefix, ".y"), "double", n)
  rows <- if (method == "l0learn") l0learn_path(X, y, k, d) else riskscores_path(X, y, method, k, tl, seed, t0)
  out <- paste0(prefix, ".out")
  if (length(rows)) write.table(do.call(rbind, rows), out, row.names = FALSE, col.names = FALSE)
  else file.create(out)
  cat("OK\n", file = out, append = TRUE)
  tryCatch({ cat("DONE\n"); flush(stdout()) }, error = function(e) quit(save = "no"))
}
