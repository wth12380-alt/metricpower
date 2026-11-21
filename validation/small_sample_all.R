# =============================================================================
#  small_sample_all.R
#
#  Small-sample validity (Table S5) for EVERY implemented variance, not only
#  the 17 confusion-matrix metrics and two per-case scores.  Same
#  data-generating processes as operating_characteristics.R (oc_catalogue.R).
#
#  For each metric and each n in NS, REPS studies of size n are simulated at the
#  planning operating point (truth theta, analytic per-unit variance V) and
#    ratio        n * Var_MC(estimate) / V      empirical / analytic variance
#                 (1 = exact; > 1 = analytic variance understates the truth;
#                  MC standard error sqrt(2 / REPS) = 1% at 20,000)
#    coverage     share of 95% Wald intervals estimate +/- crit * SE that contain
#                 theta (t critical value for per-case means, z otherwise)
#    miss_below   share of intervals lying entirely below theta
#    miss_above   share lying entirely above theta
#    miss_ratio   max(miss_below, miss_above) / min(...)  (1 = symmetric)
#    skew_mc      skewness of the simulated estimates
#    undefined    % of replicates with an undefined estimate (0/0 cells); dropped
#    se_zero      % with a plug-in SE of 0 (an empty cell); kept - the estimate
#                 enters the variance and the zero-width interval counts as a miss
#  Volumetric similarity is also compared with the folded-normal variance
#  (ratio_folded), which is the relevant one near fp = fn.
#
#  Usage (package loaded):
#    source("small_sample_all.R")                       # 20,000 reps, all metrics
#    REPS <- 2000; source("small_sample_all.R")         # quick look
#    ONLY <- c("dice", "brier"); source("small_sample_all.R")
#    NS <- c(25, 50, 100); source("small_sample_all.R")
#    FROM_CSV <- TRUE; source("small_sample_all.R")   # re-make the tables from an existing csv
#  Writes small_sample_all.csv (long), small_sample_ratio.csv and
#  small_sample_coverage.csv (wide, metric x n, ready for the supplement) and
#  prints the sign tests by family.
# =============================================================================

SCRIPT_VERSION <- "2026-11-21"
if (!exists("REPS"))    REPS    <- 20000
if (!exists("NS"))      NS      <- c(25, 50, 100, 250, 500, 1000)
if (!exists("ALPHA"))   ALPHA   <- 0.05
if (!exists("EXCLUDE")) EXCLUDE <- c("entropy", "conditional_entropy", "perplexity")   # Part II
if (!exists("ONLY"))    ONLY    <- NULL
if (!exists("BAND"))    BAND    <- 2 * sqrt(2 / REPS)   # MC noise band for a variance ratio (2 SE)
if (!exists("FROM_CSV")) FROM_CSV <- FALSE   # TRUE: skip the simulation, rebuild the tables from small_sample_all.csv

source(if (file.exists("oc_catalogue.R")) "oc_catalogue.R"
       else system.file("validation", "oc_catalogue.R", package = "metricpower"))

Z_A    <- stats::qnorm(1 - ALPHA / 2)
SE_V   <- sqrt(2 / REPS)
SE_COV <- sqrt((1 - ALPHA) * ALPHA / REPS)

skewness <- function(x) { x <- x - mean(x); mean(x^3) / (mean(x^2))^1.5 }

t0 <- Sys.time()
if (FROM_CSV && file.exists("small_sample_all.csv")) {
  tab <- utils::read.csv("small_sample_all.csv", stringsAsFactors = FALSE)   # tables and summary only
} else {
rows <- list()
for (fam in c("confusion", "multiclass", "rate", "per-case", "U-statistic", "other")) {
  for (nm in names(CATALOGUE)) {
    e <- CATALOGUE[[nm]]; if (e$family != fam) next
    if (!is.null(ONLY) && !(nm %in% ONLY)) next
    if (nm %in% EXCLUDE) next
    s <- if (!is.null(e$sim)) e$sim else e$make()
    th <- s$truth; V <- s$V
    t_based <- fam == "per-case"
    folded  <- nm == "volumetric_similarity"
    out <- character(0)
    for (n in NS) {
      es <- s$est_se(n, REPS)
      fin <- is.finite(es[, 1])                       # estimate defined (0/0 cells drop out)
      est <- es[fin, 1]; se <- es[fin, 2]
      okci <- is.finite(se)                           # SE = 0 kept: a degenerate interval that misses
      crit <- if (t_based) stats::qt(1 - ALPHA / 2, n - 1) else Z_A
      lo <- est[okci] - crit * se[okci]; hi <- est[okci] + crit * se[okci]
      r <- data.frame(metric = nm, family = fam, n = n,
                      theta = th, V = V,
                      ratio = n * stats::var(est) / V,
                      ratio_folded = if (folded) n * stats::var(est) / vs_variance_folded(s$p, n)$V else NA,
                      coverage   = mean(lo <= th & th <= hi),
                      miss_below = mean(hi < th),
                      miss_above = mean(lo > th),
                      skew_mc    = skewness(est),
                      undefined_pct = 100 * mean(!fin),
                      se_zero_pct   = 100 * mean(se[okci] == 0),
                      n_ok = sum(fin), stringsAsFactors = FALSE)
      r$miss_ratio <- if (min(r$miss_below, r$miss_above) > 0)
        max(r$miss_below, r$miss_above) / min(r$miss_below, r$miss_above) else NA
      rows[[length(rows) + 1]] <- r
      out <- c(out, sprintf("n=%4d %.3f/%.3f", n, r$ratio, r$coverage))
    }
    cat(sprintf("%-26s%-12s %s\n", nm, fam, paste(out, collapse = "  "))); flush.console()
  }
}
if (length(rows) == 0) stop("no metric was run - check ONLY (", paste(ONLY, collapse = ", "), ") and EXCLUDE (", paste(EXCLUDE, collapse = ", "), ") in your workspace; rm(ONLY) runs everything")
tab <- do.call(rbind, rows)
utils::write.csv(tab, "small_sample_all.csv", row.names = FALSE)
}

# ---- wide tables (metric x n) -------------------------------------------------
wide <- function(col, digits) {
  mets <- unique(tab$metric)
  w <- data.frame(metric = mets, family = tab$family[match(mets, tab$metric)], stringsAsFactors = FALSE)
  for (n in NS) {
    d <- tab[tab$n == n, ]
    w[[paste("n =", format(n, big.mark = ","))]] <- round(d[[col]][match(mets, d$metric)], digits)
  }
  w
}
ratio_w <- wide("ratio", 3); cov_w <- wide("coverage", 3)
utils::write.csv(ratio_w, "small_sample_ratio.csv", row.names = FALSE)
utils::write.csv(cov_w,   "small_sample_coverage.csv", row.names = FALSE)

# ---- summary ------------------------------------------------------------------
cat("\n", strrep("=", 110), "\n", sep = "")
cat(sprintf("SMALL-SAMPLE VALIDITY   %d metrics   REPS = %s   n = %s   script %s\n",
            length(unique(tab$metric)), format(REPS, big.mark = ","), paste(NS, collapse = ", "), SCRIPT_VERSION))
cat(sprintf("Monte Carlo error (2 SE): variance ratio +/- %.1f%%   coverage +/- %.3f\n", 200 * SE_V, 2 * SE_COV))
cat(strrep("=", 110), "\n", sep = "")
cat("\nVariance ratio (empirical / analytic):\n"); print(ratio_w, row.names = FALSE)
cat("\nCoverage of the 95% Wald interval:\n");     print(cov_w,   row.names = FALSE)

cat("\nSign tests, H0: ratio equally likely above or below 1 (by family and n):\n")
cat(sprintf("  %-14s %6s  %8s  %10s  %12s  %8s\n", "family", "n", "k > 1", "of", "P (binom)", "max |dev|"))
for (fam in unique(tab$family)) for (n in NS) {
  d <- tab[tab$family == fam & tab$n == n, ]
  if (nrow(d) == 0) next
  k <- sum(d$ratio > 1); m <- nrow(d)
  p <- if (m >= 1) stats::binom.test(k, m, 0.5)$p.value else NA
  cat(sprintf("  %-14s %6d  %8d  %10d  %12s  %7.1f%%\n", fam, n, k, m,
              if (m >= 5) formatC(p, digits = 3, format = "g") else "(m < 5)", 100 * max(abs(d$ratio - 1))))
}
cat("\nAll metrics pooled:\n")
for (n in NS) {
  d <- tab[tab$n == n, ]; k <- sum(d$ratio > 1); m <- nrow(d)
  cat(sprintf("  n = %4d: ratio > 1 for %d of %d (sign test P = %s); %d of %d within +/- %.0f%%; range %.3f to %.3f\n",
              n, k, m, formatC(stats::binom.test(k, m, 0.5)$p.value, digits = 3, format = "g"),
              sum(abs(d$ratio - 1) <= BAND), m, 100 * BAND, min(d$ratio), max(d$ratio)))
}
cat("\nCoverage below 0.94 (2 SE below nominal at 20,000 reps is 0.947; 0.94 leaves a margin):\n")
low <- tab[tab$coverage < 0.94, ]
if (nrow(low)) {
  print(low[, c("metric", "family", "n", "coverage", "miss_below", "miss_above", "skew_mc")], row.names = FALSE, digits = 3)
} else cat("  none\n")
cat(sprintf("\n(%.0f s)\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
