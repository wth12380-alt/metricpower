# =============================================================================
#  exact_oc.R
#
#  Exact (Monte Carlo-calibrated) operating characteristics for every
#  implemented variance - the finite-sample companion to Table S4.
#
#  Table S4 asks: at the ANALYTIC sample size n_asym, how do the large-sample
#  Wald / score tests behave?  This script asks the complementary question:
#  if the test is calibrated exactly from the finite-sample distribution of the
#  statistic, (a) what power does the analytic n actually deliver, and (b) how
#  many cases are really needed?
#
#  Exact test.  The statistic is the pivot T = (estimate - theta_0) / SE with the
#  same plug-in SE the package reports.  Its critical value c*(n) is the
#  empirical 1 - alpha quantile of |T| (two-sided; |T| folds both tails) from REPS
#  simulated studies of size n
#  under the null, instead of the normal or t quantile, so the type I error is
#  alpha by construction up to Monte Carlo error.  Two ways of simulating the null:
#    null-simulated   confusion-matrix metrics: data drawn at the null operating
#                     point p_0 with g(p_0) = theta_0 (cell_prob_at_metric, the
#                     same path as the score test), T = (est - theta_0) / SE
#    pivot-calibrated all other families (no null generator in the catalogue):
#                     data drawn at the truth, T = (est - theta_1) / SE; this is
#                     the parametric-bootstrap calibration of the Wald statistic
#  Columns
#    n_asym           analytic sample size (z Wald, t for per-case means, folded
#                     normal for volumetric similarity), as in Table S4
#    power_exact_asym exact power at n_asym (data at theta_1, exact c*)
#    n_exact          smallest n at which the exact power reaches POWER
#                     (bisection, REPS_SEARCH reps per evaluation, then a final
#                     evaluation at REPS); resolution about 1% of n
#    ratio_n          n_exact / n_asym  (> 1: the analytic n is too small)
#    type1_exact      type I error of the exact test at n_exact from an
#                     INDEPENDENT null sample (check; should be alpha +/- MC)
#    power_exact      exact power at n_exact (REPS reps)
#    crit_asym        c*(n_asym), to compare with z = 1.96 or the t quantile
#                     (volumetric similarity: on the scale of |est - theta_0|, not studentized)
#
#  Usage (package loaded):
#    source("exact_oc.R")                             # REPS 20,000, REPS_SEARCH 4,000
#    REPS <- 4000; REPS_SEARCH <- 1000; source("exact_oc.R")   # quick look
#    ONLY <- c("dice", "sensitivity"); source("exact_oc.R")
#  Writes exact_oc.csv; if operating_characteristics.csv is present, its
#  type1_wald / power_wald columns are merged in for side-by-side comparison.
# =============================================================================

SCRIPT_VERSION <- "2026-11-21"
if (!exists("REPS"))        REPS        <- 20000
if (!exists("REPS_SEARCH")) REPS_SEARCH <- 4000
if (!exists("DELTA_FRAC"))  DELTA_FRAC  <- 0.10
if (!exists("DELTA_ABS"))   DELTA_ABS   <- NA
if (!exists("ALPHA"))       ALPHA       <- 0.05
if (!exists("POWER"))       POWER       <- 0.80
if (!exists("MAX_N"))       MAX_N       <- 60000
if (!exists("EXCLUDE"))     EXCLUDE     <- c("entropy", "conditional_entropy", "perplexity")   # Part II
if (!exists("ONLY"))        ONLY        <- NULL

source(if (file.exists("oc_catalogue.R")) "oc_catalogue.R"
       else system.file("validation", "oc_catalogue.R", package = "metricpower"))

Z_A <- stats::qnorm(1 - ALPHA / 2)
SE_RATE <- sqrt(ALPHA * (1 - ALPHA) / REPS)
SE_POW  <- sqrt(POWER * (1 - POWER) / REPS)

# pivot |T| from one simulator: (est - centre) / se, undefined replicates dropped
# |T| from one simulator: (est - centre) / se, undefined estimates dropped; se = 0 with est != centre
# gives |T| = Inf (the Wald test rejects).  studentize = FALSE uses |est - centre| itself (volumetric
# similarity: the plug-in SE is 0 whenever fp = fn, so the studentized pivot is degenerate there).
pivot <- function(sim, n, reps, centre, studentize = TRUE) {
  es <- sim$est_se(n, reps)
  ok <- is.finite(es[, 1])
  d <- abs(es[ok, 1] - centre)
  if (studentize) { se <- es[ok, 2]; T <- ifelse(se > 0, d / se, ifelse(d > 0, Inf, 0)) } else T <- d
  list(T = T[is.finite(T) | T == Inf], undefined = mean(!ok))
}

rows <- list(); t0 <- Sys.time()
for (fam in c("confusion", "multiclass", "rate", "per-case", "U-statistic", "other")) {
  for (nm in names(CATALOGUE)) {
    e <- CATALOGUE[[nm]]; if (e$family != fam) next
    if (!is.null(ONLY) && !(nm %in% ONLY)) next
    if (nm %in% EXCLUDE) next
    s <- if (!is.null(e$sim)) e$sim else e$make()
    th1 <- s$truth; V1 <- s$V
    delta <- if (!is.na(DELTA_ABS)) DELTA_ABS else DELTA_FRAC * abs(th1)
    row <- list(metric = nm, family = fam, theta = th1, delta = delta)
    if (delta <= 0) { row$note <- "theta = 0: relative delta undefined"; rows[[nm]] <- row; next }
    th0 <- th1 - delta
    t_based <- fam == "per-case"
    folded  <- nm == "volumetric_similarity"

    # --- analytic n, as in Table S4 ----------------------------------------
    n_asym <- if (folded) n_for_power_vs_folded(s$p, delta, alpha = ALPHA, power = POWER)
              else if (t_based) n_for_power_t(V1, delta, alpha = ALPHA, power = POWER)
              else n_for_power(V1, delta, alpha = ALPHA, power = POWER)
    row$n_asym <- n_asym
    if (n_asym > MAX_N) { row$note <- "n above MAX_N, skipped"; rows[[nm]] <- row; next }

    # --- null simulator -----------------------------------------------------
    sim0 <- NULL
    if (fam == "confusion") {
      a <- e$args; if (is.null(a)) a <- list()
      for (pth in c("sensitivity", "specificity")) {
        p0 <- tryCatch(do.call(cell_prob_at_metric, c(list(p = s$p, name = nm, target = th0, path = pth), a)),
                       error = function(err) NULL)
        if (!is.null(p0)) break
      }
      if (!is.null(p0))
        sim0 <- sim_multinom(p0, do.call(get_metric, c(list(nm), a)), do.call(get_gradient, c(list(nm), a)))
    }
    method <- if (!is.null(sim0)) "null-simulated" else "pivot-calibrated"
    row$method <- method
    stud <- !folded
    crit_fn <- function(n, reps) {            # two-sided: |T| folds both tails, so its 1 - alpha quantile
      pv <- if (!is.null(sim0)) pivot(sim0, n, reps, th0, stud) else pivot(s, n, reps, th1, stud)
      stats::quantile(pv$T, 1 - ALPHA, names = FALSE)
    }
    power_fn <- function(n, reps) {           # data at the truth, H0: theta = theta_0, exact c*
      cstar <- crit_fn(n, reps)
      pv <- pivot(s, n, reps, th0, stud)
      list(power = mean(pv$T > cstar), crit = cstar, undefined = pv$undefined)
    }

    # --- (a) exact power at the analytic n ----------------------------------
    pa <- power_fn(n_asym, REPS)
    row$crit_asym <- pa$crit; row$power_exact_asym <- pa$power; row$undefined_pct <- 100 * pa$undefined

    # --- (b) smallest n with exact power >= POWER (bisection) ---------------
    n_min <- if (t_based) 3 else 2
    lo <- NA; hi <- NA
    if (pa$power >= POWER) {
      hi <- n_asym; cand <- n_asym
      repeat {
        cand <- max(n_min, floor(cand * 0.8))
        if (power_fn(cand, REPS_SEARCH)$power < POWER || cand == n_min) { lo <- cand; break }
        hi <- cand
      }
      if (lo == hi) lo <- max(n_min - 1, lo - 1)
    } else {
      lo <- n_asym; cand <- n_asym
      repeat {
        cand <- ceiling(cand * 1.25)
        if (cand > MAX_N) break
        if (power_fn(cand, REPS_SEARCH)$power >= POWER) { hi <- cand; break }
        lo <- cand
      }
    }
    if (is.na(hi)) {
      row$note <- "exact n above MAX_N"; rows[[nm]] <- row; next
    }
    while (hi - lo > max(1, 0.01 * hi)) {
      mid <- floor((lo + hi) / 2)
      if (power_fn(mid, REPS_SEARCH)$power >= POWER) hi <- mid else lo <- mid
    }
    n_exact <- hi
    pe <- power_fn(n_exact, REPS)
    row$n_exact <- n_exact; row$ratio_n <- n_exact / n_asym; row$power_exact <- pe$power
    # independent null sample for the type I error of the exact test at n_exact
    cstar <- crit_fn(n_exact, REPS)
    pv0 <- if (!is.null(sim0)) pivot(sim0, n_exact, REPS, th0, stud) else pivot(s, n_exact, REPS, th1, stud)
    row$type1_exact <- mean(pv0$T > cstar)
    rows[[nm]] <- row
    cat(sprintf("%-26s%-12s%-17s n_asym=%6d  power_exact(n_asym)=%.3f  n_exact=%6d  ratio=%.3f  typeI=%.3f  c*=%.3f\n",
                nm, fam, method, n_asym, pa$power, n_exact, n_exact / n_asym, row$type1_exact, pa$crit))
    flush.console()
  }
}

# -----------------------------------------------------------------------------
# Table and summary
# -----------------------------------------------------------------------------
if (length(rows) == 0) stop("no metric was run - check ONLY (", paste(ONLY, collapse = ", "), ") and EXCLUDE (", paste(EXCLUDE, collapse = ", "), ") in your workspace; rm(ONLY) runs everything")
cols <- c("metric", "family", "method", "theta", "delta", "n_asym", "crit_asym", "power_exact_asym",
          "n_exact", "ratio_n", "type1_exact", "power_exact", "undefined_pct", "note")
tab <- do.call(rbind, lapply(rows, function(r) {
  as.data.frame(lapply(setNames(cols, cols), function(k) if (is.null(r[[k]])) NA else r[[k]]),
                stringsAsFactors = FALSE) }))
if (file.exists("operating_characteristics.csv")) {
  oc <- utils::read.csv("operating_characteristics.csv", stringsAsFactors = FALSE)
  keep <- intersect(c("metric", "type1_wald", "power_wald", "n_score", "type1_score", "power_score"), names(oc))
  tab <- merge(tab, oc[, keep], by = "metric", all.x = TRUE, sort = FALSE)
  tab <- tab[order(match(tab$metric, names(rows))), ]
}
utils::write.csv(tab, "exact_oc.csv", row.names = FALSE)

done <- tab[!is.na(tab$n_exact), ]
cat("\n", strrep("=", 118), "\n", sep = "")
cat(sprintf("EXACT OPERATING CHARACTERISTICS   %d metrics   REPS = %s (search %s)   script %s   delta = %s   alpha = %s   power = %s\n",
            nrow(done), format(REPS, big.mark = ","), format(REPS_SEARCH, big.mark = ","), SCRIPT_VERSION,
            if (!is.na(DELTA_ABS)) sprintf("%s (absolute)", DELTA_ABS) else sprintf("%.0f%% of the metric value", 100 * DELTA_FRAC),
            ALPHA, POWER))
cat(sprintf("Monte Carlo error (2 SE): type I error +/- %.3f   power +/- %.3f   n_exact about +/- 1.5%%\n", 2 * SE_RATE, 2 * SE_POW))
cat(strrep("=", 118), "\n", sep = "")
show <- intersect(c("metric", "family", "method", "n_asym", "crit_asym", "power_wald", "power_exact_asym",
                    "n_exact", "ratio_n", "type1_exact", "power_exact"), names(tab))
print(tab[, show], digits = 3, row.names = FALSE)
cat(strrep("=", 118), "\nSUMMARY\n", sep = "")
cat(sprintf("  exact power at the analytic n: %.3f to %.3f; %d of %d below %.2f\n",
            min(done$power_exact_asym), max(done$power_exact_asym), sum(done$power_exact_asym < POWER - 0.02), nrow(done), POWER - 0.02))
cat(sprintf("  n_exact / n_asym: %.3f to %.3f; median %.3f; %d metrics need > 5%% more cases than the analytic n: %s\n",
            min(done$ratio_n), max(done$ratio_n), stats::median(done$ratio_n), sum(done$ratio_n > 1.05),
            paste(sprintf("%s (%.2f)", done$metric[done$ratio_n > 1.05], done$ratio_n[done$ratio_n > 1.05]), collapse = ", ")))
cat(sprintf("  type I error of the exact test (independent null sample): %.3f to %.3f\n", min(done$type1_exact), max(done$type1_exact)))
cat(sprintf("  exact critical value at n_asym: %.2f to %.2f (normal: %.2f)\n", min(done$crit_asym), max(done$crit_asym), Z_A))
for (fam in unique(done$family)) {
  d <- done[done$family == fam, ]
  cat(sprintf("  %-12s %2d metrics: ratio_n %.3f to %.3f (median %.3f)\n", fam, nrow(d), min(d$ratio_n), max(d$ratio_n), stats::median(d$ratio_n)))
}
cat(sprintf("(%.0f s)\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
