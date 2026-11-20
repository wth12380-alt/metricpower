# =============================================================================
#  operating_characteristics.R
#
#  Empirical type I error and power of the Wald test (and, for confusion-matrix
#  metrics, the score test) at the sample size metricpower calculates, for
#  every implemented variance.  Companion to validate_all.R: same
#  data-generating processes, one extra question - does the test behave?
#
#  For each metric, data are drawn at the planning operating point
#  (truth theta_1, analytic per-unit variance V_1) and
#    dV%        100 * (V_MC / V_1 - 1), simulated vs analytic variance at n_W
#    type I     rejection rate of H0: theta = theta_1 at nominal alpha
#               (data drawn at theta_1; = 1 - coverage of the CI). Per-case means
#               use the one-sample t-test (n from power.t.test, critical value
#               qt(1 - alpha/2, n - 1)); everything else the z-based Wald test
#               (needs metricpower >= 0.1.3: n_for_power_t, SMALL_N)
#    power      rejection rate of H0: theta = theta_0 = theta_1 - delta at
#               n_W = n_for_power(V_1, delta); target 0.80
#    power_S    confusion-matrix metrics only: score statistic standardised by
#               sqrt(V_0 / n_S) at n_S, V_0 from the null operating point
#  delta is DELTA_FRAC * |theta_1| (default 10% of the metric value); set
#  DELTA_ABS to use an absolute delta instead.
#
#  Usage (package loaded with library(metricpower) or devtools::load_all()):
#    source("operating_characteristics.R")          # 10,000 reps, ~15-30 min
#    REPS <- 1000; source("operating_characteristics.R")   # quick look
#    ONLY <- c("entropy", "conditional_entropy"); source("operating_characteristics.R")
#                                                   # rerun two rows and merge them into the CSV
#  Writes operating_characteristics.csv and prints the Results sentence.
# =============================================================================

SCRIPT_VERSION <- "2026-11-20"    # catalogue moved to oc_catalogue.R (shared with small_sample_all.R, exact_oc.R)
if (!exists("REPS"))       REPS       <- 10000
if (!exists("DELTA_FRAC")) DELTA_FRAC <- 0.10
if (!exists("DELTA_ABS"))  DELTA_ABS  <- NA        # e.g. 0.05 for an absolute delta
if (!exists("ALPHA"))      ALPHA      <- 0.05
if (!exists("POWER"))      POWER      <- 0.80
if (!exists("MAX_N"))      MAX_N      <- 60000     # skip a metric whose n is larger
if (!exists("EXCLUDE"))    EXCLUDE    <- c("entropy", "conditional_entropy", "perplexity")
                                                   # deferred to Part II (entropy and conditional entropy need a
                                                   # bias correction and a null path; perplexity is an exponential
                                                   # transform of a mean); EXCLUDE <- character(0) runs everything
if (!exists("ONLY"))       ONLY       <- NULL      # e.g. c("entropy", "conditional_entropy"): run these and
                                                   # merge their rows into an existing operating_characteristics.csv
# Shared data-generating processes (seeds the RNG, builds CATALOGUE, MM_CONST, SMALL_N_OC)
source(if (file.exists("oc_catalogue.R")) "oc_catalogue.R" else system.file("validation", "oc_catalogue.R", package = "metricpower"))

Z_A <- stats::qnorm(1 - ALPHA / 2)
SE_RATE <- sqrt(0.05 * 0.95 / REPS)   # MC standard error of a rejection rate near .05
SE_POW  <- sqrt(0.8 * 0.2 / REPS)
SE_V    <- sqrt(2 / REPS)             # MC standard error of a variance ratio

# -----------------------------------------------------------------------------
# The simulation
# -----------------------------------------------------------------------------

rows <- list(); t0 <- Sys.time()
for (fam in c("confusion", "multiclass", "rate", "per-case", "U-statistic", "other")) {
  for (nm in names(CATALOGUE)) {
    e <- CATALOGUE[[nm]]; if (e$family != fam) next
    if (!is.null(ONLY) && !(nm %in% ONLY)) next
    if (nm %in% EXCLUDE) next
    s <- if (!is.null(e$sim)) e$sim else e$make()
    th1 <- s$truth; V1 <- s$V
    delta <- if (!is.na(DELTA_ABS)) DELTA_ABS else DELTA_FRAC * abs(th1)
    row <- list(metric = nm, family = fam, theta = th1, V = V1, skew = s$skew)
    if (delta <= 0) { row$note <- "theta = 0: relative delta undefined"; rows[[nm]] <- row; next }
    # per-case means are analysed with a one-sample t-test (sample SD as SE): size with the
    # noncentral t (power.t.test) and test with qt(., n - 1); everything else is a z-based Wald test
    t_based <- fam == "per-case"
    folded  <- nm == "volumetric_similarity"       # kink at fp = fn: folded-normal test, not Wald
    n_w <- if (folded) n_for_power_vs_folded(s$p, delta, alpha = ALPHA, power = POWER)
           else if (t_based) n_for_power_t(V1, delta, alpha = ALPHA, power = POWER)
           else n_for_power(V1, delta, alpha = ALPHA, power = POWER)
    if (n_w > MAX_N) { row$n_wald <- n_w; row$note <- "n above MAX_N, skipped"; rows[[nm]] <- row; next }
    th0 <- th1 - delta
    crit <- if (t_based) stats::qt(1 - ALPHA / 2, n_w - 1) else Z_A
    es <- s$est_se(n_w, REPS)
    ok <- is.finite(es[, 1]) & is.finite(es[, 2]) & es[, 2] > 0
    est <- es[ok, 1]; se <- es[ok, 2]
    row$dV_pct      <- 100 * (n_w * stats::var(est) / V1 - 1)        # simulated vs analytic (delta-method) V
    if (folded)     # VS: also vs the folded-normal V, which is the relevant one near the kink
      row$dV_folded_pct <- 100 * (n_w * stats::var(est) / vs_variance_folded(s$p, n_w)$V - 1)
    row$n_wald      <- n_w
    mm <- if (nm %in% names(MM_CONST)) MM_CONST[[nm]] / n_w else 0
    row$test        <- if (folded) "folded" else if (t_based) "t" else if (mm > 0) "z+MM" else "z"
    row$small_n     <- as.integer(n_w < SMALL_N_OC)
    if (n_w < SMALL_N_OC) row$note <- sprintf("n < %d: large-sample approximation not converged", SMALL_N_OC)
    if (folded) {
      # equal-tailed folded-normal test with plug-in s and D from each sample
      a <- c(0, 1, -1, 0); b <- c(2, 1, 1, 0)
      Q <- stats::rmultinom(REPS, n_w, s$p) / n_w
      rej <- c(t1 = 0, pw = 0)
      for (j in seq_len(REPS)) {
        q <- Q[, j]; Sig <- diag(q) - tcrossprod(q)
        sq <- sqrt(max(drop(t(a) %*% Sig %*% a), 1e-12) / n_w); Dq <- sum(b * q); A <- abs(sum(a * q))
        for (k in 1:2) {
          mu0 <- Dq * (1 - c(th1, th0)[k])
          lo <- vs_folded_quantile(ALPHA / 2, mu0, sq); hi <- vs_folded_quantile(1 - ALPHA / 2, mu0, sq)
          rej[k] <- rej[k] + ((A < lo) || (A > hi))
        }
      }
      row$type1_wald <- rej[["t1"]] / REPS; row$power_wald <- rej[["pw"]] / REPS
    } else {
      row$type1_wald  <- mean(abs(est + mm - th1) > crit * se)    # H0 at the truth (est + mm: Miller-Madow where it applies)
      row$power_wald  <- mean(abs(est + mm - th0) > crit * se)    # H0 at theta_0, n = n_W
    }
    row$undefined_pct <- 100 * mean(!ok)
    # score test for confusion-matrix metrics: null variance from the null operating point
    if (fam == "confusion" && !folded) {
      r <- tryCatch(suppressWarnings(do.call(n_for_power_metric,
             c(list(p = p4, name = nm, delta = delta, alpha = ALPHA, power = POWER,
                    test = "score", null_path = "auto"), e$args))), error = function(err) NULL)
      if (!is.null(r) && !is.na(r$n_score)) {
        es_s <- s$est_se(r$n_score, REPS); est_s <- es_s[is.finite(es_s[, 1]), 1]
        row$V_null <- r$V_null; row$n_score <- r$n_score
        row$power_score <- mean(abs(est_s - th0) > Z_A * sqrt(r$V_null / r$n_score))
        # type I error of the score test: H0 at the truth, so the null variance is V at the truth (V1)
        row$type1_score <- mean(abs(est_s - th1) > Z_A * sqrt(V1 / r$n_score))
      } else row$note <- "score: null value unreachable"
    }
    rows[[nm]] <- row
    cat(sprintf("%-26s%-12s n_W=%6d  dV=%+6.1f%%%s  typeI=%.3f  power=%.3f%s\n", nm, fam, n_w,
                row$dV_pct,
                if (folded) sprintf(" (vs delta V; vs folded V %+.1f%%)", row$dV_folded_pct) else "",
                row$type1_wald, row$power_wald,
                if (!is.null(row$power_score)) sprintf("  | n_S=%6d typeI_S=%.3f power_S=%.3f", row$n_score, row$type1_score, row$power_score) else ""))
    flush.console()
  }
}

# -----------------------------------------------------------------------------
# Table, summary, Results sentence
# -----------------------------------------------------------------------------

cols <- c("metric", "family", "test", "theta", "V", "dV_pct", "dV_folded_pct", "n_wald", "type1_wald", "power_wald",
          "V_null", "n_score", "type1_score", "power_score", "undefined_pct", "skew", "small_n", "note")
tab <- do.call(rbind, lapply(rows, function(r) {
  as.data.frame(lapply(setNames(cols, cols), function(k) if (is.null(r[[k]])) NA else r[[k]]),
                stringsAsFactors = FALSE) }))
if (!is.null(ONLY) && file.exists("operating_characteristics.csv")) {
  # merge the rerun rows into the existing table (same seed, so the untouched rows are unchanged)
  old <- utils::read.csv("operating_characteristics.csv", stringsAsFactors = FALSE)
  for (k in setdiff(cols, names(old))) old[[k]] <- NA
  old <- old[, cols]; old$note[is.na(old$note)] <- ""
  hit <- tab$metric %in% old$metric
  old[match(tab$metric[hit], old$metric), ] <- tab[hit, ]
  tab <- rbind(old, tab[!hit, ])
}
utils::write.csv(tab, "operating_characteristics.csv", row.names = FALSE)

done <- tab[!is.na(tab$power_wald), ]
dv <- done$dV_pct; t1 <- done$type1_wald; pw <- done$power_wald
off_pow <- done[abs(done$power_wald - POWER) > 0.02, ]
off_t1  <- done[abs(done$type1_wald - ALPHA) > 0.01, ]
cat("\n", strrep("=", 118), "\n", sep = "")
cat(sprintf("OPERATING CHARACTERISTICS   %d metrics   REPS = %s   script %s   delta = %s   alpha = %s   target power = %s\n",
            nrow(done), format(REPS, big.mark = ","), SCRIPT_VERSION,
            if (!is.na(DELTA_ABS)) sprintf("%s (absolute)", DELTA_ABS) else sprintf("%.0f%% of the metric value", 100 * DELTA_FRAC),
            ALPHA, POWER))
cat(sprintf("Monte Carlo error (2 SE): variance ratio +/- %.1f%%   type I error +/- %.3f   power +/- %.3f\n",
            200 * SE_V, 2 * SE_RATE, 2 * SE_POW))
cat(strrep("=", 118), "\n", sep = "")
print(tab[, c("metric", "family", "test", "theta", "V", "dV_pct", "n_wald", "type1_wald", "power_wald", "n_score", "type1_score", "power_score", "skew")],
      digits = 3, row.names = FALSE)
cat(strrep("=", 118), "\nSUMMARY\n", sep = "")
nf <- is.na(done$dV_folded_pct)
cat(sprintf("  simulated vs analytic (delta-method) variance, %d metrics: %+.1f%% to %+.1f%%; %d of %d within the MC noise band\n",
            sum(nf), min(dv[nf]), max(dv[nf]), sum(abs(dv[nf]) <= 200 * SE_V), sum(nf)))
for (i in which(!nf))
  cat(sprintf("  %s: %+.1f%% vs the delta-method V (expected: the distribution is folded at the kink), %+.1f%% vs the folded-normal V; sized and tested with the folded-normal test\n",
              done$metric[i], done$dV_pct[i], done$dV_folded_pct[i]))
cat(sprintf("  type I error (nominal %s): %.3f to %.3f; %d outside %s +/- 0.01: %s\n", ALPHA, min(t1), max(t1),
            nrow(off_t1), ALPHA, paste(sprintf("%s (%.3f)", off_t1$metric, off_t1$type1_wald), collapse = ", ")))
cat(sprintf("  power at n_Wald (target %s): %.3f to %.3f; %d outside +/- 0.02: %s\n", POWER, min(pw), max(pw),
            nrow(off_pow), paste(sprintf("%s (%.3f)", off_pow$metric, off_pow$power_wald), collapse = ", ")))
small <- done[done$n_wald < SMALL_N_OC, ]
if (nrow(small)) cat(sprintf(paste0("  n below %d (%d; the large-sample approximation has not converged, type I error and power are ",
                                    "reported as found and flagged in the package): %s\n"), SMALL_N_OC, nrow(small),
                             paste(sprintf("%s (n = %d, type I %.3f, power %.3f)", small$metric, small$n_wald, small$type1_wald, small$power_wald), collapse = ", ")))
mmr <- done[!is.na(done$test) & done$test == "z+MM", ]
if (nrow(mmr)) cat(sprintf("  Miller-Madow corrected estimate used in the test (plug-in entropies are biased toward the null by (k-1)/(2n)): %s\n",
                           paste(sprintf("%s (power %.3f)", mmr$metric, mmr$power_wald), collapse = ", ")))
sc <- done[!is.na(done$power_score), ]
if (nrow(sc)) cat(sprintf("  score test (confusion-matrix metrics, n = %d): type I error %.3f to %.3f, power %.3f to %.3f\n",
                          nrow(sc), min(sc$type1_score, na.rm = TRUE), max(sc$type1_score, na.rm = TRUE), min(sc$power_score), max(sc$power_score)))
cat("\nResults sentence (fill into the manuscript):\n")
cat(sprintf(paste0("  For all %d implemented variances the Monte Carlo variance was within %.0f%% of the analytic value ",
                   "(range %+.1f%% to %+.1f%%). At nominal alpha = .05 the empirical type I error of the Wald test ranged from ",
                   "%.3f to %.3f, and empirical power at the calculated sample size ranged from %.2f to %.2f against a target of %.2f; ",
                   "the %d metrics outside +/- 0.02 of the target were %s.\n"),
            nrow(done), max(abs(dv)), min(dv), max(dv), min(t1), max(t1), min(pw), max(pw), POWER,
            nrow(off_pow), if (nrow(off_pow)) paste(off_pow$metric, collapse = ", ") else "none"))
cat(sprintf("(%.0f s)\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
