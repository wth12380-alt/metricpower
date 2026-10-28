# =============================================================================
#  variance_l4_wrap.R
#
#  L4 layer: metrics for which a package already computes sample size or power.
#
#  Nothing here re-derives anything. The contribution is a single entry point
#  with the correct citation attached, so that a user planning a study does not
#  have to know that sensitivity lives in presize, AUROC in pROC, kappa in
#  kappaSize, calibration in pmvalsampsize and FROC in RJafroc.
#
#  Each entry records the package, the function, a call template, and the
#  reference. If the package is installed the call is dispatched; if not, the
#  template is returned so the user can see exactly what to run. Refusing to
#  guess is deliberate: silently substituting our own approximation for an
#  established routine would be worse than saying which routine to use.
# =============================================================================

L4_REGISTRY <- list(

  accuracy = list(
    pkg = "presize", fun = "prec_prop",
    call = 'presize::prec_prop(p = <accuracy>, conf.width = <width>)',
    ref  = "Wilson / Agresti-Coull interval inversion",
    note = "a plain binomial proportion"),

  sensitivity = list(
    pkg = "MKpower", fun = "power.diagnostic.test",
    call = 'MKpower::power.diagnostic.test(sens = <sens>, delta = <delta>, power = 0.8)',
    ref  = "Buderer (1996); prevalence-adjusted",
    note = "presize::prec_sens for the precision-based version"),

  specificity = list(
    pkg = "MKpower", fun = "power.diagnostic.test",
    call = 'MKpower::power.diagnostic.test(spec = <spec>, delta = <delta>, power = 0.8)',
    ref  = "Buderer (1996)",
    note = "presize::prec_spec for the precision-based version"),

  ppv = list(
    pkg = "presize", fun = "prec_prop",
    call = 'presize::prec_prop(p = <ppv>, conf.width = <width>)',
    ref  = "binomial, conditional on the predicted-positive count",
    note = "the denominator is random, which the binomial form ignores"),

  npv = list(
    pkg = "presize", fun = "prec_prop",
    call = 'presize::prec_prop(p = <npv>, conf.width = <width>)',
    ref  = "binomial, conditional on the predicted-negative count", note = ""),

  lr_positive = list(
    pkg = "presize", fun = "prec_pos_lr",
    call = 'presize::prec_pos_lr(prev = <prev>, sens = <sens>, spec = <spec>, conf.width = <width>)',
    ref  = "Simel, Samsa & Matchar (1991)", note = "log-scale delta method"),

  lr_negative = list(
    pkg = "presize", fun = "prec_neg_lr",
    call = 'presize::prec_neg_lr(prev = <prev>, sens = <sens>, spec = <spec>, conf.width = <width>)',
    ref  = "Simel, Samsa & Matchar (1991)", note = ""),

  odds_ratio = list(
    pkg = "presize", fun = "prec_or",
    call = 'presize::prec_or(p1 = <p1>, p2 = <p2>, conf.width = <width>)',
    ref  = "log-odds delta method", note = "epiR offers alternative methods"),

  risk_difference = list(
    pkg = "presize", fun = "prec_riskdiff",
    call = 'presize::prec_riskdiff(p1 = <p1>, p2 = <p2>, conf.width = <width>)',
    ref  = "Newcombe / Wald",
    note = "this is the route for demographic parity, equalized odds and equality of opportunity, which are all differences of proportions between groups"),

  auroc = list(
    pkg = "pROC", fun = "power.roc.test",
    call = 'pROC::power.roc.test(auc = <auc>, sig.level = 0.05, power = 0.8, kappa = <ratio>)',
    ref  = "DeLong; Obuchowski & McClish (1997); Hanley & McNeil (1982)",
    note = "solves for sample size, power, significance level or minimum detectable AUC; one or two curves; presize::prec_auc for the precision-based version"),

  partial_auc = list(
    pkg = "pROC", fun = "power.roc.test",
    call = 'pROC::power.roc.test(<roc object with partial.auc set>)',
    ref  = "Obuchowski variance for the partial area",
    note = "build the roc object with partial.auc and partial.auc.focus first"),

  kappa_unweighted = list(
    pkg = "kappaSize", fun = "PowerBinary / Power3Cats / Power4Cats / Power5Cats",
    call = 'kappaSize::PowerBinary(kappa0 = <null>, kappa1 = <alt>, props = <p>, raters = 2, alpha = 0.05, power = 0.8)',
    ref  = "Rotondi & Donner (2012)",
    note = "12 functions covering power and CI approaches, binary through five categories, two through six raters. Does NOT cover weighted kappa - for that see kappa_w_var_fce() in variance_core.R, which is L3"),

  icc = list(
    pkg = "ICC.Sample.Size", fun = "calculateIccSampleSize",
    call = 'ICC.Sample.Size::calculateIccSampleSize(p = <icc>, p0 = <null>, k = <raters>, alpha = 0.05, tails = 2, power = 0.8)',
    ref  = "Zou (2012)",
    note = "presize::prec_icc implements the Bonett (2002) precision-based version"),

  ccc = list(
    pkg = "cccPower", fun = "cccPower",
    call = 'cccPower(<null ccc>, <alternative ccc>, power = 0.8, alpha = 0.05)',
    ref  = "asymptotic distribution of the concordance correlation coefficient",
    note = "needs the expected means and variances of both methods"),

  pearson_correlation = list(
    pkg = "pwr", fun = "pwr.r.test",
    call = 'pwr::pwr.r.test(r = <r>, sig.level = 0.05, power = 0.8)',
    ref  = "Fisher z transform",
    note = "presize::prec_cor for the precision-based version"),

  calibration_slope = list(
    pkg = "pmvalsampsize", fun = "pmvalsampsize",
    call = 'pmvalsampsize::pmvalsampsize(type = "b", prevalence = <prev>, cslope = <slope>, csciwidth = <width>)',
    ref  = "Riley et al. (2021); Archer et al. (2020)",
    note = "one call also covers the O:E ratio, the c-statistic and net benefit at a threshold"),

  oe_ratio = list(
    pkg = "pmvalsampsize", fun = "pmvalsampsize",
    call = 'pmvalsampsize::pmvalsampsize(type = "b", prevalence = <prev>, oeciwidth = <width>)',
    ref  = "Riley et al. (2021)", note = ""),

  c_statistic = list(
    pkg = "pmvalsampsize", fun = "pmvalsampsize",
    call = 'pmvalsampsize::pmvalsampsize(type = "b", prevalence = <prev>, cstatistic = <c>, cstatciwidth = <width>)',
    ref  = "Riley et al. (2021)",
    note = "for a survival c-index with censoring, survcomp::concordance.index gives the interval and cindex.comp compares two, but neither sizes a study"),

  net_benefit = list(
    pkg = "pmvalsampsize", fun = "pmvalsampsize",
    call = 'pmvalsampsize::pmvalsampsize(type = "b", prevalence = <prev>, nbciwidth = <width>, nbthreshold = <pt>)',
    ref  = "Riley et al. (2021), standardised net benefit",
    note = "dcurves and rmda give bootstrap intervals but do not size a study"),

  froc = list(
    pkg = "RJafroc", fun = "SsSampleSizeKGivenJ / SsPowerGivenJK",
    call = 'RJafroc::SsSampleSizeKGivenJ(dataset = <pilot>, J = <readers>, effectSize = <delta>)',
    ref  = "Chakraborty; Obuchowski-Rockette and Dorfman-Berbaum-Metz",
    note = "needs pilot variance components. MRMCsamplesize plans without pilot data. Both assume a multi-reader design; for a standalone algorithm with no readers there is nothing"),

  afroc = list(
    pkg = "RJafroc", fun = "SsSampleSizeKGivenJ",
    call = 'RJafroc::SsSampleSizeKGivenJ(dataset = <pilot>, FOM = "wAFROC", J = <readers>, effectSize = <delta>)',
    ref  = "Chakraborty", note = ""),

  hazard_ratio = list(
    pkg = "powerSurvEpi", fun = "ssizeCT.default",
    call = 'powerSurvEpi::ssizeCT.default(power = 0.8, k = 1, pE = <pE>, pC = <pC>, RR = <hr>, alpha = 0.05)',
    ref  = "Schoenfeld",
    note = "lrstat::lrsamplesize handles non-proportional hazards and group-sequential designs"),

  survival_probability = list(
    pkg = "powerSurvEpi", fun = "ssizeCT.default",
    call = 'powerSurvEpi::ssizeCT.default(...)',
    ref  = "Greenwood variance for the survival curve itself", note = "")
)

#' Show what to run for an L4 metric, without running it
#'
#' @param name key in L4_REGISTRY
#' @param quiet suppress the message and return the entry invisibly
l4_show <- function(name, quiet = FALSE) {
  e <- L4_REGISTRY[[name]]
  if (is.null(e)) {
    stop("no L4 entry for '", name, "'. Available: ",
         paste(names(L4_REGISTRY), collapse = ", "), call. = FALSE)
  }
  have <- requireNamespace(e$pkg, quietly = TRUE)
  if (!quiet) {
    message(sprintf(
      "%s is graded L4: '%s' already solves this.%s\n  %s\n  reference: %s%s",
      name, e$pkg,
      if (have) "" else sprintf("\n  install.packages(\"%s\")", e$pkg),
      e$call, e$ref,
      if (nzchar(e$note)) paste0("\n  note: ", e$note) else ""))
  }
  invisible(structure(c(e, list(installed = have)),
                      class = "l4_pointer", metric = name))
}

#' Sample size for a metric where an established package already does the job
#'
#' With no arguments beyond the name this is a query: it returns the call
#' template and reference. With arguments it dispatches to the package.
#'
#' Two things this deliberately does not do. It never substitutes an
#' approximation of our own when the package is missing, because silently
#' replacing an established routine is worse than naming it. And it never
#' forwards an empty argument list to the underlying function: several of these
#' are S3 generics that dispatch on their first argument, so calling them with
#' nothing produces an obscure "no applicable method" error rather than a
#' useful one.
#'
#' @param name key in L4_REGISTRY
#' @param ... arguments forwarded to the underlying function
n_for_power_external <- function(name, ...) {
  e <- L4_REGISTRY[[name]]
  if (is.null(e)) {
    stop("no L4 entry for '", name, "'. Available: ",
         paste(names(L4_REGISTRY), collapse = ", "), call. = FALSE)
  }
  args <- list(...)
  if (!length(args)) return(l4_show(name))
  if (!requireNamespace(e$pkg, quietly = TRUE)) {
    message(sprintf("'%s' is not installed. install.packages(\"%s\")", e$pkg, e$pkg))
    return(l4_show(name, quiet = TRUE))
  }
  fn <- strsplit(e$fun, " / ")[[1]][1]
  f  <- getExportedValue(e$pkg, fn)
  tryCatch(do.call(f, args),
           error = function(err) stop(sprintf(
             "%s::%s() rejected those arguments.\n  underlying error: %s\n  expected form: %s",
             e$pkg, fn, conditionMessage(err), e$call), call. = FALSE))
}

#' Which packages does a planned study need?
l4_requirements <- function(metrics) {
  known <- intersect(metrics, names(L4_REGISTRY))
  if (!length(known)) return(character(0))
  pk <- vapply(L4_REGISTRY[known], function(e) e$pkg, character(1))
  data.frame(metric = known, package = unname(pk),
             installed = vapply(unname(pk), function(p)
               requireNamespace(p, quietly = TRUE), logical(1)),
             row.names = NULL)
}

#' Print the full L4 map
l4_map <- function() {
  for (k in names(L4_REGISTRY)) {
    e <- L4_REGISTRY[[k]]
    cat(sprintf("%-22s %-16s %s\n", k, e$pkg, e$ref))
  }
  invisible(names(L4_REGISTRY))
}
