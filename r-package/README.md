# metricpower

Sample size and power for the performance metrics used to evaluate
medical-imaging AI.

## Install

```r
# from GitHub (this folder is a sub-directory of the metricpower repository)
remotes::install_github("wth12380-alt/metricpower", subdir = "r-package")
# or from the bundled tarball
install.packages("metricpower_0.1.1.tar.gz", repos = NULL, type = "source")
library(metricpower)
```

The interactive Shiny app (metric → n, with the formula and reference shown) is part of
the Python package at the repository root; see the top-level README.

## The one idea

The package grades every metric by what already exists:

| Grade | Meaning | What the package does |
|---|---|---|
| **L4** | An established package already computes sample size | Points you to it, with the call template and reference |
| **L3** | The variance is published or elementary, but nothing inverts it to n | Implements it, across five estimator families |
| L1/L2 | No analytic variance exists | Refuses, with the reason |

## Use

### An L3 metric: one call, name the metric

```r
p <- cell_prob_from_operating_point(sensitivity = 0.70,
                                    specificity = 0.80,
                                    prevalence  = 0.50)

n_for_power_l3("dsc_f1",     p = p,  delta = 0.05)$n      # 810
n_for_power_l3("mcc",        p = p,  delta = 0.05)$n      # 2323
n_for_power_l3("markedness", p = p,  delta = 0.05)$n      # 2339
```

Swapping the metric is the whole change. Every confusion-matrix metric is a
plug-in to one delta-method engine, so there is no per-metric derivation.

### The arguments differ by family, because the information differs

```r
# multiclass: a k x k joint table
P <- matrix(c(.30,.04,.02, .05,.22,.05, .03,.06,.23), 3, 3, byrow = TRUE)
n_for_power_l3("macro_f1", P = P, delta = 0.05)$n        # 594

# count rate: false positives per image, with overdispersion
n_for_power_l3("fppi", lambda = 0.5, ratio = 0.7)$n              # 124  (Poisson)
n_for_power_l3("fppi", lambda = 0.5, ratio = 0.7, theta = 2)$n   # 155  (neg. binomial)

# per-case mean: pilot scores, one per image
pilot_dice <- rbeta(60, 9, 2)
n_for_power_l3("dice_percase", x = pilot_dice, delta = 0.03)$n
```

### An L4 metric: ask what to run

```r
l4_show("auroc")
#  auroc is graded L4: 'pROC' already solves this.
#    pROC::power.roc.test(auc = <auc>, sig.level = 0.05, power = 0.8, kappa = <ratio>)
#    reference: DeLong; Obuchowski & McClish (1997); Hanley & McNeil (1982)

l4_requirements(c("auroc", "icc", "kappa_unweighted"))   # which packages you need
```

### Two design decisions you will need to make

```r
# paired comparison of two models on the same test set: needs the 8-cell joint
p8 <- c(0.30, 0.05, 0.03, 0.12, 0.04, 0.06, 0.02, 0.38)
delta_var_paired(p8, "dsc_f1")$V          # 0.202, against 0.529 for two independent sets

# clustering: 3 lesions per patient, ICC 0.30
n_for_power_l3("sensitivity", p = p, delta = 0.05, deff = design_effect(3, 0.30))$n
```

### Is the delta method trustworthy at your n?

```r
check_delta_reliable(p, "dsc_f1", n = 30)
#  Caution. At n = 30 the analytic variance understates the truth by about 10% ...
check_delta_reliable(p, "dsc_f1", n = 810)
#  OK. The delta method is reliable at n = 810.
```

## What to know before trusting a number

- **`delta` has no default.** Three conventions are in use (absolute, 2% of
  the mean, 0.1 SD) and none is neutral. With a standardised delta every metric
  needs the same n, because V cancels out. Choose on clinical grounds.
- **The delta method needs n of roughly 250.** Below that it understates the
  variance by up to 10% for F1. Above it, no inflation is warranted.
- **Per-case metrics fail through skew, not sample size.** Their variance is
  exact; what degrades is the normal approximation. Per-case Dice is left-skewed,
  so its intervals sit above the truth and overstate performance.
- **Some metrics are refused on purpose.** ECE, KCE, entropy at the uniform
  distribution, R-squared at zero, Mahalanobis distance: the null sits at a
  stationary point, the gradient vanishes, and the sample-size formula is invalid.

## Validation

Every implemented variance is checked against simulation:

```r
source(system.file("validation", "validate_all.R", package = "metricpower"))
```

85 metrics, 10,000 replicates each, all within Monte Carlo noise.

## Files

```
R/01_core.R        engine, confusion-matrix metrics, sample size, paired, clustering
R/02_l3.R          multiclass, count rate, per-case, entropy, CV
R/03_l3_all.R      all 47 L3 metrics, unified dispatcher n_for_power_l3()
R/04_l4_wrap.R     23 L4 pointers to established packages
R/zzz.R            registers the L3 metrics at load
inst/validation/   the simulation check
```
