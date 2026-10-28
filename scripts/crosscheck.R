# Print the R column of the cross-check table. Needs the R package metricpower >= 0.1.1.
library(metricpower)
p  <- cell_prob_from_operating_point(0.70, 0.80, 0.50)
P3 <- matrix(c(.30,.04,.02, .05,.22,.05, .03,.06,.23), 3, 3, byrow = TRUE)
for (m in c("accuracy","balanced_accuracy","dsc_f1","specificity","iou","sensitivity","mcc")) {
  r <- n_for_power_l3(m, p = p, delta = 0.05)
  cat(sprintf("%-42s%24s\n", m, paste(r$n_wald, "/", r$n_score)))
}
cat(sprintf("%-42s%24s\n", "macro_f1", n_for_power_l3("macro_f1", P = P3, delta = 0.05)$n))
cat(sprintf("%-42s%24s\n", "weighted_kappa", n_for_power_l3("weighted_kappa", P = P3, delta = 0.05)$n))
cat(sprintf("%-42s%24s\n", "fppi Poisson / NB(2)", paste(n_for_power_l3("fppi", lambda = 0.5, ratio = 0.7)$n, "/",
                                                        n_for_power_l3("fppi", lambda = 0.5, ratio = 0.7, theta = 2)$n)))
cat(sprintf("%-42s%24s\n", "risk_difference", n_for_power_l3("risk_difference", p1 = .6, p2 = .45, delta = .15)$n))
cat(sprintf("%-42s%24s\n", "paired dsc_f1 V", round(delta_var_paired(c(0.30,0.05,0.03,0.12,0.04,0.06,0.02,0.38), "dsc_f1")$V, 4)))
cat(sprintf("%-42s%24s\n", "dsc_f1 power .9 / alpha .01 / one-sided",
            paste(n_for_power_l3("dsc_f1", p = p, delta = .05, power = .9)$n, "/",
                  n_for_power_l3("dsc_f1", p = p, delta = .05, alpha = .01)$n, "/",
                  n_for_power_l3("dsc_f1", p = p, delta = .05, two_sided = FALSE)$n)))
cat(sprintf("%-42s%24s\n", "dsc_f1 deff 1.6", n_for_power_l3("dsc_f1", p = p, delta = .05, deff = design_effect(3, .3))$n))
