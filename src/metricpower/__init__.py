"""
metricpower: a priori sample size and power for the performance metrics used
to evaluate medical-imaging AI.

Group A  established formulas already implemented in R packages, ported natively
Group B  published or elementary variances, inverted to a sample size here
Group C  no analytic variance (refused, with the reason)

Quick start
-----------
>>> import metricpower as mp
>>> p = mp.cell_prob_from_operating_point(0.70, 0.80, 0.50)
>>> mp.n_for_power_group_b("dsc_f1", p=p, delta=0.05).n
810
>>> mp.n_for_power_group_a("auroc", auc1=0.80, auc0=0.70).n
"""
from .core import (n_for_power_t, power_for_n_t, kink_distance, vs_variance_folded, vs_folded_power, vs_folded_quantile, n_for_power_vs_folded, Result, cell_prob, cell_prob_from_operating_point, delta_var, numeric_gradient,
                   METRIC_REGISTRY, CM_METRICS, get_metric, get_gradient, metric_variance,
                   paired_marginal, delta_var_paired, design_effect, n_for_power, power_for_n,
                   n_for_precision, metric_ci, cell_prob_at_metric, n_for_power_metric,
                   check_cell_counts, check_delta_reliable, sensitivity_by_n, sensitivity_by_n_percase,
                   ratio_noise_band, kappa_w, kappa_w_var_fce, kappa_w_var_delta,
                   weights_identity, weights_linear, weights_quadratic,
                   var_percase, var_percase_paired, brier_scores, var_sqrt_transform,
                   n_for_power_calibration_error)
from .families import (metric_micro, metric_macro, metric_variance_multiclass, var_rate, n_for_power_rate,
                       PERCASE_REGISTRY, var_percase_transformed, n_for_power_percase,
                       metric_entropy, grad_entropy, entropy_variance, metric_conditional_entropy,
                       conditional_entropy_variance, cv_variance, ustat_variance, kendall_tau_variance,
                       cluster_index_variance, r_squared_variance, repeatability_variance,
                       discrimination_slope_variance, risk_difference_variance, paired_proportion_variance,
                       mahalanobis_sample_size, n_for_target_rse, timedep_auc_sample_size)
from .groupb import GROUP_B_FAMILY, GROUP_B_INPUTS, family_of, group_b_manifest, n_for_power_group_b
from .groupa import GROUP_A_REGISTRY, n_for_power_group_a
from . import groupa, groupb, families, core
from .references import REFERENCES
from .groupc import GROUP_C, explain_group_c
# experimental, not wired into the dispatcher or the app: from metricpower import bleu

__version__ = "0.1.1"
