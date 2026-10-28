"""
Group C: metrics with no analytic sampling variance. No a priori sample size
exists for them; uncertainty must be obtained after data collection (bootstrap),
and a priori sizing would require simulation under an assumed data-generating
model. Each entry records why.
"""
from .core import Result

GROUP_C = {
    "hausdorff_distance":      "a maximum over boundary points; the limit law is Gumbel (extreme value), not normal",
    "hd95":                    "an upper quantile of a per-case distance; as a per-case MEAN over cases it is Group B (hd95_percase)",
    "assd_masd":               "per-case and heavy-tailed; no published variance (as a per-case mean see assd_percase)",
    "nsd":                     "per-case with a tolerance hyperparameter; no published variance (as a per-case mean see nsd_percase)",
    "boundary_iou":            "per-case with a tolerance hyperparameter; no published variance",
    "cldice":                  "needs more than the confusion matrix; excluded from the Metrics Reloaded pool",
    "box_iou":                 "per-case mean IoU of matched boxes; no published variance",
    "mean_iou":                "mean over classes on the same images, so the per-class terms are correlated",
    "global_consistency_error": "minimum of two asymmetric terms; not differentiable where they are equal",
    "ece":                     "degenerate under the null of perfect calibration; convergence rate differs between calibrated and miscalibrated models",
    "cwce":                    "same non-regular asymptotics as ECE",
    "kce":                     "the unbiased U-statistic can be negative; degenerate under the null",
    "ece_kde":                 "a potentially biased estimator of the canonical calibration error",
    "mce":                     "a maximum over bins: extreme-value rather than normal",
    "metric_at_target":        "a quantile-based cut-off read off a curve; the estimated threshold adds a variance term that needs the ROC slope",
    "mutual_information":      "gradient vanishes under independence; the limit is chi-square",
    "kl_divergence":           "degenerate when the two distributions coincide",
    "variation_of_information": "built from mutual information, inherits the degeneracy",
    "average_precision":       "a step function of a ranked list; terms are dependent and no closed form is accepted",
    "panoptic_quality":        "SQ and RQ are dependent (SQ computed only over matched segments); bootstrap only",
    "bleu":                    "not the mean of sentence-level scores (Koehn 2004); bootstrap in the literature. An experimental delta-method variance exists in metricpower.bleu but is not wired in",
    "wasserstein":             "optimal transport; convergence rate depends on dimension and is not root-n",
    "mauve":                   "divergence curves over embeddings; no sampling theory",
    "radcliq":                 "composite of other metrics; no sampling theory",
    "radgraph_f1":             "depends on an extraction model; the estimand moves with that model",
    "llm_scores":              "RaTEScore / GREEN / FineRadScore: the estimand is not stable across model versions",
}


def explain_group_c(metric: str) -> Result:
    if metric not in GROUP_C:
        raise KeyError(f"unknown Group C metric '{metric}'")
    return Result(n=None, metric=metric, group="C",
                  notes=["No analytic sampling variance exists: " + GROUP_C[metric],
                         "Uncertainty can only be quantified after data collection, by bootstrap; an a priori "
                         "sample size would require simulation under an assumed data-generating model."],
                  reference=["maier_hein2024"])
