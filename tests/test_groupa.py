import math
import pytest

import metricpower as mp
from metricpower import groupa as A

z = 1.959964; zb = 0.841621


def test_pearson_classical_value():
    # r = 0.30 vs 0, 80% power, two-sided 0.05: textbook n = 85 (pwr::pwr.r.test gives 84.07)
    assert mp.n_for_power_group_a("pearson_correlation", r1=0.30, r0=0.0).n == 85


def test_proportion_formulas():
    r = mp.n_for_power_group_a("accuracy", p1=0.80, halfwidth=0.05)
    assert r.n == math.ceil(z ** 2 * 0.16 / 0.05 ** 2)
    r = mp.n_for_power_group_a("accuracy", p1=0.80, p0=0.75, method="score")
    assert r.n == math.ceil((z * math.sqrt(.75 * .25) + zb * math.sqrt(.8 * .2)) ** 2 / .05 ** 2)
    assert mp.n_for_power_group_a("accuracy", p1=0.80, halfwidth=0.05, method="wilson").n in (245, 246)


def test_buderer_total_is_subgroup_over_prevalence():
    r = mp.n_for_power_group_a("sensitivity", value1=0.80, halfwidth=0.05, prevalence=0.30)
    assert r.n == math.ceil(r.extra["n_subgroup"] / 0.30)
    r2 = mp.n_for_power_group_a("specificity", value1=0.80, halfwidth=0.05, prevalence=0.30)
    assert r2.n == math.ceil(r2.extra["n_subgroup"] / 0.70)


def test_likelihood_ratio_and_odds_ratio():
    r = mp.n_for_power_group_a("lr_positive", sens=.8, spec=.9, prevalence=.3, lr0=4)
    assert r.estimate == pytest.approx(8.0)
    assert r.V_alt == pytest.approx(0.2 / (0.8 * 0.3) + 0.9 / (0.1 * 0.7))
    r = mp.n_for_power_group_a("odds_ratio", p1=.6, p2=.4, or0=1)
    assert r.estimate == pytest.approx(2.25) and r.V_alt == pytest.approx(2 / (.6 * .4) + 2 / (.4 * .6))


def test_auc_hanley_behaviour():
    assert A._hanley_var(0.5, 1, 1) == pytest.approx(0.25)     # at n1 = n0 = 1 the variance is A(1-A)
    r = mp.n_for_power_group_a("auroc", auc1=0.80, auc0=0.70)
    assert r.extra["n1"] == r.extra["n0"] and 150 < r.n < 250
    assert mp.n_for_power_group_a("auroc", auc1=0.90, auc0=0.70).n < r.n
    assert mp.n_for_power_group_a("auroc", auc1=0.75, auc0=0.5).n < 60
    r2 = mp.n_for_power_group_a("auroc", auc1=0.80, auc0=0.70, method="obuchowski")
    assert 0.5 < r2.n / r.n < 2


def test_kappa_icc_ccc():
    r = mp.n_for_power_group_a("kappa_unweighted", kappa1=.6, kappa0=.4, marginals=[.3, .7])
    assert r.estimate == pytest.approx(0.6) and r.null_value == pytest.approx(0.4) and r.n > 50
    assert mp.n_for_power_group_a("icc", icc1=.8, icc0=.6, k=2).n == 50
    assert mp.n_for_power_group_a("icc", icc1=.8, halfwidth=.1, k=2).n == 51
    assert mp.n_for_power_group_a("ccc", ccc1=.9, ccc0=.8, r=.92, u=.1).n > 10


def test_riley_criteria():
    assert mp.n_for_power_group_a("oe_ratio", prevalence=.2, ci_width=.2).n == 1542
    assert mp.n_for_power_group_a("c_statistic", c=.8, prevalence=.2, halfwidth=.05).n == 471
    cs = mp.n_for_power_group_a("calibration_slope", lp_mean=-1.5, lp_sd=1.2, halfwidth=0.1)
    assert cs.n > 1000
    snb = mp.n_for_power_group_a("net_benefit", sens=.8, spec=.7, prevalence=.2, threshold=.1, delta=.05)
    assert snb.estimate == pytest.approx(0.8 - (0.1 / 0.9) * 4 * 0.3)


def test_hazard_ratio_schoenfeld():
    r = mp.n_for_power_group_a("hazard_ratio", hr=.7, p_event_control=.5)
    d = (z + zb) ** 2 / (0.25 * math.log(.7) ** 2)
    assert r.steps[0][1] == math.ceil(d)


def test_pointers_only():
    assert mp.n_for_power_group_a("froc").n is None
    assert mp.n_for_power_group_a("partial_auc").n is None
    assert mp.explain_group_c("hausdorff_distance").n is None
