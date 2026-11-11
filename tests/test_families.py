import math
import numpy as np
import pytest

import metricpower as mp
from metricpower import families as F

P3 = np.array([[.30, .04, .02], [.05, .22, .05], [.03, .06, .23]])


def test_multiclass_R():
    r = mp.n_for_power_group_b("macro_f1", P=P3, delta=0.05)
    assert r.n == 594 and r.V_alt == pytest.approx(0.1890, abs=1e-3)   # (R)
    assert mp.metric_variance_multiclass(P3, "f1", "micro")["V"] == pytest.approx(0.1875, abs=1e-4)


def test_macro_less_stable_under_imbalance():
    Pim = np.array([[.80, .02, .01], [.03, .08, .01], [.01, .01, .03]])
    mi = mp.metric_variance_multiclass(Pim, "f1", "micro")["V"]
    ma = mp.metric_variance_multiclass(Pim, "f1", "macro")["V"]
    assert ma > 2 * mi


def test_weighted_kappa_fce_equals_engine_R():
    W = mp.weights_linear(3)
    assert mp.kappa_w_var_fce(P3, W) == pytest.approx(mp.kappa_w_var_delta(P3, W), rel=1e-5)
    assert mp.n_for_power_group_b("weighted_kappa", P=P3, delta=0.05).n == 1245   # (R)


def test_rate_R():
    assert mp.n_for_power_group_b("fppi", lam=0.5, ratio=0.7).n == 124            # (R)
    assert mp.n_for_power_group_b("fppi", lam=0.5, ratio=0.7, theta=2).n == 155   # (R)


def test_percase_transforms():
    x = np.random.default_rng(1).gamma(2, 0.1, 500)
    r = F.var_percase_transformed(x, "sqrt")
    assert r["V"] == pytest.approx(r["V_mean"] / (4 * x.mean()))
    r = F.var_percase_transformed(x, "exp")
    assert r["V"] == pytest.approx(r["V_mean"] * math.exp(2 * x.mean()))
    assert F.var_percase_transformed(mean=0.2, sd=0.1)["V"] == pytest.approx(0.01)


def test_percase_variance_is_exact_at_any_n():
    rng = np.random.default_rng(2)
    x = rng.beta(9, 2, 10000)
    V = mp.var_percase(x)
    for n in (25, 100):
        m = rng.choice(x, size=(20000, n)).mean(1)
        assert abs(n * m.var(ddof=1) / V - 1) < 0.05


def test_entropy_degenerate_at_uniform():
    with pytest.warns(UserWarning):
        r = F.entropy_variance([.25, .25, .25, .25])
    assert r["degenerate"] and r["V"] < 1e-10
    r = F.entropy_variance([.5, .3, .15, .05])
    assert r["V"] == pytest.approx(r["V_basharin"])


def test_kendall_independence_value():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=4000), rng.normal(size=4000)
    r = F.kendall_tau_variance(x, y)
    assert abs(r["V"] - 4 / 9) < 0.03


def test_cluster_index_matches_brute_force():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 3, 60); b = np.where(rng.random(60) < .7, a, rng.integers(0, 3, 60))
    r = F.cluster_index_variance(a, b, "rand")
    sa = np.equal.outer(a, a); sb = np.equal.outer(b, b)
    H = (sa == sb).astype(float); np.fill_diagonal(H, np.nan)
    assert r["estimate"] == pytest.approx(np.nanmean(H))
    assert r["V"] == pytest.approx(4 * np.var(np.nanmean(H, 1), ddof=1))
    fm = F.cluster_index_variance(a, b, "fowlkes_mallows")
    assert 0 < fm["estimate"] < 1 and fm["V"] > 0


def test_r_squared_and_others_R():
    assert mp.n_for_power_group_b("r_squared", r2=0.49, delta=0.1).n == 401
    with pytest.warns(UserWarning):
        F.r_squared_variance(0.0)
    assert mp.n_for_power_group_b("risk_difference", p1=.6, p2=.45, delta=.15).n == 341      # (R)
    assert mp.n_for_power_group_b("discrimination_slope", mu1=.7, mu0=.3, sd1=.2, sd0=.2, prevalence=.4, delta=.05).n == 524  # (R)
    assert F.repeatability_variance(0.5, 2)["V"] == pytest.approx(0.9604, abs=1e-4)          # (R)
    assert F.paired_proportion_variance(.05, .12)["V"] == pytest.approx(0.1651, abs=1e-4)    # (R)


def test_refusals():
    for m in ("mahalanobis", "relative_standard_error", "timedep_auc"):
        with pytest.raises(NotImplementedError):
            mp.n_for_power_group_b(m, delta=0.1)
    with pytest.raises(KeyError):
        mp.n_for_power_group_b("not_a_metric", delta=0.1)


def test_manifest_counts():
    man = mp.group_b_manifest()
    assert len(man) == sum(len(v) for v in mp.GROUP_B_FAMILY.values())
    assert len(mp.PERCASE_REGISTRY) == 44


# ---------------------------------------------------------------- Miller–Madow and the small-n caveat (2026-11-11)

def test_miller_madow_helpers():
    import numpy as np
    import metricpower as mp
    c = np.array([50, 30, 15, 5])
    assert abs(mp.entropy_miller_madow(c) - (mp.metric_entropy(c / 100) + 3 / 200)) < 1e-12
    P = np.array([[30, 4, 2], [5, 22, 5], [3, 6, 23]])
    assert abs(mp.conditional_entropy_miller_madow(P) - (mp.metric_conditional_entropy(P.ravel() / 100, 3) + 6 / 200)) < 1e-12
    r = mp.conditional_entropy_variance(P / 100)
    assert r["bias_plugin"] == -3.0


def test_entropy_result_carries_miller_madow_note():
    import metricpower as mp
    r = mp.n_for_power_group_b("entropy", p=[.5, .3, .15, .05], delta=0.1142)
    assert abs(r.extra["miller_madow_correction"] - 1.5 / r.n) < 1e-12
    assert any("Miller" in nt for nt in r.notes)
    r = mp.n_for_power_group_b("conditional_entropy", P=[[.3, .04, .02], [.05, .22, .05], [.03, .06, .23]], delta=0.07)
    assert abs(r.extra["miller_madow_correction"] - 3.0 / r.n) < 1e-12


def test_small_n_note_attached_below_30_only():
    import metricpower as mp
    small = mp.n_for_power_group_b("bertscore", mean=0.87, sd=0.07, delta=0.087)
    assert small.n < mp.SMALL_N and any("below 30" in nt for nt in small.notes)
    big = mp.n_for_power_group_b("brier", mean=0.145, sd=0.154, delta=0.0145)
    assert big.n >= mp.SMALL_N and not any("below 30" in nt for nt in big.notes)
