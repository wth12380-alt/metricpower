import numpy as np
import pytest

import metricpower as mp
from metricpower.core import n_calibrated, power_calibrated, n_for_power_t


def test_calibrated_matches_t_for_normal_scores():
    # with normal scores the t test is exact, so the calibrated n equals the noncentral-t n up to MC noise
    x = np.random.default_rng(0).normal(12, 2, 100_000)
    r = n_calibrated(x, 1.2, reps=20_000, seed=1)
    assert abs(r["n_exact"] - r["n_t"]) <= 1
    assert r["n_t"] == n_for_power_t(float(np.var(x, ddof=1)), 1.2)


def test_calibrated_exceeds_t_for_bounded_skewed_scores():
    # Table S4b: per-case Dice ~ Beta(9, 2), delta = 10% -> t formula 16-17, exact 19-20
    x = np.random.default_rng(1).beta(9, 2, 200_000)
    r = n_calibrated(x, 0.1 * x.mean(), reps=20_000, seed=2)
    assert 16 <= r["n_t"] <= 17
    assert 18 <= r["n_exact"] <= 21
    assert r["power_at_n_t"] < 0.78 < 0.80 <= r["power_exact"]


def test_power_calibrated_is_deterministic_and_holds_level():
    x = np.random.default_rng(2).beta(8, 2, 500)
    a = power_calibrated(x, 20, 0.05, reps=5000, seed=3)
    b = power_calibrated(x, 20, 0.05, reps=5000, seed=3)
    assert a == b
    # the calibrated critical value exceeds the t quantile for a left-skewed score at n = 20
    assert a["crit_exact"] > a["crit_t"]


def test_group_b_exact_option_updates_result():
    x = np.random.default_rng(3).beta(9, 2, 400)
    r0 = mp.n_for_power_group_b("dice_percase", x=x, delta=0.08)
    r1 = mp.n_for_power_group_b("dice_percase", x=x, delta=0.08, exact=True, seed=1)
    assert r0.extra.get("n_exact") is None
    assert r1.extra["n_t"] == r0.n
    assert r1.n == r1.extra["n_exact"] >= r0.n
    assert any("calibrated exactly" in nt for nt in r1.notes)


def test_group_b_exact_transformed_metric():
    x = np.random.default_rng(4).gamma(2, 1, 300)   # per-case squared errors; RMSE = sqrt(mean)
    r = mp.n_for_power_group_b("rmse", x=x, delta=0.1, exact=True, seed=1)
    assert r.n >= 2 and r.extra["n_exact"] == r.n


def test_calibrated_rejects_bad_input():
    with pytest.raises(ValueError):
        n_calibrated([0.5], 0.1)
    with pytest.raises(ValueError):
        n_calibrated([0.5, 0.5, 0.5], 0.1)
    with pytest.raises(ValueError):
        n_calibrated(np.random.default_rng(0).beta(9, 2, 50), 0.0)
