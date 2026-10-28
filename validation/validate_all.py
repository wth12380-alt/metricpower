"""
validate_all.py — one Monte Carlo table for every implemented variance.

Port of inst/validation/validate_all.R. For each metric the analytic per-unit
variance V is compared with the empirical variance of the estimator under the
metric's own data-generating process, and Wald coverage is checked at the
family's reference n.  The last two columns are the required n to detect a
change of 10% and 25% of the metric value (alpha 0.05 two-sided, power 0.80).

    python validation/validate_all.py            # REPS_V = 10000 (about 5-10 min)
    python validation/validate_all.py --reps 2000 --reps-cov 1000   # quick
"""
import argparse
import math
import sys
import time
import warnings

import numpy as np
from scipy.stats import norm

import metricpower as mp
from metricpower import families as F

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10000)
ap.add_argument("--reps-cov", type=int, default=5000)
ap.add_argument("--seed", type=int, default=90210)
args = ap.parse_args()
REPS_V, REPS_COV, N_POOL, N_COV = args.reps, args.reps_cov, 10000, 60
rng = np.random.default_rng(args.seed)
Z = norm.ppf(0.975)
SE_COV = math.sqrt(0.94 * 0.06 / REPS_COV)
SE_V = math.sqrt(4 / REPS_V)
warnings.simplefilter("ignore")


# ------------------------------------------------------------------ simulators
def sim_multinom(p, g, grad=None):
    p = np.asarray(p, float); V = mp.delta_var(p, g, grad=grad)
    def draw_many(n, reps):
        return np.array([g(q) for q in rng.multinomial(n, p, size=reps) / n], float)
    def cover(n, reps):
        D = rng.multinomial(n, p, size=reps) / n
        e = np.array([g(q) for q in D]); s = np.sqrt(np.array([mp.delta_var(q, g, grad=grad) for q in D]) / n)
        ok = np.isfinite(e) & np.isfinite(s)
        return float(np.mean(np.abs(e[ok] - g(p)) <= Z * s[ok]))
    return dict(V=V, truth=float(g(p)), draw_many=draw_many, cover=cover)


def sim_rate(lam, theta=math.inf):
    V = F.var_rate(lam, theta)
    gen = (lambda size: rng.poisson(lam, size)) if math.isinf(theta) else (lambda size: rng.negative_binomial(theta, theta / (theta + lam), size))
    def draw_many(n, reps): return gen((reps, n)).mean(1)
    def cover(n, reps):
        M = gen((reps, n)); e = M.mean(1); s = np.sqrt(M.var(1, ddof=1) / n)
        return float(np.mean(np.abs(e - lam) <= Z * s))
    return dict(V=V, truth=lam, draw_many=draw_many, cover=cover)


def sim_percase(x, transform):
    r = F.var_percase_transformed(x, transform)
    h = {"identity": lambda m: m, "sqrt": lambda m: np.sqrt(np.maximum(m, 1e-12)), "exp": np.exp}[transform]
    dh = {"identity": lambda m: 1.0, "sqrt": lambda m: 1 / (2 * np.sqrt(np.maximum(m, 1e-12))), "exp": np.exp}[transform]
    def draw_many(n, reps): return h(rng.choice(x, size=(reps, n)).mean(1))
    def cover(n, reps):
        M = rng.choice(x, size=(reps, n)); mu = M.mean(1); v = M.var(1, ddof=1)
        se = np.sqrt(v * dh(mu) ** 2 / n)
        return float(np.mean(np.abs(h(mu) - r["estimate"]) <= Z * se))
    return dict(V=r["V"], truth=r["estimate"], draw_many=draw_many, cover=cover)


def sim_parametric(V, truth, draw_fn, est_se_fn):
    def draw_many(n, reps): return np.array([draw_fn(n) for _ in range(reps)])
    def cover(n, reps):
        es = np.array([est_se_fn(n) for _ in range(reps)])
        return float(np.mean(np.abs(es[:, 0] - truth) <= Z * es[:, 1]))
    return dict(V=V, truth=truth, draw_many=draw_many, cover=cover)


# ------------------------------------------------------------------ catalogue
p4 = mp.cell_prob_from_operating_point(0.70, 0.80, 0.50)
P3 = np.array([[.30, .04, .02], [.05, .22, .05], [.03, .06, .23]]); P3 /= P3.sum()
cm_args = dict(expected_cost=dict(c_fp=1, c_fn=5), net_benefit=dict(threshold=0.2), fbeta=dict(beta=2))
y = rng.binomial(1, 0.3, N_POOL); ph = 1 / (1 + np.exp(-rng.normal(-0.8 + 1.8 * y, 1)))
PC = dict(brier=(ph - y) ** 2, root_brier=(ph - y) ** 2, nll=-np.log(np.where(y == 1, ph, 1 - ph)),
          cross_entropy=-np.log(np.where(y == 1, ph, 1 - ph)), perplexity=-np.log(np.where(y == 1, ph, 1 - ph)),
          mae=np.abs(rng.normal(0, 2, N_POOL)), mse=rng.normal(0, 2, N_POOL) ** 2, rmse=rng.normal(0, 2, N_POOL) ** 2,
          mrr=1 / (1 + rng.geometric(0.45, N_POOL) - 1), mape=np.abs(rng.normal(0, 1, N_POOL)) / (2 + np.abs(rng.normal(0, .3, N_POOL))),
          smape=rng.beta(2, 6, N_POOL), mase=rng.gamma(2, 0.5, N_POOL),
          rouge_n=rng.beta(5, 4, N_POOL), rouge_1=rng.beta(6, 3, N_POOL), rouge_2=rng.beta(3, 6, N_POOL), rouge_3=rng.beta(2, 8, N_POOL),
          rouge_l=rng.beta(5, 3, N_POOL), rouge_w=rng.beta(4, 4, N_POOL), rouge_s=rng.beta(3, 5, N_POOL), rouge_su=rng.beta(4, 5, N_POOL),
          bertscore=rng.beta(20, 3, N_POOL), meteor=rng.beta(4, 4, N_POOL), flesch=rng.normal(45, 12, N_POOL),
          flesch_kincaid=rng.normal(11, 2.5, N_POOL), gunning_fog=rng.normal(13, 3, N_POOL), smog=rng.normal(12, 2, N_POOL),
          ari=rng.normal(11, 3, N_POOL), forcast=rng.normal(10, 1.5, N_POOL), levenshtein=rng.poisson(12, N_POOL).astype(float),
          damerau=rng.poisson(11, N_POOL).astype(float), hamming=rng.poisson(9, N_POOL).astype(float), jaro=rng.beta(8, 3, N_POOL),
          lcs=rng.poisson(20, N_POOL).astype(float), snr=rng.normal(22, 4, N_POOL), psnr=rng.normal(31, 5, N_POOL), cnr=rng.normal(6, 2, N_POOL),
          ssim=rng.beta(9, 2, N_POOL), ensemble_mse=rng.gamma(3, 1 / 6, N_POOL), macro_mean_prob=rng.beta(5, 5, N_POOL),
          dice_percase=rng.beta(9, 2, N_POOL), iou_percase=rng.beta(7, 3, N_POOL), nsd_percase=rng.beta(8, 2, N_POOL),
          assd_percase=rng.gamma(2, 1 / 1.2, N_POOL), hd95_percase=rng.gamma(1.4, 4, N_POOL))

CAT = {}
for nm in mp.CM_METRICS:
    a = cm_args.get(nm, {})
    CAT[nm] = dict(family="confusion", how="multinomial draw", sim=sim_multinom(p4, mp.get_metric(nm, **a), mp.get_gradient(nm, **a)), n=500)
for nm in mp.GROUP_B_FAMILY["multiclass"]:
    avg, what = nm.split("_", 1)
    CAT[nm] = dict(family="multiclass", how="k^2 multinomial draw", sim=sim_multinom(P3.ravel(), (F.metric_micro if avg == "micro" else F.metric_macro)(what)), n=500)
CAT["fppi"] = dict(family="rate", how="Poisson draw", sim=sim_rate(0.5), n=500)
CAT["fppi_overdispersed"] = dict(family="rate", how="negative binomial draw", sim=sim_rate(0.5, 2), n=500)
for nm, e in F.PERCASE_REGISTRY.items():
    CAT[nm] = dict(family="per-case", how="bootstrap resample", make=(lambda nm=nm, e=e: sim_percase(PC[nm], e["transform"])), n=N_COV)
CAT["entropy"] = dict(family="other", how="multinomial draw", sim=sim_multinom([.5, .3, .15, .05], F.metric_entropy, F.grad_entropy), n=500)
CAT["conditional_entropy"] = dict(family="other", how="k^2 multinomial draw", sim=sim_multinom(P3.ravel(), lambda q: F.metric_conditional_entropy(q, 3)), n=500)
CAT["weighted_kappa"] = dict(family="other", how="k^2 multinomial draw", sim=sim_multinom(P3.ravel(), lambda q: mp.kappa_w(np.asarray(q).reshape(3, 3), mp.weights_linear(3))), n=500)


def _r2(n):
    x = rng.normal(size=n); yv = 0.7 * x + math.sqrt(1 - 0.49) * rng.normal(size=n)
    return np.corrcoef(x, yv)[0, 1] ** 2
CAT["r_squared"] = dict(family="other", how="bivariate normal", sim=sim_parametric(F.r_squared_variance(0.49)["V"], 0.49, _r2,
                        lambda n: (lambda r2: (r2, math.sqrt(4 * r2 * (1 - r2) ** 2 / n)))(_r2(n))), n=500)


def _cv(n):
    b = rng.normal(4, 1, n); return b.std(ddof=1) / b.mean()
CAT["coefficient_of_variation"] = dict(family="other", how="normal draw", sim=sim_parametric(0.25 ** 2 * (0.5 + 0.25 ** 2), 0.25, _cv,
                                       lambda n: (lambda c: (c, math.sqrt(c ** 2 * (0.5 + c ** 2) / n)))(_cv(n))), n=400)


def _rc(n):
    d = rng.normal(0, 0.5, (n, 2)); return 1.96 * math.sqrt(2) * math.sqrt(d.var(1, ddof=1).mean())
CAT["repeatability_coefficient"] = dict(family="other", how="replicate design", sim=sim_parametric(F.repeatability_variance(0.5, 2)["V"], 1.96 * math.sqrt(2) * 0.5, _rc,
                                        lambda n: (lambda rc: (rc, math.sqrt(F.repeatability_variance(rc / (1.96 * math.sqrt(2)), 2)["V"] / n)))(_rc(n))), n=200)


def _ds(n):
    yy = rng.binomial(1, .4, n); a = rng.normal(.7, .2, yy.sum()); b = rng.normal(.3, .2, n - yy.sum())
    return a.mean() - b.mean(), math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
CAT["discrimination_slope"] = dict(family="other", how="two-group draw", sim=sim_parametric(F.discrimination_slope_variance(.7, .3, .2, .2, .4)["V"], 0.4, lambda n: _ds(n)[0], _ds), n=2000)


def _rd(n):
    a = rng.binomial(1, .6, n // 2); b = rng.binomial(1, .45, n // 2); pa, pb = a.mean(), b.mean()
    return pa - pb, math.sqrt(F.risk_difference_variance(pa, pb, .5)["V"] / n)
CAT["risk_difference"] = dict(family="other", how="two binomial draws", sim=sim_parametric(F.risk_difference_variance(.6, .45, .5)["V"], 0.15, lambda n: _rd(n)[0], _rd), n=2000)


def _pp(n):
    q = rng.multinomial(n, [.05, .12, .5, .33]) / n
    return q[1] - q[0], math.sqrt(F.paired_proportion_variance(q[0], q[1])["V"] / n)
CAT["paired_proportion"] = dict(family="other", how="4-cell multinomial", sim=sim_parametric(F.paired_proportion_variance(.05, .12)["V"], 0.07, lambda n: _pp(n)[0], _pp), n=2000)


# --- U-statistics (pair draws) ---
def _kgen(N):
    zz = rng.normal(size=N); return zz + rng.normal(size=N), zz + rng.normal(size=N)
_kbig = _kgen(4000); _kest = F.kendall_tau_variance(*_kbig)
_ktau = 2 / math.pi * math.asin(0.5)      # population tau for corr(x, y) = 0.5
def _kdraw(n): from scipy.stats import kendalltau; return float(kendalltau(*_kgen(n)).statistic)
def _kse(n): r = F.kendall_tau_variance(*_kgen(n)); return r["estimate"], math.sqrt(r["V"] / n)
CAT["kendall_tau"] = dict(family="U-statistic", how="pair draw", sim=sim_parametric(_kest["V"], _ktau, _kdraw, _kse), n=300)

Jab = np.array([[.22, .04, .02], [.03, .25, .05], [.02, .05, .32]]); Jab /= Jab.sum()
def _pair(n):
    idx = rng.choice(9, n, p=Jab.ravel()); return idx // 3, idx % 3
_cbig = _pair(3000)
_sab = float((Jab ** 2).sum()); _sa = float((Jab.sum(1) ** 2).sum()); _sb = float((Jab.sum(0) ** 2).sum())
_truth = {"rand": _sab + 1 - _sa - _sb + _sab, "fowlkes_mallows": _sab / math.sqrt(_sa * _sb)}   # population values
for _nm, _idx in (("rand_index", "rand"), ("fowlkes_mallows", "fowlkes_mallows")):
    _est = F.cluster_index_variance(*_cbig, _idx); _est["estimate"] = _truth[_idx]
    def _cdraw(n, _idx=_idx): return F.cluster_index_variance(*_pair(n), _idx)["estimate"]
    def _cse(n, _idx=_idx): r = F.cluster_index_variance(*_pair(n), _idx); return r["estimate"], math.sqrt(r["V"] / n)
    CAT[_nm] = dict(family="U-statistic", how="pair draw", sim=sim_parametric(_est["V"], _est["estimate"], _cdraw, _cse), n=400)

# ------------------------------------------------------------------ table
t0 = time.time()
print("=" * 122)
print(f"UNIFIED VALIDATION   {len(CAT)} metrics   REPS = {REPS_V:,}   delta = 10% and 25% of the mean")
print(f"Monte Carlo error: variance ratio +/- {100 * 2 * SE_V:.1f}%   coverage +/- {2 * SE_COV:.4f}")
print("=" * 122)
print(f"{'metric':<27}{'family':<13}{'simulated by':<24}{'mu':>9}{'V':>10}{'V_MC':>10}{'diff':>8}{'cover':>8}{'n(10% mu)':>11}{'n(25% mu)':>11}")
print("-" * 122)
nfail = 0
for fam in ("confusion", "multiclass", "rate", "per-case", "U-statistic", "other"):
    for nm, e in CAT.items():
        if e["family"] != fam:
            continue
        s = e["sim"] if "sim" in e else e["make"](); nn = e["n"]
        ests = s["draw_many"](nn, REPS_V); ests = ests[np.isfinite(ests)]
        V_mc = nn * ests.var(ddof=1); diff = 100 * (V_mc - s["V"]) / s["V"]
        good = abs(diff) < 12; nfail += (not good)
        cov = s["cover"](nn, REPS_COV)
        mu = abs(s["truth"])
        n10 = mp.n_for_power(s["V"], 0.10 * mu) if mu > 1e-9 else None
        n25 = mp.n_for_power(s["V"], 0.25 * mu) if mu > 1e-9 else None
        print(f"{nm:<27}{fam:<13}{e['how']:<24}{s['truth']:>9.4f}{s['V']:>10.4f}{V_mc:>10.4f}{diff:>+7.1f}%{cov:>8.4f}"
              f"{(f'{n10:,}' if n10 else '-'):>11}{(f'{n25:,}' if n25 else '-'):>11}{'' if good else '  <-- FAIL'}")
    print()
print("=" * 122)
print(f"{'ALL VALIDATED' if nfail == 0 else 'FAILURES PRESENT'}   {len(CAT)} metrics, {nfail} discrepancies beyond the noise band   ({time.time() - t0:.0f} s)")
print(f"""
Reading the columns:
  V          analytic, from the package
  V_MC       empirical, from simulation. diff within +/- {100 * 2 * SE_V:.1f}% is noise
  cover      Wald coverage at the family's own n; nominal 0.95, noise +/- {2 * SE_COV:.4f}
  mu         the metric value the simulation is centred on; the deltas are fractions of it
  n(10% mu)  cases needed to detect a change of 10% of mu (alpha 0.05 two-sided, power 0.80, Wald)
  n(25% mu)  same for 25% of mu; = n(10% mu)/6.25 because n is proportional to 1/delta^2
  -          mu is zero, so a relative delta is undefined""")
sys.exit(1 if nfail else 0)
