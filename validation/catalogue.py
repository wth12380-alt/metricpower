"""catalogue.py — the data-generating process of every implemented variance (shared by validate_all.py
and operating_characteristics.py).  build(rng, n_pool, n_cov) returns the CAT dict: name -> dict(family, how, sim|make, n).
Each sim exposes V, truth, draw_many(n, reps), cover(n, reps) and est_se(n, reps) -> (estimates, plug-in SEs)."""
import math
import numpy as np
from scipy.stats import norm
import metricpower as mp
from metricpower import families as F

Z = norm.ppf(0.975)


def build(rng, N_POOL=10000, N_COV=60):
    globals()["rng"] = rng
    # ------------------------------------------------------------------ simulators
    def sim_multinom(p, g, grad=None):
        p = np.asarray(p, float); V = mp.delta_var(p, g, grad=grad)
        def draw_many(n, reps):
            return np.array([g(q) for q in rng.multinomial(n, p, size=reps) / n], float)
        def est_se(n, reps):
            D = rng.multinomial(n, p, size=reps) / n
            e = np.array([g(q) for q in D]); s = np.sqrt(np.array([mp.delta_var(q, g, grad=grad) for q in D]) / n)
            return e, s
        def cover(n, reps):
            e, s = est_se(n, reps); ok = np.isfinite(e) & np.isfinite(s)
            return float(np.mean(np.abs(e[ok] - g(p)) <= Z * s[ok]))
        return dict(V=V, truth=float(g(p)), draw_many=draw_many, cover=cover, est_se=est_se, p=p, g=g, grad=grad)


    def sim_rate(lam, theta=math.inf):
        V = F.var_rate(lam, theta)
        gen = (lambda size: rng.poisson(lam, size)) if math.isinf(theta) else (lambda size: rng.negative_binomial(theta, theta / (theta + lam), size))
        def draw_many(n, reps): return gen((reps, n)).mean(1)
        def est_se(n, reps):
            M = gen((reps, n)); return M.mean(1), np.sqrt(M.var(1, ddof=1) / n)
        def cover(n, reps):
            e, s = est_se(n, reps); return float(np.mean(np.abs(e - lam) <= Z * s))
        return dict(V=V, truth=lam, draw_many=draw_many, cover=cover, est_se=est_se)


    def sim_percase(x, transform):
        r = F.var_percase_transformed(x, transform)
        h = {"identity": lambda m: m, "sqrt": lambda m: np.sqrt(np.maximum(m, 1e-12)), "exp": np.exp}[transform]
        dh = {"identity": lambda m: 1.0, "sqrt": lambda m: 1 / (2 * np.sqrt(np.maximum(m, 1e-12))), "exp": np.exp}[transform]
        def draw_many(n, reps): return h(rng.choice(x, size=(reps, n)).mean(1))
        def est_se(n, reps):
            M = rng.choice(x, size=(reps, n)); mu = M.mean(1); v = M.var(1, ddof=1)
            return h(mu), np.sqrt(v * dh(mu) ** 2 / n)
        def cover(n, reps):
            e, se = est_se(n, reps); return float(np.mean(np.abs(e - r["estimate"]) <= Z * se))
        from scipy.stats import skew
        return dict(V=r["V"], truth=r["estimate"], draw_many=draw_many, cover=cover, est_se=est_se, skew=float(skew(x)))


    def sim_parametric(V, truth, draw_fn, est_se_fn):
        def draw_many(n, reps): return np.array([draw_fn(n) for _ in range(reps)])
        def est_se(n, reps):
            es = np.array([est_se_fn(n) for _ in range(reps)]); return es[:, 0], es[:, 1]
        def cover(n, reps):
            e, s = est_se(n, reps); return float(np.mean(np.abs(e - truth) <= Z * s))
        return dict(V=V, truth=truth, draw_many=draw_many, cover=cover, est_se=est_se)


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


    return CAT
