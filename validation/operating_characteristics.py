"""
operating_characteristics.py — empirical type I error and power of the Wald (and, for confusion-matrix
metrics, the score) test at the sample size the package calculates, for every implemented variance.

For each metric the simulation draws data at the planning operating point (truth theta_1, analytic
per-unit variance V_1) under the metric's own data-generating process (validation/catalogue.py) and
  * type I error : rejection rate of H0: theta = theta_1 at nominal alpha = .05 (data drawn at theta_1),
                   i.e. 1 - coverage of the 95% Wald interval, at n_Wald;
  * power        : rejection rate of H0: theta = theta_0 = theta_1 - delta at n_Wald, target 0.80;
  * score power  : for confusion-matrix metrics, the same with the null SE sqrt(V_0/n) at n_score;
  * dV%          : 100 * (V_MC / V_1 - 1), the deviation of the simulated from the analytic variance.
delta is a fraction of the metric value by default (--delta-frac 0.10) or an absolute value (--delta-abs).

    python validation/operating_characteristics.py                      # 10,000 reps, ~15-30 min
    python validation/operating_characteristics.py --reps 1000          # quick look
Writes operating_characteristics.csv / .txt next to this file and prints the Results sentence.
"""
import argparse
import csv
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import norm, skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metricpower as mp
import catalogue

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10000)
ap.add_argument("--delta-frac", type=float, default=0.10, help="delta as a fraction of |theta_1| (default 0.10)")
ap.add_argument("--delta-abs", type=float, default=None, help="absolute delta on the metric scale (overrides --delta-frac)")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--power", type=float, default=0.80)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--max-n", type=int, default=60000, help="skip a metric whose n exceeds this (rare; relative delta on a tiny mean)")
args = ap.parse_args()
warnings.simplefilter("ignore")

rng = np.random.default_rng(args.seed)
CAT = catalogue.build(rng)
z_a = norm.ppf(1 - args.alpha / 2); z_b = norm.ppf(args.power)
REPS = args.reps
SE_RATE = math.sqrt(0.05 * 0.95 / REPS)          # MC standard error of a rejection rate near .05
SE_POW = math.sqrt(0.8 * 0.2 / REPS)
SE_V = math.sqrt(2 / REPS)                        # MC standard error of a variance ratio
here = os.path.dirname(os.path.abspath(__file__))

rows = []
t0 = time.time()
for fam in ("confusion", "multiclass", "rate", "per-case", "U-statistic", "other"):
    for nm, e in CAT.items():
        if e["family"] != fam:
            continue
        s = e["sim"] if "sim" in e else e["make"]()
        th1, V1 = float(s["truth"]), float(s["V"])
        delta = args.delta_abs if args.delta_abs is not None else args.delta_frac * abs(th1)
        if delta <= 0:
            rows.append(dict(metric=nm, family=fam, theta=th1, V=V1, note="theta = 0: relative delta undefined")); continue
        n_w = mp.n_for_power(V1, delta, alpha=args.alpha, power=args.power)
        if n_w > args.max_n:
            rows.append(dict(metric=nm, family=fam, theta=th1, V=V1, n_wald=n_w, note="n above --max-n, skipped")); continue
        th0 = th1 - delta
        est, se = s["est_se"](n_w, REPS)
        ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
        est, se = est[ok], se[ok]
        dV = 100 * (n_w * est.var(ddof=1) / V1 - 1)
        t1e = float(np.mean(np.abs(est - th1) > z_a * se))          # H0 at the truth, data at the truth
        pw = float(np.mean(np.abs(est - th0) > z_a * se))           # H0 at theta_0, data at theta_1, n = n_Wald
        row = dict(metric=nm, family=fam, theta=th1, V=V1, dV_pct=dV, n_wald=n_w, type1_wald=t1e, power_wald=pw,
                   undefined_pct=100 * (1 - ok.mean()), skew=s.get("skew", float("nan")))
        # score test for confusion-matrix metrics: null variance from the null operating point (sensitivity / specificity path)
        if fam == "confusion" and "p" in s:
            try:
                r = mp.n_for_power_metric(s["p"], nm, delta, alpha=args.alpha, power=args.power, test="score", null_path="auto")
                V0, n_s = float(r.V_null), int(r.n)
                est_s, _ = s["est_se"](n_s, REPS); est_s = est_s[np.isfinite(est_s)]
                se0 = math.sqrt(V0 / n_s)
                row.update(V_null=V0, n_score=n_s, power_score=float(np.mean(np.abs(est_s - th0) > z_a * se0)))
            except Exception as ex:                                 # null value unreachable on both paths
                row.update(note=f"score: {type(ex).__name__}")
        rows.append(row)
        print(f"{nm:<26}{fam:<12} n_W={n_w:>6}  dV={dV:+6.1f}%  typeI={t1e:.3f}  power={pw:.3f}"
              + (f"  | n_S={row['n_score']:>6} power_S={row['power_score']:.3f}" if "power_score" in row else ""), flush=True)

# ------------------------------------------------------------------ write
cols = ["metric", "family", "theta", "V", "dV_pct", "n_wald", "type1_wald", "power_wald", "V_null", "n_score", "power_score", "undefined_pct", "skew", "note"]
with open(os.path.join(here, "operating_characteristics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in cols})

done = [r for r in rows if "power_wald" in r]
dv = np.array([r["dV_pct"] for r in done]); t1 = np.array([r["type1_wald"] for r in done]); pw = np.array([r["power_wald"] for r in done])
off_pow = [r for r in done if abs(r["power_wald"] - args.power) > 0.02]
off_t1 = [r for r in done if abs(r["type1_wald"] - args.alpha) > 0.01]
lines = []
lines.append("=" * 118)
lines.append(f"OPERATING CHARACTERISTICS   {len(done)} metrics   REPS = {REPS:,}   delta = "
             + (f"{args.delta_abs} (absolute)" if args.delta_abs is not None else f"{100 * args.delta_frac:.0f}% of the metric value")
             + f"   alpha = {args.alpha}   target power = {args.power}")
lines.append(f"Monte Carlo error (2 SE): variance ratio +/- {200 * SE_V:.1f}%   type I error +/- {2 * SE_RATE:.3f}   power +/- {2 * SE_POW:.3f}")
lines.append("=" * 118)
lines.append(f"{'metric':<26}{'family':<12}{'theta':>8}{'V':>9}{'dV%':>8}{'n_W':>7}{'typeI':>7}{'power':>7}{'n_S':>7}{'pow_S':>7}{'skew':>7}  note")
lines.append("-" * 118)
for r in rows:
    def f(k, fmt, width):
        return (fmt % r[k]).rjust(width) if k in r and r[k] == r[k] else "-".rjust(width)
    lines.append(f"{r['metric']:<26}{r['family']:<12}{f('theta', '%.3f', 8)}{f('V', '%.4f', 9)}{f('dV_pct', '%+.1f%%', 8)}"
                 f"{f('n_wald', '%d', 7)}{f('type1_wald', '%.3f', 7)}{f('power_wald', '%.3f', 7)}{f('n_score', '%d', 7)}{f('power_score', '%.3f', 7)}{f('skew', '%.2f', 7)}  {r.get('note', '')}")
lines.append("=" * 118)
lines.append("SUMMARY")
lines.append(f"  simulated vs analytic variance: {dv.min():+.1f}% to {dv.max():+.1f}%; {int((np.abs(dv) <= 200 * SE_V).sum())} of {len(done)} within the MC noise band")
lines.append(f"  type I error (nominal {args.alpha}): {t1.min():.3f} to {t1.max():.3f}; {len(off_t1)} outside {args.alpha} +/- 0.01: " + ", ".join(f"{r['metric']} ({r['type1_wald']:.3f})" for r in off_t1))
lines.append(f"  power at n_Wald (target {args.power}): {pw.min():.3f} to {pw.max():.3f}; {len(off_pow)} outside +/- 0.02: " + ", ".join(f"{r['metric']} ({r['power_wald']:.3f})" for r in off_pow))
sc = [r for r in done if "power_score" in r]
if sc:
    ps = np.array([r["power_score"] for r in sc])
    lines.append(f"  score test (confusion-matrix metrics, n = {len(sc)}): power {ps.min():.3f} to {ps.max():.3f}")
lines.append("")
lines.append("Results sentence (fill into the manuscript):")
lines.append(f"  For all {len(done)} implemented variances the Monte Carlo variance was within {max(abs(dv.min()), abs(dv.max())):.0f}% of the analytic value "
             f"(range {dv.min():+.1f}% to {dv.max():+.1f}%). At nominal alpha = .05 the empirical type I error of the Wald test ranged from "
             f"{t1.min():.3f} to {t1.max():.3f}, and empirical power at the calculated sample size ranged from {pw.min():.2f} to {pw.max():.2f} "
             f"against a target of {args.power:.2f}; the {len(off_pow)} metrics outside +/- 0.02 of the target were "
             + (", ".join(r['metric'] for r in off_pow) if off_pow else "none") + ".")
lines.append(f"({time.time() - t0:.0f} s)")
txt = "\n".join(lines)
open(os.path.join(here, "operating_characteristics.txt"), "w").write(txt)
print(txt)
