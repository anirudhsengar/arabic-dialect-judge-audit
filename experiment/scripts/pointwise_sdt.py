"""Signal-detection split of the pointwise (H5) condition with log-linear correction, all 8 judges,
plus gold-acceptance gap vs solve gap.

Reads stored outputs only: results/<judge>.jsonl via ajd.analyze.load (read-only; no API calls).

Per judge and variety v (valid pointwise outputs only):
  H_v = P(accept | gold), F_v = P(accept | distractor), with the log-linear (Hautus 1995) correction
  applied to every rate: (accepts + 0.5) / (n_valid + 1), counted over keys (one candidate row per
  key x variety x cand). d' = z(H) - z(F); c = -(z(H) + z(F)) / 2.
Reported: d'(MSA), c(MSA), Δd' = mean_{5 dialects} d'(v) - d'(MSA), Δc likewise.
Gap comparison: per key, gold gap = mean over usable dialect cells of [A_gold(v) - A_gold(MSA)]
  (as pointwise_solved.py "all"), solve gap = mean over dialects of [solve(v) - solve(MSA)]
  (one-shot 4-option solve, invalid = 0), and the per-key difference-in-differences
  mean over usable dialect cells of [(A_gold(v) - A_gold(MSA)) - (solve(v) - solve(MSA))].
95% CIs: domain-stratified key bootstrap (ajd.stats.stratified_weights, B=2000, seed 2026); the same
key weights are used for every variety and candidate, so resampled keys are shared across varieties;
in each replicate H/F are recomputed (weighted counts, same log-linear correction) and d', c re-derived.
Output: results/analysis/pointwise_sdt.json and a printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
VS = ["MSA"] + DIALECTS

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)
ONE = np.ones((1, len(keys)))


def loglin(Wm, x):
    """Log-linear corrected rate(s) of key vector x (NaN = no valid output) under weights Wm."""
    m = ~np.isnan(x)
    return (Wm @ np.where(m, x, 0.0) + 0.5) / (Wm @ m + 1.0)


def sdt(Wm, g, w):
    zh, zf = norm.ppf(loglin(Wm, g)), norm.ppf(loglin(Wm, w))
    return zh - zf, -(zh + zf) / 2


def summ(est, boot):
    return {"est": float(est), "ci95": S._ci(boot)}


out = {"method": "log-linear corrected H/F over valid pointwise outputs; d'=z(H)-z(F), c=-(z(H)+z(F))/2; "
                 "domain-stratified key bootstrap B=2000 seed 2026, keys shared across varieties",
       "judges": {}}
for j in M["design"]["judges"]:
    dj = df[df.judge == j]
    po = dj[dj.condition == "point"]
    # one row per key x variety x cand expected
    assert not po.duplicated(["key", "variety", "cand"]).any(), j
    Ag = A.kmat(po[po.cand == "gold"], "accepted", keys)
    Aw = A.kmat(po[po.cand == "wrong"], "accepted", keys)
    Sv = A.kmat(dj[dj.condition == "solve"], "correct", keys)

    raw01 = []
    pt, bt = {}, {}
    for v in VS:
        g, w = Ag[v].to_numpy(float), Aw[v].to_numpy(float)
        for nm, x in (("H", g), ("F", w)):
            r = np.nanmean(x)
            if r in (0.0, 1.0):
                raw01.append(f"{v}:{nm}")
        pt[v] = tuple(a[0] for a in sdt(ONE, g, w))
        bt[v] = sdt(W, g, w)
    dd = np.mean([pt[v][0] for v in DIALECTS]) - pt["MSA"][0]
    dc = np.mean([pt[v][1] for v in DIALECTS]) - pt["MSA"][1]
    bdd = np.mean([bt[v][0] for v in DIALECTS], axis=0) - bt["MSA"][0]
    bdc = np.mean([bt[v][1] for v in DIALECTS], axis=0) - bt["MSA"][1]

    # gold-acceptance gap vs solve gap
    ga, gb = paired_mean_over(Ag, Ag["MSA"], DIALECTS)
    gold_gap = S.paired_summary(ga, gb, W, scale=False)
    sa, sb = paired_mean_over(Sv, Sv["MSA"], DIALECTS)
    solve_gap = S.paired_summary(sa, sb, W, scale=False)
    did = np.column_stack([((Ag[v] - Ag["MSA"]) - (Sv[v] - Sv["MSA"])).to_numpy(float) for v in DIALECTS])
    did_s = S.diff_summary(S.nanmean_rows(did), W)

    out["judges"][j] = {
        "hit_msa_loglin": float(loglin(ONE, Ag["MSA"].to_numpy(float))[0]),
        "fa_msa_loglin": float(loglin(ONE, Aw["MSA"].to_numpy(float))[0]),
        "dprime_msa": summ(pt["MSA"][0], bt["MSA"][0]),
        "c_msa": summ(pt["MSA"][1], bt["MSA"][1]),
        "dprime_change": summ(dd, bdd),
        "c_change": summ(dc, bdc),
        "per_variety": {v: {"dprime": float(pt[v][0]), "c": float(pt[v][1])} for v in VS},
        "raw_rates_at_0_or_1": raw01,
        "gold_gap": {k: gold_gap[k] for k in ("est", "ci95", "n_keys", "p_two")},
        "solve_gap": {k: solve_gap[k] for k in ("est", "ci95", "n_keys", "p_two")},
        "gold_minus_solve_gap": {k: did_s[k] for k in ("est", "ci95", "n_keys", "p_two")},
    }

(ROOT / "results/analysis/pointwise_sdt.json").write_text(json.dumps(out, indent=2))
ci = lambda s, k=1: f"{k*s['est']:+.2f} [{k*s['ci95'][0]:+.2f},{k*s['ci95'][1]:+.2f}]"  # noqa: E731
print(f"{'judge':17s} {'d_MSA':>18s} {'Δd':>20s} {'c_MSA':>8s} {'Δc':>20s} | {'gold gap pp':>20s} "
      f"{'solve gap pp':>20s} {'gold-solve pp':>20s}")
for j, r in out["judges"].items():
    print(f"{j:17s} {ci(r['dprime_msa']):>18s} {ci(r['dprime_change']):>20s} {r['c_msa']['est']:+8.2f} "
          f"{ci(r['c_change']):>20s} | {ci(r['gold_gap'],100):>20s} {ci(r['solve_gap'],100):>20s} "
          f"{ci(r['gold_minus_solve_gap'],100):>20s} {r['raw_rates_at_0_or_1'] or ''}")
