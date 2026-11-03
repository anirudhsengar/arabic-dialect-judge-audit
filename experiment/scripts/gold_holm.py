"""Per-judge test of the pointwise gold-acceptance change vs MSA, Holm-adjusted over 8 judges (post hoc).

Reads stored outputs only (results/<judge>.jsonl via ajd.analyze.load; read-only, no API calls) plus
results/analysis/pointwise_figure.json and identity_check.json for cross-checks.
(1) Delta Gold = per-key mean over the 5 dialects of [accepted(v) - accepted(MSA)] over VALID pointwise
    gold outputs (ajd.competence.paired_mean_over; the figure definition, = pointwise_figure delta_gold).
    p: key-level sign-flip permutation test on the per-key differences (ajd.stats.signflip: 10,000 sign
    vectors, default_rng(2026), two-sided), Holm-adjusted across the 8 judges.
(2) Check of the figure caption's "0.04 pp" bound: CONTRASTS (dialect-minus-MSA Delta Gold / Delta Dist) valid-only
    (figure definition) vs all-rows (identity_check dH / dF, invalid = not accepted); max |diff|.
(3) For completeness: LEVELS, MSA H and F valid-only (accepted over valid rows) vs all-rows; max |diff|.
Output: results/analysis/gold_holm.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
ER = json.loads((ROOT / "results/analysis/pointwise_figure.json").read_text())
IC = json.loads((ROOT / "results/analysis/identity_check.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])

out = {"method": __doc__, "judges": {}}
for j in M["design"]["judges"]:
    po = df[(df.judge == j) & (df.condition == "point")]
    assert not po.duplicated(["key", "variety", "cand"]).any(), j
    res = {}
    for cand, lab in (("gold", "delta_gold"), ("wrong", "delta_dist")):
        Mx = A.kmat(po[po.cand == cand], "accepted", keys)
        a, b = paired_mean_over(Mx, Mx["MSA"], DIALECTS)
        d = a - b
        res[lab] = {"est": float(np.nanmean(d)), "n_keys": int((~np.isnan(d)).sum())}
        if cand == "gold":
            res[lab]["p_two"] = S.signflip(d, "two-sided")
    ref = ER["judges"][j]
    assert abs(res["delta_gold"]["est"] - ref["delta_gold"]["est"]) < 1e-12, j
    assert abs(res["delta_gold"]["p_two"] - ref["delta_gold"]["p_two"]) < 1e-12, j
    assert abs(res["delta_dist"]["est"] - ref["delta_dist"]["est"]) < 1e-12, j
    ic = IC["judges"][j]
    pv = ic["per_variety"]["MSA"]
    res["contrast_valid_minus_all"] = {"dGold": res["delta_gold"]["est"] - ic["dH"],
                                       "dDist": res["delta_dist"]["est"] - ic["dF"]}
    res["msa_level_valid_minus_all"] = {"H": pv["H_valid"] - pv["H"], "F": pv["F_valid"] - pv["F"]}
    out["judges"][j] = res

J = list(out["judges"])
adj = multipletests([out["judges"][j]["delta_gold"]["p_two"] for j in J], method="holm")[1]
for j, a in zip(J, adj):
    out["judges"][j]["delta_gold"]["p_holm8"] = float(a)
out["holm_survivors_0.05"] = [j for j, a in zip(J, adj) if a < 0.05]
out["max_abs_contrast_diff"] = max(abs(v) for j in J for v in out["judges"][j]["contrast_valid_minus_all"].values())
out["max_abs_msa_level_diff"] = max(abs(v) for j in J for v in out["judges"][j]["msa_level_valid_minus_all"].values())
out["min_signflip_p_resolution"] = 1 / 10001
(ROOT / "results/analysis/gold_holm.json").write_text(json.dumps(out, indent=2))

for j in J:
    r = out["judges"][j]
    c, l = r["contrast_valid_minus_all"], r["msa_level_valid_minus_all"]
    print(f"{j:17s} dGold={100*r['delta_gold']['est']:+6.2f}pp p={r['delta_gold']['p_two']:.4g} "
          f"holm={r['delta_gold']['p_holm8']:.4g} | contrast diff G={100*c['dGold']:+.3f} D={100*c['dDist']:+.3f} "
          f"| MSA level diff H={100*l['H']:+.3f} F={100*l['F']:+.3f}")
print("max |contrast diff| pp:", 100 * out["max_abs_contrast_diff"],
      "max |MSA level diff| pp:", 100 * out["max_abs_msa_level_diff"])
