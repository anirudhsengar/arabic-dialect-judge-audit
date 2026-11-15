"""Accounting identity for pointwise (H5) accuracy, all judges, from stored outputs only.

Reads results/<judge>.jsonl via ajd.analyze.load (read-only; no API calls). Pointwise rows only.
Per judge x variety v, over ALL rows (one gold + one distractor row per key):
  H   = P(parsed == CORRECT | gold)        (invalid counts as not accepted)
  F   = P(parsed == CORRECT | distractor)  (invalid counts as not accepted)
  I_G = P(invalid | gold), I_D = P(invalid | distractor)
  A   = pointwise accuracy over gold + distractor rows, invalid = wrong (df.correct)
Identity: A = (H + 1 - F - I_D)/2, hence dA = (dH - dF - dI_D)/2 with d = mean_{5 dialects} - MSA.
Also: valid-only H/F (accepted over valid rows) and their dialect-minus-MSA change vs all-rows change.
Output: results/analysis/identity_check.json and a printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
VS = ["MSA"] + DIALECTS + ["ENG"]
TOL = 1e-12

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
po_all = df[df.condition == "point"]
n_expected = len(meta["keys"])

out = {"method": "all pointwise rows; invalid = not accepted for H/F and wrong for A; "
                 "d = mean over 5 dialects minus MSA; valid-only H/F use accepted over valid rows",
       "n_keys": n_expected, "judges": {}}
rows = []
for j in M["design"]["judges"]:
    po = po_all[po_all.judge == j]
    assert not po.duplicated(["key", "variety", "cand"]).any(), j
    per = {}
    for v in VS:
        g = po[(po.variety == v) & (po.cand == "gold")]
        w = po[(po.variety == v) & (po.cand == "wrong")]
        assert len(g) == len(w) == n_expected, (j, v, len(g), len(w))
        H = float((g.parsed == "CORRECT").mean())
        F = float((w.parsed == "CORRECT").mean())
        IG = float((~g.valid).mean())
        ID = float((~w.valid).mean())
        Acc = float(np.concatenate([g.correct.to_numpy(float), w.correct.to_numpy(float)]).mean())
        rhs = (H + 1 - F - ID) / 2
        assert abs(Acc - rhs) < TOL, (j, v, Acc, rhs)
        Hv = float(g.accepted.mean())  # NaN rows (invalid) skipped
        Fv = float(w.accepted.mean())
        per[v] = {"H": H, "F": F, "I_G": IG, "I_D": ID, "A": Acc, "A_identity_rhs": rhs,
                  "H_valid": Hv, "F_valid": Fv}

    def d(k):
        return float(np.mean([per[v][k] for v in DIALECTS]) - per["MSA"][k])
    dA, dH, dF, dIG, dID = d("A"), d("H"), d("F"), d("I_G"), d("I_D")
    rhs = (dH - dF - dID) / 2
    assert abs(dA - rhs) < TOL, (j, dA, rhs)
    dHv, dFv = d("H_valid"), d("F_valid")
    r = {"per_variety": per,
         "msa_H": per["MSA"]["H"], "msa_F": per["MSA"]["F"],
         "msa_I_G": per["MSA"]["I_G"], "msa_I_D": per["MSA"]["I_D"],
         "dA": dA, "dH": dH, "dF": dF, "dI_G": dIG, "dI_D": dID,
         "dA_identity_rhs": rhs, "identity_abs_err": abs(dA - rhs),
         "term_H": dH / 2, "term_F": -dF / 2, "term_I_D": -dID / 2,
         "valid_only": {"dH": dHv, "dF": dFv,
                        "dH_valid_minus_all": dHv - dH, "dF_valid_minus_all": dFv - dF}}
    out["judges"][j] = r
    rows.append((j, r))

out["identity_holds_all"] = True
(ROOT / "results/analysis/identity_check.json").write_text(json.dumps(out, indent=2))
pp = lambda x: f"{100*x:+6.2f}"  # noqa: E731
print(f"{'judge':17s} {'H_MSA':>6s} {'F_MSA':>6s} {'ΔA':>7s} {'ΔH':>7s} {'ΔF':>7s} {'ΔI_D':>7s} "
      f"{'-ΔI_D/2':>7s} {'ΔI_G':>7s} {'ΔHv-ΔH':>7s} {'ΔFv-ΔF':>7s}  (pp)")
for j, r in rows:
    v = r["valid_only"]
    print(f"{j:17s} {100*r['msa_H']:6.1f} {100*r['msa_F']:6.1f} {pp(r['dA'])} {pp(r['dH'])} {pp(r['dF'])} "
          f"{pp(r['dI_D'])} {pp(r['term_I_D'])} {pp(r['dI_G'])} {pp(v['dH_valid_minus_all'])} "
          f"{pp(v['dF_valid_minus_all'])}")
