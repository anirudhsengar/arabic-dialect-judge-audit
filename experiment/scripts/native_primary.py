"""Native-primary dialect gaps (post hoc regrouping): every judge's mean dialect - MSA
gap over the native renderings EGY, SYR, MAG (primary) and over KSA, UAE (secondary; non-native
renderings), with all five dialects for reference.

Measures (as in metrics.json): pairwise accuracy pair_en (order-matched vs MSA, invalid = incorrect),
pairwise accuracy pair_ar (same), pointwise accuracy (candidate-matched), gold acceptance and distractor
acceptance (valid pointwise outputs). Per key, the gap is averaged over the stratum's dialect cells,
then over keys. 95% CIs: domain-stratified key bootstrap (ajd.stats.stratified_weights, B=2000, seed
2026); p_two: sign-flip test (uncorrected; descriptive). Reads stored outputs only (no API calls).
Output: results/analysis/native_primary.json and a printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import config as C  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

GROUPS = {"native": ["EGY", "SYR", "MAG"], "secondary": ["KSA", "UAE"], "all5": list(C.DIALECTS)}
M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)


def summ(AB, cols):
    a, b = paired_mean_over(AB[0], AB[1], cols)
    s = S.paired_summary(a, b, W, scale=False)
    return {k: s[k] for k in ("est", "ci95", "p_two", "acc_a", "acc_b", "n_keys")}


out = {"groups": GROUPS, "B": S.B, "seed": S.SEED, "judges": {}}
for j in M["design"]["judges"]:
    dj = df[df.judge == j]
    en, ar, po = (dj[dj.condition == c] for c in ("pair_en", "pair_ar", "point"))
    mats = {"pair_en": A.mpair(en, en, "correct", keys, "order", ref="MSA"),
            "pair_ar": A.mpair(ar, ar, "correct", keys, "order", ref="MSA"),
            "point": A.mpair(po, po, "correct", keys, "cand", ref="MSA")}
    for cand, name in (("gold", "gold_accept"), ("wrong", "dist_accept")):
        K = A.kmat(po[po.cand == cand], "accepted", keys)
        mats[name] = (K, K[["MSA"] * len(C.VARIETIES)].set_axis(C.VARIETIES, axis=1))
    out["judges"][j] = {g: {m: summ(AB, cols) for m, AB in mats.items()} for g, cols in GROUPS.items()}
    # consistency with metrics.json (all-5 primary)
    for h, m in (("H1", "pair_en"), ("H5", "point")):
        assert abs(out["judges"][j]["all5"][m]["est"] - M[h][j]["primary"]["est"]) < 1e-12, (j, h)

(ROOT / "results/analysis/native_primary.json").write_text(json.dumps(out, indent=2))
f = lambda s: f"{100*s['est']:+5.1f} [{100*s['ci95'][0]:+5.1f},{100*s['ci95'][1]:+5.1f}]"  # noqa: E731
for g in GROUPS:
    print(f"== {g} {GROUPS[g]}")
    print(f"{'judge':17s} {'pair_en':>20s} {'pair_ar':>20s} {'point':>20s} {'gold_acc':>20s} {'dist_acc':>20s}")
    for j, r in out["judges"].items():
        print(f"{j:17s} " + " ".join(f"{f(r[g][m]):>20s}" for m in ("pair_en", "pair_ar", "point", "gold_accept", "dist_accept")))
