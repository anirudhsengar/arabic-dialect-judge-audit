"""Pointwise (H5) gold / distractor acceptance changes on keys the judge SOLVED (post hoc
competence control for the pointwise result, mirroring the pairwise solved-both gap SB).

Reads stored outputs only: results/<judge>.jsonl via ajd.analyze.load (read-only; no API calls).

For each judge and dialect v, a (key, v) cell enters only if the same judge's one-shot 4-option solve
answer is correct in BOTH v and MSA (invalid solve = not solved, as in ajd.competence). Acceptance is
over VALID pointwise outputs (as in pointwise_sdt.py / pointwise_strata.py). Per key the
dialect - MSA change is averaged over its usable dialect cells, then over keys.
Also reported: the same change on all keys (should reproduce metrics.json pointwise.*.accept_*) and
on cells NOT solved in both, for contrast.
95% CIs: domain-stratified key bootstrap (ajd.stats.stratified_weights, B=2000, seed 2026).
Output: results/analysis/pointwise_solved.json and a printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)


def change(Mx, mask=None):
    a, b = paired_mean_over(Mx, Mx["MSA"], DIALECTS, mask=mask)
    s = S.paired_summary(a, b, W, scale=False)
    return {"est": s["est"], "ci95": s["ci95"], "n_keys": s["n_keys"]}


out = {"judges": {}}
for j in M["design"]["judges"]:
    dj = df[df.judge == j]
    po = dj[dj.condition == "point"]
    Ag = A.kmat(po[po.cand == "gold"], "accepted", keys)
    Aw = A.kmat(po[po.cand == "wrong"], "accepted", keys)
    Sv = A.kmat(dj[dj.condition == "solve"], "correct", keys)
    msa = Sv["MSA"].to_numpy(float) == 1
    both = {v: (Sv[v].to_numpy(float) == 1) & msa for v in DIALECTS}
    notb = {v: ~both[v] for v in DIALECTS}
    out["judges"][j] = {
        "n_cells_solved_both": int(sum(both[v].sum() for v in DIALECTS)),
        "solved_both": {"gold": change(Ag, both), "dist": change(Aw, both)},
        "not_solved_both": {"gold": change(Ag, notb), "dist": change(Aw, notb)},
        "all": {"gold": change(Ag), "dist": change(Aw)},
    }

(ROOT / "results/analysis/pointwise_solved.json").write_text(json.dumps(out, indent=2))
f = lambda s: f"{100*s['est']:+5.1f} [{100*s['ci95'][0]:+5.1f},{100*s['ci95'][1]:+5.1f}]"  # noqa: E731
for j, r in out["judges"].items():
    print(f"{j:17s} n={r['n_cells_solved_both']:5d} SB: G {f(r['solved_both']['gold'])} D {f(r['solved_both']['dist'])} | "
          f"notSB: G {f(r['not_solved_both']['gold'])} D {f(r['not_solved_both']['dist'])} | "
          f"all: G {f(r['all']['gold'])} D {f(r['all']['dist'])}")
