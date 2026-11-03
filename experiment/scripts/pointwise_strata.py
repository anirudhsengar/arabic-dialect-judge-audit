"""Pointwise (H5) gold / distractor acceptance changes split two ways (post hoc; addresses whether the
under-crediting of correct answers survives when the gold string is not a dialectal rendering, and
when the non-native KSA/UAE renderings are left out).

Reads stored outputs only: results/<judge>.jsonl via ajd.analyze.load (read-only; no API calls),
data/item_features.parquet (ajd.features: gold_same_as_msa = dialect gold string identical to the
MSA gold string after strip) and results/analysis/metrics.json (per-cell H1/H5 gaps).

For each judge (acceptance over VALID pointwise outputs, as in pointwise_sdt.py and the paper's pointwise table):
  (i)  change in gold / distractor acceptance, dialect - MSA, per key averaged over the dialect cells
       in the stratum, then over keys, for
         gold_same:    (key, dialect) cells whose dialect gold string equals the MSA gold string
                       (only the question, and possibly the distractor, is dialectal), and
         gold_differs: cells with a dialectal gold string;
  (ii) the same changes on the native-rendered dialects EGY, SYR, MAG only, plus the native-only mean
       H1 (pair_en) and H5 (point) accuracy gap = mean of metrics.json H1/H5.<j>.cells.<v>.est over
       EGY, SYR, MAG.
95% CIs: domain-stratified key bootstrap (ajd.stats.stratified_weights, B=2000, seed 2026).
Output: results/analysis/pointwise_strata.json and a printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import config as C  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
NATIVE = ["EGY", "SYR", "MAG"]

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)
feat = pd.read_parquet(C.DATA_DIR / "item_features.parquet")
gs = feat.pivot(index="key", columns="variety", values="gold_same_as_msa").reindex(keys).fillna(False).astype(bool)


def change(Mx, cols, mask=None):
    a, b = paired_mean_over(Mx, Mx["MSA"], cols, mask=mask)
    s = S.paired_summary(a, b, W, scale=False)
    return {"est": s["est"], "ci95": s["ci95"], "n_keys": s["n_keys"]}


out = {"n_cells": {"gold_same": int(gs[DIALECTS].to_numpy().sum()),
                   "gold_differs": int((~gs[DIALECTS]).to_numpy().sum())}, "judges": {}}
for j in M["design"]["judges"]:
    po = df[(df.judge == j) & (df.condition == "point")]
    Ag = A.kmat(po[po.cand == "gold"], "accepted", keys)
    Aw = A.kmat(po[po.cand == "wrong"], "accepted", keys)
    same = {v: gs[v].to_numpy() for v in DIALECTS}
    diff = {v: ~gs[v].to_numpy() for v in DIALECTS}
    r = {
        "gold_same": {"gold": change(Ag, DIALECTS, same), "dist": change(Aw, DIALECTS, same)},
        "gold_differs": {"gold": change(Ag, DIALECTS, diff), "dist": change(Aw, DIALECTS, diff)},
        "all": {"gold": change(Ag, DIALECTS), "dist": change(Aw, DIALECTS)},
        "native": {"gold": change(Ag, NATIVE), "dist": change(Aw, NATIVE),
                   "H1_gap_mean_cells": float(np.mean([M["H1"][j]["cells"][v]["est"] for v in NATIVE])),
                   "H5_gap_mean_cells": float(np.mean([M["H5"][j]["cells"][v]["est"] for v in NATIVE]))},
    }
    out["judges"][j] = r

(ROOT / "results/analysis/pointwise_strata.json").write_text(json.dumps(out, indent=2))
print("cells:", out["n_cells"])
f = lambda s: f"{100*s['est']:+5.1f} [{100*s['ci95'][0]:+5.1f},{100*s['ci95'][1]:+5.1f}]"  # noqa: E731
for j, r in out["judges"].items():
    print(f"{j:17s} same: G {f(r['gold_same']['gold'])} D {f(r['gold_same']['dist'])} | "
          f"differs: G {f(r['gold_differs']['gold'])} D {f(r['gold_differs']['dist'])} | "
          f"native: G {f(r['native']['gold'])} D {f(r['native']['dist'])} "
          f"H1 {100*r['native']['H1_gap_mean_cells']:+.1f} H5 {100*r['native']['H5_gap_mean_cells']:+.1f}")
