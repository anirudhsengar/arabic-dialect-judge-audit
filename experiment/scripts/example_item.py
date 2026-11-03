"""Export per-judge verdicts for the paper's example key (pair_en, both orders; solve) to results/analysis/example_item.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402

KEY = "us_foreign_policy::75"  # chosen because its options do not refer to other options
PENDING: set = set()  # all 8 judges complete (gpt-6.1-sol included)
meta, items, req, df, cov = A.load(False)
sub = df[(df.key == KEY) & (~df.judge.isin(PENDING))]
out = {"key": KEY, "description": "Stored judge outputs for the example key; pair_en correct per order and solve correct, by variety.",
       "judges": {}}
for j, g in sub.groupby("judge"):
    d = {}
    for cond in ("pair_en", "solve"):
        c = g[g.condition == cond]
        for r in c.itertuples():
            k = f"{cond}|{r.variety}" + (f"|order={r.order}" if cond.startswith("pair") else "")
            d[k] = None if r.correct != r.correct else int(r.correct)
    out["judges"][j] = dict(sorted(d.items()))
(ROOT / "results/analysis/example_item.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({j: {k: v for k, v in d.items() if "MSA" in k or "MAG" in k} for j, d in out["judges"].items()}, indent=1))
