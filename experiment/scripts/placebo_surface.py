"""Break down the register-probe placebo pairs by the kind of surface difference (post hoc).

Reads only data/sample_items.parquet via ajd.analyze.load (no API calls, no results/*.jsonl writes).
Uses the paper's own placebo rule (ajd.register.pair_table / placebo_key). Each placebo pair is put
in the first matching class:
  identical_raw      : identical once leading/trailing whitespace is stripped (cannot occur: excluded)
  whitespace_only    : equal after removing all whitespace
  punct_ws_only      : equal after removing the paper's punctuation set (ajd.register._PUNCT) and whitespace
  marks_punct_ws     : additionally after removing diacritics/tatweel (ajd.register._MARKS)
  letters            : needs letter folding (hamza/alif, ta marbuta, alif maqsura)
Output: results/analysis/placebo_surface.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import config as C  # noqa: E402
from ajd import register as R  # noqa: E402

meta, items, req, df, cov = A.load(False)
items = items[items.key.isin(set(meta["keys"]))]
pairs = R.pair_table(items, C.DIALECTS)
by = {(r.key, r.dialect): r for r in items.itertuples()}
WS = re.compile(r"\s+")


def cls(m, d):
    if WS.sub("", m) == WS.sub("", d):
        return "whitespace_only"
    f = lambda s: WS.sub("", R._PUNCT.sub("", s))
    if f(m) == f(d):
        return "punct_ws_only"
    g = lambda s: f(R._MARKS.sub("", s))
    if g(m) == g(d):
        return "marks_punct_ws"
    return "letters"


tot = Counter(); per = defaultdict(Counter)
for r in pairs[pairs.placebo].itertuples():
    m, d = by[(r.key, "MSA")], by[(r.key, r.variety)]
    c = cls(m.choices[m.answer].strip(), d.choices[d.answer].strip())
    tot[c] += 1; per[r.variety][c] += 1
out = {"n_pairs": int(len(pairs)), "n_placebo": int(pairs.placebo.sum()),
       "classes": dict(tot), "by_dialect": {k: dict(v) for k, v in per.items()}}
(ROOT / "results/analysis/placebo_surface.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
