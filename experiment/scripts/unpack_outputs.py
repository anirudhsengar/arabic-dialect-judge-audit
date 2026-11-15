"""Release helper: expand data/outputs/<judge>.jsonl.gz into results/<judge>.jsonl so that
`python -m ajd analyze` and the scripts in experiment/scripts/ can read them.

The compact rows carry rid, judge, condition, variety, key, order/cand, raw and parsed. ajd.analyze.load
needs only judge, rid and raw (it re-parses raw text); the other fields are kept for convenience.
Provider metadata (served model, timestamps, finish reasons, hashed batch IDs, token usage) is in
data/provenance/ (see README), not in these rows.

Usage (from experiment/):  uv run --project .. python scripts/unpack_outputs.py
"""
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src, dst = ROOT / "data" / "outputs", ROOT / "results"
dst.mkdir(exist_ok=True)
for gz in sorted(src.glob("*.jsonl.gz")):
    out = dst / gz.name[: -len(".gz")]
    if out.exists():
        print(f"skip {out.name} (exists)")
        continue
    with gzip.open(gz, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        n = sum(1 for line in f if g.write(line))
    print(f"{out.name}: {n} rows")
