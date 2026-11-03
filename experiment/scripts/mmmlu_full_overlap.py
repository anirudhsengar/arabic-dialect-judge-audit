"""Full-dataset MSA overlap with OpenAI MMMLU Arabic (paper Sec. 3, caveat ii).

Reads only local, pinned copies: data/raw (DialectalArabicMMLU parquet, revision d7ca5ac8)
and data/mmmlu/test/mmlu_AR-XY.csv. No API calls. Writes results/analysis/mmmlu_full_overlap.json.
Matching rule identical to mmmlu_split.py (whitespace normalisation: re \\s+ -> ' ', strip).
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()  # noqa: E731

df = pd.read_parquet(ROOT / "data/raw/data/test-00000-of-00001.parquet")
mm = pd.read_csv(ROOT / "data/mmmlu/test/mmlu_AR-XY.csv", dtype=str, keep_default_na=False)
mm_q = set(mm.Question.map(norm))
msa = df[df.dialect == "MSA"]
n_match = int(msa.question.map(lambda q: norm(q) in mm_q).sum())
out = {"n_msa_questions": int(len(msa)), "n_matched_mmmlu_ar": n_match,
       "share": n_match / len(msa), "mmmlu_rows": int(len(mm))}
(ROOT / "results/analysis/mmmlu_full_overlap.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
