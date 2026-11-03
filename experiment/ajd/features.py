"""Item-level divergence of each dialect rendering from its MSA counterpart (exploratory analysis).

For every (key, dialect): character-level similarity of the question to the MSA question
(difflib ratio), word-level Jaccard overlap, length ratio, and whether the gold/distractor
option strings are identical to MSA. Used to ask whether judge errors concentrate on items
whose dialect rendering drifts further from MSA.
"""
import re
from difflib import SequenceMatcher

import pandas as pd

from . import config as C
from .data import load_sample

_WORD = re.compile(r"\w+", re.UNICODE)


def _jaccard(a: str, b: str) -> float:
    x, y = set(_WORD.findall(a)), set(_WORD.findall(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def build_features() -> pd.DataFrame:
    _, items = load_sample()
    by = {(r.key, r.dialect): r for r in items.itertuples()}
    rows = []
    for (key, v), r in by.items():
        if v not in C.DIALECTS:
            continue
        m = by[(key, "MSA")]
        rows.append({
            "key": key, "variety": v,
            "q_char_sim": SequenceMatcher(None, r.question, m.question, autojunk=False).ratio(),
            "q_word_jacc": _jaccard(r.question, m.question),
            "q_len_ratio": len(r.question) / max(1, len(m.question)),
            "gold_same_as_msa": r.choices[r.answer].strip() == m.choices[m.answer].strip(),
            "distr_same_as_msa": r.choices[r.distractor].strip() == m.choices[m.distractor].strip(),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(C.DATA_DIR / "item_features.parquet", index=False)
    return df
