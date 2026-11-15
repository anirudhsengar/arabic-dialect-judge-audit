"""Exploratory (post hoc): MMMLU-matched vs unmatched keys; option-identity counts."""
import json
import re
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

PENDING: set = set()  # all 8 judges complete (gpt-6.1-sol included)
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()  # noqa: E731

mm = pd.read_csv(ROOT / "data/mmmlu/test/mmlu_AR-XY.csv", dtype=str, keep_default_na=False)
mm_q = set(mm.Question.map(norm))
mm_q4 = {(norm(r.Question), tuple(norm(x) for x in (r.A, r.B, r.C, r.D))) for r in mm.itertuples()}

meta, items, req, df, cov = A.load(False)
keys = sorted(meta["keys"])
msa = items[items.dialect == "MSA"].set_index("key").reindex(keys)
matched = np.array([norm(q) in mm_q for q in msa.question])
matched_opts = np.array([(norm(r.question), tuple(norm(x) for x in r.choices)) in mm_q4 for r in msa.itertuples()])
W = S.stratified_weights(keys)
judges = [j for j in C.MODELS if j in set(df.judge) and j not in PENDING]

dom = pd.Series([k.split("::")[0] for k in keys])
out = {"description": "Exploratory: pair_en mean dialect gap vs MSA (5 dialects, order-matched, per key), "
                      "MMMLU-AR-matched vs unmatched keys. Matched = MSA question string equals an openai/MMMLU "
                      "test/mmlu_AR-XY.csv question after whitespace normalisation (re \\s+ -> ' ', strip). "
                      "Key-level domain-stratified bootstrap (B=2000, seed 2026) over all 1,024 keys, subset by masking "
                      "(same procedure as metrics.json strata). Primary scoring (invalid = wrong).",
       "mmmlu_source": {"repo": "openai/MMMLU", "file": "test/mmlu_AR-XY.csv", "revision": "325a01dc3e173cac1578df94120499aaca2e2504",
                        "n_rows": int(len(mm))},
       "n_keys": len(keys), "n_matched": int(matched.sum()), "n_unmatched": int((~matched).sum()),
       "n_matched_question_and_4_options": int(matched_opts.sum()),
       "unmatched_by_domain": {d: int(n) for d, n in dom[~matched].value_counts().items()},
       "judges_excluded_pending": sorted(PENDING), "judges": {}}

for j in judges:
    en = df[(df.judge == j) & (df.condition == "pair_en")]
    Am, Bm = A.mpair(en, en, "correct", keys, "order", ref="MSA")
    a, b = paired_mean_over(Am, Bm, C.DIALECTS)
    Mmsa = A.kmat(en, "correct", keys)["MSA"].to_numpy()
    res = {}
    for name, mask in (("all", np.ones(len(keys), bool)), ("matched", matched), ("unmatched", ~matched)):
        s = S.paired_summary(np.where(mask, a, np.nan), np.where(mask, b, np.nan), W, scale=False)
        res[name] = {"gap": s["est"], "gap_ci95": s["ci95"], "n_keys": s["n_keys"],
                     "dialect_acc": s["acc_a"], "msa_acc_paired": s["acc_b"],
                     "msa_acc": S.mean_summary(np.where(mask, Mmsa, np.nan), W)}
    # difference of gaps (matched - unmatched), bootstrap with the same W
    dk = a - b
    bm = S.boot_mean(W, np.where(matched, dk, np.nan)) - S.boot_mean(W, np.where(~matched, dk, np.nan))
    res["matched_minus_unmatched"] = {"est": float(np.nanmean(dk[matched]) - np.nanmean(dk[~matched])),
                                      "ci95": S._ci(bm)}
    out["judges"][j] = res

# Revision 5: 4-option identity vs MSA, per dialect (in-sample, 1,024 keys)
by = {(r.key, r.dialect): r for r in items.itertuples()}
opt4 = {}
gd = {}
for v in C.DIALECTS:
    n4 = ng = 0
    for k in keys:
        r, m = by[(k, v)], by[(k, "MSA")]
        n4 += all(x.strip() == y.strip() for x, y in zip(r.choices, m.choices))
        ng += (r.choices[r.answer].strip() == m.choices[m.answer].strip()
               and r.choices[r.distractor].strip() == m.choices[m.distractor].strip())
    opt4[v], gd[v] = int(n4), int(ng)
out["rev5_option_identity"] = {"all_4_options_identical_to_msa": opt4,
                               "gold_and_distractor_identical_to_msa": gd,
                               "note": "comparison after .strip() of each option string, out of 1,024 keys"}
out["rev5_tokenisation"] = ("features.py q_word_jacc: Jaccard over sets of word types, tokens = "
                            "re.findall(r'\\w+', text, re.UNICODE) (Python Unicode \\w: letters/digits/underscore; "
                            "punctuation dropped; no normalisation of alef/hamza/ta-marbuta/diacritics, no case-folding, "
                            "no stemming); empty union -> 1.0. q_char_sim = difflib SequenceMatcher(autojunk=False).ratio() "
                            "on raw question strings. Option identity: exact string equality after .strip().")
p = ROOT / "results/analysis/mmmlu_split.json"
p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
