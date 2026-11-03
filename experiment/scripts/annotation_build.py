"""Build the blinded native-speaker annotation package (annotation/). Reads stored data only; no API calls.

Inputs (read-only): data/sample_items.parquet (1,024-key sample, all 7 varieties), data/item_features.parquet
(ajd.features: q_word_jacc, gold_same_as_msa), results/<judge>.jsonl via ajd.analyze.load (diagnostic
stratum only; the primary sample is drawn BEFORE any result is loaded).

Design (seed 20261101):
  PRIMARY (120 cells): for each dialect (UAE, KSA, EGY, SYR, MAG) 24 (key, dialect) cells, drawn without
    reference to any model output. Domains: 24 of the 32 domains chosen at random, one key per domain.
    Within the dialect, the first 12 chosen domains are asked for a key whose dialect gold string equals the
    MSA gold string (gold_same) and the other 12 for a key whose gold string differs (gold_differs); if a
    domain has no key of the requested type, the other type is used (logged in the key file).
  DIAGNOSTIC (30 cells, 6 per dialect), excluding that dialect's primary keys, one cell per key. Criteria:
    gold_reject   MSA gold accepted but dialect gold rejected by >= 3 of the 8 judges in pointwise
                  (valid outputs only);
    low_overlap   dialect-vs-MSA question word Jaccard (q_word_jacc) in the lowest 5% for that dialect;
    grammar_fit   candidate grammatical-form mismatch (heuristic, NOT a verified mismatch): the ENG question
                  is a sentence-completion stem (does not end in '?') and the ENG gold option starts with a
                  lower-case letter (continues the stem), restricted to cells whose dialect gold string equals
                  the MSA gold string (an MSA-form completion attached to a dialect stem);
    option_ref    ENG gold or distractor refers to other options (REGEX copied verbatim from
                  experiment/scripts/construct_checks.py).
    Quota per dialect: gold_reject 2, low_overlap 2, grammar_fit 1, option_ref 1; a short pool is filled from
    the remaining criteria in that order. Within a criterion cells are drawn at random. All criteria a cell
    meets are recorded in the key file.
  Candidate order (gold / distractor -> candidate_1 / candidate_2) and row order are randomised with
  seed-derived generators. Candidates are the dialect option strings shown to the judges (gold index and
  hash-chosen distractor index, identical across varieties).

Outputs:
  annotation/sheets/annotation_<DIALECT>.csv (UTF-8 with BOM, for spreadsheet software) [+ .xlsx if openpyxl]
  annotation/key/answer_key.csv   (DO NOT give to annotators)
  annotation/key/composition.json (sample composition)
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import config as C  # noqa: E402

SEED = 20261101
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
N_PRIMARY, N_DIAG = 24, 6
QUOTA = [("gold_reject", 2), ("low_overlap", 2), ("grammar_fit", 1), ("option_ref", 1)]
LOW_OVERLAP_Q = 0.05
MIN_JUDGES_REJECT = 3
# verbatim from experiment/scripts/construct_checks.py
REGEX = (r"\b(all|none|both|neither|either)\s+(of\s+)?(the\s+)?(above|these|those|options|answers|choices)\b"
         r"|\b(all|none)\s+the\s+above\b"
         r"|^\(?[a-d]\)?(\s*,?\s*\(?[a-d]\)?)*\s*(,|and|or|&)\s*\(?[a-d]\)?\s*(only)?\s*\.?$"
         r"|^(both|only|neither|either)\s+\(?[a-d]\)?\b"
         r"|^(both|neither|either)\.?$")
PAT = re.compile(REGEX, re.I)

OUT = ROOT / "annotation"
SHEET_COLS = ["annotation_id", "english_question", "dialect_question", "candidate_1", "candidate_2",
              "question_fidelity", "dialect_naturalness", "dialect_authenticity",
              "candidate_1_status", "candidate_2_status", "grammatical_fit_1", "grammatical_fit_2",
              "option_reference_problem", "confidence", "notes", "annotator_native_dialect"]
EMPTY_COLS = SHEET_COLS[5:]

items = pd.read_parquet(C.DATA_DIR / "sample_items.parquet")
feat = pd.read_parquet(C.DATA_DIR / "item_features.parquet").set_index(["key", "variety"])
by = {(r.key, r.dialect): r for r in items.itertuples()}
keys = sorted(items.key.unique())
assert len(keys) == 1024
dom_of = {k: k.split("::", 1)[0] for k in keys}
domains = sorted(set(dom_of.values()))
assert len(domains) == 32

# ------------------------------------------------------------------ PRIMARY (no model outputs loaded yet)
rng = np.random.default_rng(SEED)
primary = []
for v in DIALECTS:
    chosen_domains = list(rng.permutation(domains)[:N_PRIMARY])
    for i, d in enumerate(chosen_domains):
        want = i < N_PRIMARY // 2  # True -> gold_same requested
        dk = [k for k in keys if dom_of[k] == d]
        pool = [k for k in dk if bool(feat.loc[(k, v), "gold_same_as_msa"]) == want]
        fallback = not pool
        if fallback:
            pool = dk
        k = pool[int(rng.integers(len(pool)))]
        primary.append({"key": k, "dialect": v, "stratum": "primary",
                        "requested_gold_stratum": "gold_same" if want else "gold_differs",
                        "stratum_fallback": fallback})
assert "ajd.analyze" not in sys.modules, "primary sample must be drawn before loading results"

# ------------------------------------------------------------------ DIAGNOSTIC criteria
from ajd import analyze as A  # noqa: E402  (loaded only now, after the primary draw)

meta, _, _, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
judges = json.loads((ROOT / "results/analysis/metrics.json").read_text())["design"]["judges"]
assert len(judges) == 8
po = df[(df.condition == "point") & (df.cand == "gold") & df.judge.isin(judges)]
acc = po.pivot_table(index=["judge", "key"], columns="variety", values="accepted", aggfunc="mean")
n_reject = {}
for v in DIALECTS:
    hit = (acc["MSA"] == 1) & (acc[v] == 0)  # NaN (invalid) compares False
    n_reject[v] = hit.groupby(level="key").sum().reindex(keys).fillna(0).astype(int)

eng = {k: by[(k, "ENG")] for k in keys}
optref = {k: bool(PAT.search(eng[k].choices[eng[k].answer].strip())
                  or PAT.search(eng[k].choices[eng[k].distractor].strip())) for k in keys}
completion = {k: (not eng[k].question.strip().endswith("?"))
              and eng[k].choices[eng[k].answer].strip()[:1].islower() for k in keys}


def criteria(k, v):
    f = feat.loc[(k, v)]
    thr = feat.xs(v, level="variety")["q_word_jacc"].quantile(LOW_OVERLAP_Q)
    return {"gold_reject": int(n_reject[v][k]) >= MIN_JUDGES_REJECT,
            "low_overlap": float(f["q_word_jacc"]) <= thr,
            "grammar_fit": completion[k] and bool(f["gold_same_as_msa"]),
            "option_ref": optref[k]}


crit = {(k, v): criteria(k, v) for v in DIALECTS for k in keys}
pool_sizes = {v: {c: int(sum(crit[(k, v)][c] for k in keys)) for c, _ in QUOTA} for v in DIALECTS}

diag = []
for v in DIALECTS:
    used = {p["key"] for p in primary if p["dialect"] == v}
    picked, deficit = [], 0
    for c, q in QUOTA:
        pool = [k for k in keys if crit[(k, v)][c] and k not in used]
        take = list(rng.permutation(pool)[:q]) if pool else []
        deficit += q - len(take)
        for k in take:
            used.add(k)
            picked.append((k, c))
    for c, _ in QUOTA:  # fill any shortfall
        if deficit <= 0:
            break
        pool = [k for k in keys if crit[(k, v)][c] and k not in used]
        for k in list(rng.permutation(pool)[:deficit]):
            used.add(k)
            picked.append((k, c))
            deficit -= 1
    assert len(picked) == N_DIAG, (v, picked)
    for k, c in picked:
        diag.append({"key": k, "dialect": v, "stratum": "diagnostic", "selected_by": c})

# ------------------------------------------------------------------ assemble rows
crng = np.random.default_rng([SEED, 1])  # candidate order
orng = np.random.default_rng([SEED, 2])  # row order
OUT.mkdir(exist_ok=True)
(OUT / "sheets").mkdir(exist_ok=True)
(OUT / "key").mkdir(exist_ok=True)
(OUT / "filled").mkdir(exist_ok=True)

key_rows, sheets = [], {}
for v in DIALECTS:
    cells = [c for c in primary + diag if c["dialect"] == v]
    cells = [cells[i] for i in orng.permutation(len(cells))]
    rows = []
    for n, c in enumerate(cells, 1):
        k = c["key"]
        r, e = by[(k, v)], eng[k]
        gold, dist = r.choices[r.answer], r.choices[r.distractor]
        gold_pos = 1 if crng.random() < 0.5 else 2
        c1, c2 = (gold, dist) if gold_pos == 1 else (dist, gold)
        aid = f"{v}-{n:03d}"
        rows.append({"annotation_id": aid, "english_question": e.question, "dialect_question": r.question,
                     "candidate_1": c1, "candidate_2": c2} | {col: "" for col in EMPTY_COLS})
        f = feat.loc[(k, v)]
        cr = crit[(k, v)]
        key_rows.append({
            "annotation_id": aid, "key": k, "domain": dom_of[k], "dialect": v, "stratum": c["stratum"],
            "gold_candidate": gold_pos, "distractor_candidate": 3 - gold_pos,
            "gold_index": int(r.answer), "distractor_index": int(r.distractor),
            "english_gold": e.choices[e.answer], "english_distractor": e.choices[e.distractor],
            "gold_same_as_msa": bool(f["gold_same_as_msa"]), "distr_same_as_msa": bool(f["distr_same_as_msa"]),
            "q_word_jacc": round(float(f["q_word_jacc"]), 4),
            "requested_gold_stratum": c.get("requested_gold_stratum", ""),
            "stratum_fallback": c.get("stratum_fallback", ""),
            "selected_by": c.get("selected_by", ""),
            "meets_criteria": ";".join(x for x, ok in cr.items() if ok),
            "n_judges_msa_gold_acc_dialect_gold_rej": int(n_reject[v][k]),
        })
    sheets[v] = rows


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


try:
    import openpyxl  # noqa: F401
    HAVE_XLSX = True
except ImportError:
    HAVE_XLSX = False

for v, rows in sheets.items():
    write_csv(OUT / "sheets" / f"annotation_{v}.csv", rows, SHEET_COLS)
    if HAVE_XLSX:
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        wb = Workbook()
        ws = wb.active
        ws.title = v
        ws.append(SHEET_COLS)
        for r in rows:
            ws.append([r[c] for c in SHEET_COLS])
        lists = {"question_fidelity": "2,1,0", "dialect_naturalness": "1,2,3,4,5",
                 "dialect_authenticity": "yes,mixed,no",
                 "candidate_1_status": "definitely_correct,definitely_incorrect,ambiguous,cannot_determine",
                 "candidate_2_status": "definitely_correct,definitely_incorrect,ambiguous,cannot_determine",
                 "grammatical_fit_1": "yes,no,uncertain", "grammatical_fit_2": "yes,no,uncertain",
                 "option_reference_problem": "yes,no", "confidence": "1,2,3,4,5"}
        for col, opts in lists.items():
            L = openpyxl.utils.get_column_letter(SHEET_COLS.index(col) + 1)
            dv = DataValidation(type="list", formula1=f'"{opts}"', allow_blank=True)
            dv.add(f"{L}2:{L}{len(rows) + 1}")
            ws.add_data_validation(dv)
        wb.save(OUT / "sheets" / f"annotation_{v}.xlsx")

kdf = pd.DataFrame(key_rows)
kdf.to_csv(OUT / "key" / "answer_key.csv", index=False, encoding="utf-8")

comp = {
    "seed": SEED, "xlsx_written": HAVE_XLSX,
    "n_items": len(kdf), "by_stratum": kdf.stratum.value_counts().to_dict(),
    "per_dialect": {}, "diagnostic_pool_sizes_before_excluding_primary": pool_sizes,
    "low_overlap_quantile": LOW_OVERLAP_Q, "min_judges_reject": MIN_JUDGES_REJECT,
}
for v in DIALECTS:
    s = kdf[kdf.dialect == v]
    p, d = s[s.stratum == "primary"], s[s.stratum == "diagnostic"]
    comp["per_dialect"][v] = {
        "primary": {"n": len(p), "n_domains": int(p.domain.nunique()),
                    "gold_same_as_msa": int(p.gold_same_as_msa.sum()),
                    "gold_differs": int((~p.gold_same_as_msa).sum()),
                    "stratum_fallbacks": int(p.stratum_fallback.astype(bool).sum()),
                    "gold_is_candidate_1": int((p.gold_candidate == 1).sum()),
                    "meets_option_ref": int(p.meets_criteria.str.contains("option_ref").sum())},
        "diagnostic": {"n": len(d), "selected_by": d.selected_by.value_counts().to_dict(),
                       "gold_is_candidate_1": int((d.gold_candidate == 1).sum())},
    }
comp["primary_total"] = {"n": int((kdf.stratum == "primary").sum()),
                         "distinct_keys": int(kdf[kdf.stratum == "primary"].key.nunique()),
                         "gold_same_as_msa": int(kdf[kdf.stratum == "primary"].gold_same_as_msa.sum()),
                         "distinct_domains": int(kdf[kdf.stratum == "primary"].domain.nunique())}
comp["gold_is_candidate_1_total"] = int((kdf.gold_candidate == 1).sum())
(OUT / "key" / "composition.json").write_text(json.dumps(comp, indent=2))
print(json.dumps(comp, indent=2))
