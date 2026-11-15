"""Analyse the filled native-speaker annotation sheets (annotation/filled/). Reads stored outputs only; no API calls.

Inputs:
  annotation/filled/annotation_<DIALECT>_<ANNOTATOR>.csv (or .xlsx; needs openpyxl): the sheets from
      annotation/sheets/ filled by annotators. <ANNOTATOR> is any short code (e.g. A1, A2).
  annotation/key/answer_key.csv: annotation_id -> key, dialect, stratum, gold candidate (never shown to annotators).
  annotation/sheets/annotation_<DIALECT>.csv: the blank originals (to check that item columns were not edited).
  results/<judge>.jsonl via ajd.analyze.load (read-only) and results/analysis/metrics.json (judge list).

Refuses to run (exit != 0) if there is no filled sheet, if a filled sheet has no annotated row, if any row is
incomplete (unless --allow-partial, which drops incomplete rows but still refuses a sheet with none
complete), on an unknown code, on unknown/duplicated annotation ids, or if an item column was changed.

Pre-defined critical failure (per annotator x cell):
    question_fidelity == 0  OR  gold candidate status != definitely_correct
                            OR  distractor candidate status != definitely_incorrect.
A cell is VERIFIED iff every annotator who rated it reports no critical failure (conservative rule for
two annotators). Validity rate = share of PRIMARY cells without critical failure; Wilson 95% CI. Reported
per dialect, per annotator and at the cell level, with the component rates. The diagnostic stratum is
reported descriptively (by selection criterion) and never enters the validity rates or the gap.

Inter-annotator agreement (cells rated by >= 2 annotators; first two annotators per dialect in sorted
file-name order): raw agreement and Cohen's kappa for nominal fields (incl. the critical-failure flag and
the gold/distractor statuses), quadratic-weighted kappa for ordinal fields (fidelity, naturalness, confidence).

Pointwise gold-acceptance gap on verified cells (all 8 judges): per judge, acceptance of the gold (and of the
distractor) over VALID pointwise outputs, dialect - MSA, paired on the same key: d_k = mean over the
key's selected (key, dialect) cells of (acc_dialect - acc_MSA), then mean over keys (ajd.competence.paired_mean_over).
Cell sets: verified primary cells; all primary cells (same cells, no filter); failed primary cells; and
verified cells per dialect. 95% CIs: domain-stratified key bootstrap (ajd.stats.stratified_weights over the
1,024 sample keys, B=2000, seed 2026; keys without a selected cell are dropped).

Output: results/analysis/annotation_validation.json and a printed summary.
Usage: cd experiment && uv run --project .. python scripts/annotation_analysis.py [--allow-partial] [--filled-dir DIR]
"""
import argparse
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
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
ANN = ROOT / "annotation"
ITEM_COLS = ["annotation_id", "english_question", "dialect_question", "candidate_1", "candidate_2"]
STATUS = {"definitely_correct", "definitely_incorrect", "ambiguous", "cannot_determine"}
CODES = {
    "question_fidelity": {"0", "1", "2"},
    "dialect_naturalness": {"1", "2", "3", "4", "5"},
    "dialect_authenticity": {"yes", "mixed", "no"},
    "candidate_1_status": STATUS, "candidate_2_status": STATUS,
    "grammatical_fit_1": {"yes", "no", "uncertain"}, "grammatical_fit_2": {"yes", "no", "uncertain"},
    "option_reference_problem": {"yes", "no"},
    "confidence": {"1", "2", "3", "4", "5"},
}
REQUIRED = list(CODES) + ["annotator_native_dialect"]
ORDINAL = ["question_fidelity", "dialect_naturalness", "confidence"]
NOMINAL = ["dialect_authenticity", "gold_status", "dist_status", "gold_grammatical_fit", "dist_grammatical_fit",
           "option_reference_problem", "critical_failure"]
FNAME = re.compile(r"annotation_(UAE|KSA|EGY|SYR|MAG)_([A-Za-z0-9-]+)\.(csv|xlsx)$")


def die(msg):
    sys.exit(f"annotation_analysis: REFUSING TO RUN: {msg}")


def norm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s  # 2.0 -> 2 from spreadsheet software


def read_sheet(p):
    if p.suffix == ".csv":
        with open(p, encoding="utf-8-sig", newline="") as fh:
            return [{k: norm(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    try:
        import openpyxl
    except ImportError:
        die(f"{p.name}: .xlsx needs openpyxl; save the sheet as CSV (UTF-8) instead")
    ws = openpyxl.load_workbook(p, read_only=True, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    head = [norm(h) for h in rows[0]]
    return [{h: norm(v) for h, v in zip(head, r)} for r in rows[1:] if any(v is not None for v in r)]


def load_filled(fdir, allow_partial, key):
    files = sorted(p for p in Path(fdir).glob("annotation_*") if p.suffix in (".csv", ".xlsx"))
    if not files:
        die(f"no filled sheets in {fdir} (expected annotation_<DIALECT>_<ANNOTATOR>.csv)")
    out, report = [], {}
    for p in files:
        m = FNAME.fullmatch(p.name)
        if not m:
            die(f"{p.name}: file name must be annotation_<DIALECT>_<ANNOTATOR>.csv|.xlsx")
        v, who = m.group(1), m.group(2)
        rows = read_sheet(p)
        miss = [c for c in ITEM_COLS + REQUIRED if c not in (rows[0] if rows else {})]
        if miss:
            die(f"{p.name}: missing columns {miss}")
        orig = {r["annotation_id"]: r for r in read_sheet(ANN / "sheets" / f"annotation_{v}.csv")}
        ids = [r["annotation_id"] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) - set(orig):
            die(f"{p.name}: duplicated or unknown annotation_id")
        changed = [r["annotation_id"] for r in rows if any(r[c] != orig[r["annotation_id"]][c] for c in ITEM_COLS[1:])]
        if changed:
            die(f"{p.name}: item text changed in rows {changed[:10]}")
        if not any(r[c] for r in rows for c in CODES):
            die(f"{p.name}: sheet is empty (no annotations)")
        bad = [(r["annotation_id"], c, r[c]) for r in rows for c in CODES if r[c] and r[c] not in CODES[c]]
        if bad:
            die(f"{p.name}: invalid codes {bad[:10]}")
        incomplete = [r["annotation_id"] for r in rows if not all(r[c] for c in REQUIRED)]
        if incomplete and not allow_partial:
            die(f"{p.name}: {len(incomplete)} incomplete rows (e.g. {incomplete[:5]}); use --allow-partial to drop them")
        done = [r for r in rows if all(r[c] for c in REQUIRED)]
        if not done:
            die(f"{p.name}: no complete rows")
        missing_ids = sorted(set(orig) - set(ids))
        report[p.name] = {"dialect": v, "annotator": who, "n_rows": len(rows), "n_complete": len(done),
                          "n_dropped_incomplete": len(rows) - len(done), "n_missing_ids": len(missing_ids)}
        for r in done:
            out.append(r | {"dialect_file": v, "annotator": who})
    a = pd.DataFrame(out).merge(key, on="annotation_id", how="left", validate="many_to_one")
    assert a.key.notna().all() and (a.dialect == a.dialect_file).all(), "answer key mismatch"
    if a.duplicated(["annotation_id", "annotator"]).any():
        die("the same annotator code appears twice for one dialect")
    g1 = a.gold_candidate == 1
    a["gold_status"] = np.where(g1, a.candidate_1_status, a.candidate_2_status)
    a["dist_status"] = np.where(g1, a.candidate_2_status, a.candidate_1_status)
    a["gold_grammatical_fit"] = np.where(g1, a.grammatical_fit_1, a.grammatical_fit_2)
    a["dist_grammatical_fit"] = np.where(g1, a.grammatical_fit_2, a.grammatical_fit_1)
    a["fid0"] = a.question_fidelity == "0"
    a["gold_not_dc"] = a.gold_status != "definitely_correct"
    a["dist_not_di"] = a.dist_status != "definitely_incorrect"
    a["critical_failure"] = a.fid0 | a.gold_not_dc | a.dist_not_di
    for c in ORDINAL:
        a[c] = a[c].astype(int)
    return a, report


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return {"rate": None, "ci95": [None, None], "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "ci95": [float(c - h), float(c + h)], "k": int(k), "n": int(n)}


def kappa(x, y, weights=None):
    x, y = np.asarray(x), np.asarray(y)
    cats = sorted(set(x) | set(y))
    n, k = len(x), len(cats)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for a, b in zip(x, y):
        O[idx[a], idx[b]] += 1
    O /= n
    E = np.outer(O.sum(1), O.sum(0))
    if weights == "quadratic":
        vals = np.array(cats, float)
        Wt = (vals[:, None] - vals[None, :]) ** 2
        Wt = Wt / Wt.max() if Wt.max() > 0 else Wt
    else:
        Wt = 1 - np.eye(k)
    de = (Wt * E).sum()
    return None if de == 0 else float(1 - (Wt * O).sum() / de)


def agreement(a):
    out = {}
    for v in DIALECTS:
        s = a[a.dialect == v]
        who = sorted(s.annotator.unique())
        if len(who) < 2:
            continue
        x = s[s.annotator == who[0]].set_index("annotation_id")
        y = s[s.annotator == who[1]].set_index("annotation_id")
        ids = x.index.intersection(y.index)
        r = {"annotators": who[:2], "n_items": len(ids),
             "n_primary": int((x.loc[ids, "stratum"] == "primary").sum()), "fields": {}}
        for c in NOMINAL + ORDINAL:
            xa, ya = x.loc[ids, c].astype(str).to_numpy(), y.loc[ids, c].astype(str).to_numpy()
            if c in ORDINAL:
                xa, ya = xa.astype(int), ya.astype(int)
            r["fields"][c] = {"raw_agreement": float(np.mean(xa == ya)) if len(ids) else None,
                              "kappa": kappa(xa, ya, "quadratic" if c in ORDINAL else None),
                              "kappa_type": "quadratic_weighted" if c in ORDINAL else "cohen"}
        out[v] = r
    return out


def validity(a):
    p = a[a.stratum == "primary"]
    cell = p.groupby(["annotation_id", "dialect", "key"]).agg(
        n_annotators=("annotator", "nunique"), any_failure=("critical_failure", "max"),
        all_failure=("critical_failure", "min")).reset_index()
    cell["verified"] = ~cell.any_failure.astype(bool)
    out = {"per_dialect": {}, "pooled": {}}
    for v in DIALECTS + ["ALL"]:
        s = p if v == "ALL" else p[p.dialect == v]
        cs = cell if v == "ALL" else cell[cell.dialect == v]
        if s.empty:
            continue
        r = {"cell_level_verified": wilson(int(cs.verified.sum()), len(cs)),
             "cells_rated_by_2plus": int((cs.n_annotators >= 2).sum()),
             "cells_disagreeing_on_failure": int((cs.any_failure != cs.all_failure).sum()),
             "per_annotator": {}}
        for who, t in s.groupby("annotator"):
            n = len(t)
            r["per_annotator"][who] = {
                "valid": wilson(int((~t.critical_failure).sum()), n),
                "fidelity_0": wilson(int(t.fid0.sum()), n),
                "gold_not_definitely_correct": wilson(int(t.gold_not_dc.sum()), n),
                "distractor_not_definitely_incorrect": wilson(int(t.dist_not_di.sum()), n),
                "gold_status_counts": t.gold_status.value_counts().to_dict(),
                "dist_status_counts": t.dist_status.value_counts().to_dict(),
                "naturalness_mean": float(t.dialect_naturalness.mean()),
                "authenticity_counts": t.dialect_authenticity.value_counts().to_dict(),
                "gold_grammatical_fit_no": wilson(int((t.gold_grammatical_fit == "no").sum()), n),
                "option_reference_problem_yes": wilson(int((t.option_reference_problem == "yes").sum()), n),
                "confidence_mean": float(t.confidence.mean()),
                "native_dialects_reported": sorted(t.annotator_native_dialect.unique()),
            }
        (out["pooled"] if v == "ALL" else out["per_dialect"])[v] = r
    return out, cell


def diagnostic(a):
    d = a[a.stratum == "diagnostic"]
    return {c: {"n_ratings": len(t), "critical_failure": wilson(int(t.critical_failure.sum()), len(t)),
                "gold_not_definitely_correct": wilson(int(t.gold_not_dc.sum()), len(t)),
                "gold_grammatical_fit_no": wilson(int((t.gold_grammatical_fit == "no").sum()), len(t)),
                "option_reference_problem_yes": wilson(int((t.option_reference_problem == "yes").sum()), len(t))}
            for c, t in d.groupby("selected_by")}


def gaps(cell):
    from ajd import analyze as A
    meta, _, _, df, cov = A.load(False)
    assert cov["complete"], "coverage incomplete"
    keys = sorted(meta["keys"])
    W = S.stratified_weights(keys)
    judges = json.loads((ROOT / "results/analysis/metrics.json").read_text())["design"]["judges"]
    kidx = {k: i for i, k in enumerate(keys)}

    def mask_of(cells):
        m = {v: np.zeros(len(keys), bool) for v in DIALECTS}
        for r in cells.itertuples():
            m[r.dialect][kidx[r.key]] = True
        return m

    sets = {"verified_primary": cell[cell.verified], "all_primary": cell, "failed_primary": cell[~cell.verified]}
    sets |= {f"verified_primary_{v}": cell[cell.verified & (cell.dialect == v)] for v in DIALECTS}
    out = {"n_cells": {s: len(c) for s, c in sets.items()},
           "n_keys": {s: int(c.key.nunique()) for s, c in sets.items()}, "judges": {}}
    for j in judges:
        po = df[(df.judge == j) & (df.condition == "point")]
        mats = {"gold": A.kmat(po[po.cand == "gold"], "accepted", keys),
                "dist": A.kmat(po[po.cand == "wrong"], "accepted", keys)}
        r = {}
        for sname, cells in sets.items():
            if cells.empty:
                r[sname] = None
                continue
            m = mask_of(cells)
            r[sname] = {}
            for cname, Mx in mats.items():
                a, b = paired_mean_over(Mx, Mx["MSA"], DIALECTS, mask=m)
                s = S.paired_summary(a, b, W, scale=False)
                r[sname][cname] = {k: s.get(k) for k in ("est", "ci95", "acc_a", "acc_b", "n_keys")}
        out["judges"][j] = r
    return out


def composition(key, a):
    comp = {"package": json.loads((ANN / "key" / "composition.json").read_text())}
    comp["rated"] = (a.groupby(["dialect", "stratum"]).agg(n_ratings=("annotator", "size"),
                                                           n_items=("annotation_id", "nunique"),
                                                           n_annotators=("annotator", "nunique"))
                     .reset_index().to_dict(orient="records"))
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--filled-dir", default=str(ANN / "filled"))
    ap.add_argument("--out", default=str(ROOT / "results/analysis/annotation_validation.json"))
    args = ap.parse_args()
    key = pd.read_csv(ANN / "key" / "answer_key.csv")
    key = key[["annotation_id", "key", "dialect", "stratum", "gold_candidate", "selected_by", "gold_same_as_msa"]]
    a, files = load_filled(args.filled_dir, args.allow_partial, key)
    val, cell = validity(a)
    out = {"description": __doc__, "files": files, "composition": composition(key, a),
           "validity_primary": val, "diagnostic": diagnostic(a), "agreement": agreement(a),
           "pointwise_gap": gaps(cell),
           "verified_cells": cell[cell.verified][["annotation_id", "dialect", "key"]].to_dict(orient="records")}
    Path(args.out).write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"written {args.out}")
    for v, r in val["per_dialect"].items():
        c = r["cell_level_verified"]
        print(f"{v}: verified {c['k']}/{c['n']} = {100 * c['rate']:.1f}% "
              f"[{100 * c['ci95'][0]:.1f}, {100 * c['ci95'][1]:.1f}]")
    for v, r in out["agreement"].items():
        print(v, "kappa(critical_failure) =", r["fields"]["critical_failure"]["kappa"], "n =", r["n_items"])
    f = lambda s: "   n/a" if s is None or s["est"] is None else (  # noqa: E731
        f"{100 * s['est']:+5.1f} [{100 * s['ci95'][0]:+5.1f},{100 * s['ci95'][1]:+5.1f}] k={s['n_keys']}")
    for j, r in out["pointwise_gap"]["judges"].items():
        vp, ap_ = r.get("verified_primary"), r.get("all_primary")
        print(f"{j:17s} gold acc change, verified: {f(vp and vp['gold'])} | all primary: {f(ap_ and ap_['gold'])}")


if __name__ == "__main__":
    main()
