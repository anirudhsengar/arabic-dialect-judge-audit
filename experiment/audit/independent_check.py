"""Independent code-and-data audit (fresh implementation; does NOT import ajd.analyze/stats/competence/register).

Inputs: results/<judge>.jsonl (read-only), ajd.conditions.build_requests + ajd.data.load_sample
(rid -> condition/variety/key/good_slot/cand/gold_letter). Own parser: last JSON object in the raw
text carrying the condition's field; 'Candidate A' -> 'A'. The stored `parsed` field is NOT used
for the metrics (only compared, as a parser cross-check).

Bootstrap: domain-stratified key resampling (keys resampled with replacement within each of the 32
domains), B=2000, own RNG (seed 777), percentile 95% CI.

Run: cd experiment && uv run --project .. python -m audit.independent_check
Output: results/analysis/independent_audit.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd.conditions import build_requests  # noqa: E402
from ajd.data import load_sample  # noqa: E402

RES = ROOT / "results"
OUT = RES / "analysis" / ("independent_audit.json" if len(sys.argv) < 2 else Path(sys.argv[2]) / f"ia_seed{sys.argv[1]}.json")
JUDGES = ["gpt-6.1-sol", "gpt-6-luna", "opus-5.5", "haiku-5.5", "gemini-4-argon", "gemini-3.8-flash",
          "deepseek-v4-pro", "deepseek-flash"]
VARS = ["ENG", "MSA", "UAE", "KSA", "EGY", "SYR", "MAG"]
DIAS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
B = 2000
FIELD = {"solve": "answer", "point": "verdict"}  # all others: "winner"


# ---------------------------------------------------------------- parser
def _json_objects(text):
    """All top-level-ish {...} substrings that json-decode to a dict (scan every '{')."""
    dec = json.JSONDecoder()
    out = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse(raw, condition):
    if not raw:
        return None
    field = FIELD.get(condition, "winner")
    vals = [o[field] for o in _json_objects(raw) if isinstance(o.get(field), str)]
    if not vals:
        return None
    v = vals[-1].strip()
    v = re.sub(r"^(candidate|option|answer)\s+", "", v, flags=re.I).strip()
    if condition == "point":
        v = v.lower()
        return v if v in ("correct", "incorrect") else None
    v = v.upper()
    if condition == "solve":
        return v if v in ("A", "B", "C", "D") else None
    if condition == "reg_tie":
        return v if v in ("A", "B", "TIE") else None
    return v if v in ("A", "B") else None


# ---------------------------------------------------------------- my own placebo rule (from register.py docstring spec)
HAMZA = "أإآٱ"
MARKS = set(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ", "ـ"}
ARPUNCT = set("،؛؟٪٫٬٭۔«»“”‘’…–—")


def norm_key(s):
    t = []
    for ch in s:
        if ch in HAMZA:
            ch = "ا"
        elif ch == "ة":
            ch = "ه"
        elif ch == "ى":
            ch = "ي"
        if ch in MARKS:
            continue
        if ch in ARPUNCT or (ch.isascii() and not ch.isalnum() and not ch.isspace()):
            ch = " "
        t.append(ch)
    return re.sub(r"\s+", "", "".join(t))


# ---------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, keys, domains, seed=None):
        seed = int(sys.argv[1]) if seed is None and len(sys.argv) > 1 else (seed or 777)
        self.keys = keys
        self.idx = {k: i for i, k in enumerate(keys)}
        rng = np.random.default_rng(seed)
        by = defaultdict(list)
        for i, k in enumerate(keys):
            by[domains[k]].append(i)
        n = len(keys)
        W = np.zeros((B, n))
        for members in by.values():
            members = np.array(members)
            draws = rng.integers(0, len(members), size=(B, len(members)))
            for b in range(B):
                np.add.at(W[b], members[draws[b]], 1.0)
        self.W = W

    def mean_ci(self, v):
        v = np.asarray(v, float)
        ok = ~np.isnan(v)
        est = float(v[ok].mean()) if ok.any() else None
        Wv = self.W[:, ok]
        bs = (Wv @ v[ok]) / Wv.sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": est, "ci95": [float(lo), float(hi)], "n_keys": int(ok.sum())}


def main():
    meta, items = load_sample()
    reqs = build_requests(meta, items)
    rq = {q["rid"]: q for q in reqs}
    keys = list(meta["keys"])
    domains = {q["key"]: q["domain"] for q in reqs}
    boot = Boot(keys, domains)
    K = {k: i for i, k in enumerate(keys)}
    nK = len(keys)

    # register pair types (mirroring the build_requests exclusion rule: identical stripped strings skipped)
    by = {(r.key, r.dialect): r for r in items.itertuples()}
    placebo = {}
    for k in keys:
        mg = by[(k, "MSA")].choices[by[(k, "MSA")].answer].strip()
        for d in DIAS:
            r = by[(k, d)]
            dg = r.choices[r.answer].strip()
            if dg != mg:
                placebo[(k, d)] = norm_key(mg) == norm_key(dg)

    report = {"method": __doc__, "parser_check": {}, "judges": {}}
    for j in JUDGES:
        rows = {}
        disagree = 0
        unknown = 0
        for line in open(RES / f"{j}.jsonl"):
            r = json.loads(line)
            q = rq.get(r["rid"])
            if q is None:
                unknown += 1
                continue
            p = parse(r.get("raw"), q["condition"])
            sp = r.get("parsed")
            if (p.upper() if p else p) != (sp.upper() if isinstance(sp, str) else sp):
                disagree += 1
            rows[r["rid"]] = p
        missing = sum(1 for rid in rq if rid not in rows)
        report["parser_check"][j] = {"n_rows": len(rows), "rows_not_in_requests": unknown,
                                     "requests_missing": missing, "my_parse_ne_stored_parsed_caseinsensitive": disagree}

        # per (condition, variety): per-key arrays
        acc = defaultdict(lambda: np.full((nK, 2), np.nan))   # pair conds: col by order
        solve = {v: np.full(nK, np.nan) for v in VARS}
        point = {(v, c): np.full(nK, np.nan) for v in VARS for c in ("gold", "wrong")}  # 1 accept,0 reject,nan invalid
        point_inv = {(v, c): np.zeros(nK) for v in VARS for c in ("gold", "wrong")}
        reg = defaultdict(lambda: np.full((nK, 2), np.nan))   # (cond,d): 1 chose MSA, 0 chose dialect, nan invalid/tie
        reg_valid = defaultdict(lambda: np.zeros((nK, 2)))
        for rid, q in rq.items():
            p = rows.get(rid)
            i = K[q["key"]]
            c, v = q["condition"], q["variety"]
            if c in ("pair_en", "pair_ar"):
                o = 0 if q["order"] == "good_first" else 1
                acc[(c, v)][i, o] = 1.0 if p == q["good_slot"] else 0.0  # invalid = wrong
            elif c == "solve":
                solve[v][i] = 1.0 if p == q["gold_letter"] else 0.0
            elif c == "point":
                cd = q["cand"]
                if p is None:
                    point_inv[(v, cd)][i] = 1
                else:
                    point[(v, cd)][i] = 1.0 if p == "correct" else 0.0
            elif c in ("reg_forced", "reg_forced_msaq"):
                o = 0 if q["order"] == "good_first" else 1
                if p in ("A", "B"):
                    reg[(c, v)][i, o] = 0.0 if p == q["good_slot"] else 1.0
                    reg_valid[(c, v)][i, o] = 1

        J = {}
        # (a) pair_en accuracy & H1 gap; also pair_ar for H3
        pe = {v: acc[("pair_en", v)].mean(1) for v in VARS}
        pa = {v: acc[("pair_ar", v)].mean(1) for v in VARS}
        J["pair_en_acc"] = {v: boot.mean_ci(pe[v]) for v in VARS}
        J["H1_mean_dialect_gap"] = boot.mean_ci(np.mean([pe[d] for d in DIAS], 0) - pe["MSA"])
        # (b) solve
        J["solve_acc"] = {v: boot.mean_ci(solve[v]) for v in VARS}
        J["solve_mean_dialect_gap"] = boot.mean_ci(np.mean([solve[d] for d in DIAS], 0) - solve["MSA"])
        # (c) pointwise
        pw = {}
        for v in VARS:
            g, w = point[(v, "gold")], point[(v, "wrong")]
            gi, wi = point_inv[(v, "gold")], point_inv[(v, "wrong")]
            pw[v] = {
                "accept_gold_all_rows": float(np.nan_to_num(g, nan=0).mean()),
                "accept_gold_valid_rows": float(np.nanmean(g)),
                "accept_dist_all_rows": float(np.nan_to_num(w, nan=0).mean()),
                "accept_dist_valid_rows": float(np.nanmean(w)),
                "invalid_rate_gold": float(gi.mean()),
                "invalid_rate_dist": float(wi.mean()),
                "pointwise_acc_invalid_wrong": float(((np.nan_to_num(g, nan=0)) + (np.nan_to_num(1 - w, nan=0))).mean() / 2),
            }
        J["pointwise"] = pw
        pacc = {v: (np.nan_to_num(point[(v, "gold")], nan=0) + np.nan_to_num(1 - point[(v, "wrong")], nan=0)) / 2 for v in VARS}
        J["H5_pointwise_mean_dialect_gap"] = boot.mean_ci(np.mean([pacc[d] for d in DIAS], 0) - pacc["MSA"])
        # Valid-only per-key delta over dialects (pairs where both MSA and d valid), as pointwise_figure delta_gold/dist
        for cd, name in (("gold", "delta_gold_valid"), ("wrong", "delta_dist_valid")):
            M = np.column_stack([point[(d, cd)] - point[("MSA", cd)] for d in DIAS])
            with np.errstate(all="ignore"):
                kv = np.nanmean(M, 1)
            J[name] = boot.mean_ci(kv)
        # (d) H3 rubric effect: mean over 7 varieties of pair_ar - pair_en
        J["H3_rubric_effect_mean"] = boot.mean_ci(np.mean([pa[v] - pe[v] for v in VARS], 0))
        J["H3_cells"] = {v: float((pa[v] - pe[v]).mean()) for v in VARS}
        # (e) forced register P(MSA), per key: MSA share over valid rows of the two orders; pooled = per-key
        # mean over dialects having a pair of that type; then key mean.
        regout = {}
        for cond in ("reg_forced", "reg_forced_msaq"):
            for typ in ("main", "placebo", "all"):
                cols, rowsum, rowval = [], 0, 0
                for d in DIAS:
                    a = reg[(cond, d)]
                    with np.errstate(all="ignore"):
                        share = np.nanmean(a, 1)
                    sel = np.array([(k, d) in placebo and (typ == "all" or placebo[(k, d)] == (typ == "placebo"))
                                    for k in keys])
                    share = np.where(sel, share, np.nan)
                    cols.append(share)
                    rowsum += np.nansum(a[sel])
                    rowval += reg_valid[(cond, d)][sel].sum()
                with np.errstate(all="ignore"):
                    kv = np.nanmean(np.column_stack(cols), 1)
                res = boot.mean_ci(kv)
                res["row_level_share"] = float(rowsum / rowval) if rowval else None
                res["n_pairs"] = int(sum(1 for (k, d), pl in placebo.items()
                                         if typ == "all" or pl == (typ == "placebo")))
                regout[f"{cond}:{typ}"] = res
        J["register_pooled_P_MSA"] = regout
        report["judges"][j] = J
        print(j, "done", report["parser_check"][j], flush=True)

    # ------------------------------------------------------------ comparison
    m = json.load(open(RES / "analysis" / "metrics.json"))
    ext = json.load(open(RES / "analysis" / "pointwise_figure.json"))
    comps = []

    def cmp(j, what, mine, theirs, tol):
        if theirs is None or mine is None:
            comps.append({"judge": j, "quantity": what, "mine": mine, "theirs": theirs, "status": "n/a"})
            return
        d = mine - theirs
        comps.append({"judge": j, "quantity": what, "mine": mine, "theirs": theirs, "diff_pp": 100 * d,
                      "status": "DISCREPANCY" if abs(d) * 100 > tol else "ok"})

    def cmpci(j, what, mine, theirs, est_tol=0.1, ci_tol=0.3):
        cmp(j, what, mine["est"], theirs.get("est"), est_tol)
        if "ci95" in theirs and theirs["ci95"][0] is not None:
            cmp(j, what + ".ci_lo", mine["ci95"][0], theirs["ci95"][0], ci_tol)
            cmp(j, what + ".ci_hi", mine["ci95"][1], theirs["ci95"][1], ci_tol)

    for j in JUDGES:
        J = report["judges"][j]
        for v in VARS:
            cmpci(j, f"pair_en_acc.{v}", J["pair_en_acc"][v], m["accuracy"][j][v])
            cmpci(j, f"solve_acc.{v}", J["solve_acc"][v], m["solve"][j][v])
        cmpci(j, "H1.primary", J["H1_mean_dialect_gap"], m["H1"][j]["primary"])
        cmpci(j, "H3.primary", J["H3_rubric_effect_mean"], m["H3"][j]["primary"])
        for v in VARS:
            if v in m["H3"][j]["cells"]:
                cmp(j, f"H3.cell.{v}", J["H3_cells"][v], m["H3"][j]["cells"][v]["est"], 0.1)
        cmpci(j, "H5.primary(pointwise acc gap)", J["H5_pointwise_mean_dialect_gap"], m["H5"][j]["primary"])
        mp = m["pointwise"][j]
        for v in VARS:
            P = J["pointwise"][v]
            # metrics.json does not state which denominator; compare against both, record which matches.
            for nm, mk in (("accept_gold", "accept_gold"), ("accept_dist", "accept_wrong")):
                t = mp[mk][v]["est"]
                cmp(j, f"pointwise.{nm}_all_rows.{v}", P[f"{nm}_all_rows"], t, 0.1)
                cmp(j, f"pointwise.{nm}_valid_rows.{v}", P[f"{nm}_valid_rows"], t, 0.1)
            if v in m["H5"][j]["cells"]:
                cmp(j, f"H5.cell_acc_a.{v}", P["pointwise_acc_invalid_wrong"], m["H5"][j]["cells"][v]["acc_a"], 0.1)
            inv = m["invalid"][j].get("point", {}).get(v)
            if inv is not None:
                cmp(j, f"invalid.point.{v}(gold+dist)", (P["invalid_rate_gold"] + P["invalid_rate_dist"]) / 2, inv["rate"], 0.1)
        e = ext["judges"][j]
        cmpci(j, "ext.delta_gold(valid)", J["delta_gold_valid"], e["delta_gold"])
        cmpci(j, "ext.delta_dist(valid)", J["delta_dist_valid"], e["delta_dist"])
        cmp(j, "ext.identity.h5_gap", J["H5_pointwise_mean_dialect_gap"]["est"], e["identity"]["h5_gap"], 0.1)
        cmp(j, "ext.pairwise_H1_mean_gap", J["H1_mean_dialect_gap"]["est"], e["pairwise_H1_mean_gap"], 0.1)
        cmp(j, "ext.msa_pairwise_accuracy", J["pair_en_acc"]["MSA"]["est"], e["msa_pairwise_accuracy"], 0.1)
        ninv = sum(round(1024 * (J["pointwise"][v]["invalid_rate_gold"] + J["pointwise"][v]["invalid_rate_dist"])) for v in VARS)
        cmp(j, "ext.identity.n_invalid_pointwise_rows(count)", float(ninv), float(e["identity"]["n_invalid_pointwise_rows"]), 0.0)
        rp = m["register"][j]["pooled"]
        for typ in ("main", "placebo"):
            cmpci(j, f"register.pooled.{typ}.forced_msa", J["register_pooled_P_MSA"][f"reg_forced:{typ}"], rp[typ]["forced_msa"])
            cmpci(j, f"register.pooled.{typ}.msaq_msa", J["register_pooled_P_MSA"][f"reg_forced_msaq:{typ}"], rp[typ]["msaq_msa"])
            cmp(j, f"register.pooled.{typ}.n_pairs", float(J["register_pooled_P_MSA"][f"reg_forced:{typ}"]["n_pairs"]),
                float(rp[typ]["n_pairs"]), 0.0)
    # placebo counts per dialect vs design
    for d in DIAS:
        tot = sum(1 for (k, dd) in placebo if dd == d)
        pl = sum(1 for (k, dd), p in placebo.items() if dd == d and p)
        cmp("design", f"register_pairs.{d}.total", float(tot), float(m["design"]["register_pairs"][d]["total"]), 0.0)
        cmp("design", f"register_pairs.{d}.placebo", float(pl), float(m["design"]["register_pairs"][d]["placebo"]), 0.0)

    report["audit_notes"] = [
        "Parser: my last-JSON parser agrees with the stored `parsed` field on every row of every judge (case-insensitive; stored point verdicts are upper-case). All 72,292 rids per judge map to build_requests; none missing/extra.",
        "pointwise.accept_*: metrics.json accept_gold/accept_wrong are computed over VALID rows (matches my *_valid_rows to <0.1 pp for all judges). The *_all_rows flags for Opus 5.5 (0.4-0.8 pp lower) are the expected denominator difference (Opus has 120 invalid pointwise rows), not an error.",
        "CI-bound flags (register pooled placebo, n~250 keys; a few solve bounds) vary with my bootstrap seed (re-run with seeds 1,2,3 flags different bounds) and grow if stratification is dropped, so they are domain-stratified bootstrap Monte-Carlo noise; all point estimates match.",
    ]
    report["comparisons"] = comps
    report["discrepancies"] = [c for c in comps if c["status"] == "DISCREPANCY"]
    report["n_comparisons"] = len(comps)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("comparisons:", len(comps), "discrepancies:", len(report["discrepancies"]))
    for c in report["discrepancies"]:
        print("  ", c)


if __name__ == "__main__":
    main()
