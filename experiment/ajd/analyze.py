"""Turn raw judge outputs into metrics, LaTeX tables and figures (spec, incl. pre-analysis amendment A1).

Pipeline (``analyze(mock)``):

1. :func:`load` joins every result row to the frozen request table by ``rid`` and re-parses the raw
   text with the documented rule. A row is *valid* iff the parsed label is one the condition allows.
   Transport errors (requests with no result row) are **missing**, not wrong; coverage is computed
   per judge x condition and asserted: a real run stops before any test if a cell is incomplete
   (beyond ``COVERAGE_WAIVERS``) unless ``allow_incomplete`` (then the metrics are marked non-final);
   mock runs only warn (``metrics['coverage_complete']``).
2. Paired gaps are order-matched (DR3a): d_k = mean over orders/candidates present on both sides of
   (x_v - x_ref) (:func:`kmatched`). Everything is reduced to per-key vectors (see ``ajd.stats``): primary scoring counts invalid as
   wrong; sensitivity versions use complete cases (invalid dropped) and invalid = 0.5.
3. Confirmatory families (A1, DR3b), Holm within each, all on key-level sign-flip tests:
     H1  pair_en mean dialect gap (5 dialects) vs MSA, one test per judge, two-sided;
     H2  the same statistic, one-sided dialect < MSA (one test per judge);
     H1s secondary: the judge x dialect cells (40 for 8 judges) (two-sided);
     H3  pair_ar - pair_en averaged over the 7 varieties, per judge (secondary: judge x 7 variety cells);
     H4  repetition effect acc(verbose) - acc(pair_en) averaged over varieties, 128-key subset, per
         judge; class uses two-sided p_holm, one-sided "lowers" p_holm_less also reported (which
         one is confirmatory is still to be pinned in the spec);
     H5  pointwise mean dialect gap, per judge (secondary: judge x dialect cells).
   ENG - MSA is reported as context, outside every family. Each primary gap gets an equivalence
   class (delta = 2 pp on the 90% CI), error ratio and log-odds difference. Class precedence:
   "equivalent" (90% CI inside +-delta) is checked before "different" (Holm rejects), so a small,
   precisely estimated, significant gap is labelled equivalent; both flags (within_margin,
   holm_reject) are stored.
4. Exploratory/headline decompositions: judging vs competence (``ajd.competence``, DR5), register
   placebo/main split and length-parity logistic (``ajd.register``, DR6), strata by candidates
   identical to MSA and a q_char_sim dose-response (DR7), stability with CI and noise floor
   (DR9/DR11), MDE per condition (DR3f), invalid rates per judge x variety x condition (DR4).

Judge comparisons are descriptive only (DR11). Mock runs read and write only ``results_mock/``
(metrics, tables and figures all go to ``results_mock/analysis``); real runs write metrics to
``results/analysis`` (with tables/ and figures/ subfolders).
"""
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import stats as S
from .competence import competence_analysis, paired_mean_over
from .conditions import LETTERS, build_requests
from .data import load_sample
from .register import pair_table, register_analysis
from .report import write_figures, write_tables  # noqa: F401  (re-exported)
from .runner import parse

B = S.B
PAIR_CONDS = ["pair_en", "pair_ar", "verbose", "verbose_gold", "pair_en_rep"]
REG_CONDS = ["reg_tie", "reg_forced", "reg_forced_msaq"]
ALLOWED = {"solve": {"A", "B", "C", "D"}, "point": {"CORRECT", "INCORRECT"}, "reg_tie": {"A", "B", "TIE"}}
ALLOWED_DEFAULT = {"A", "B"}
SUBSET_CONDS = {"verbose", "verbose_gold", "pair_en_rep"}  # analysed on the 128-key subset (W128)
# Pre-specified coverage waivers (spec DR1: Opus pair_ar was deferred until the rubric check). A waived
# cell may be entirely absent (0 rows) without failing the coverage assertion; a partial cell still fails.
# The Opus pair_ar cell is now complete, so the waiver has been retired (empty set).
COVERAGE_WAIVERS: set = set()


# ============================================================================ loading & coverage
def load(mock=False):
    """Load, join and score all result rows.

    Returns (meta, items, req, df, coverage). ``df`` has one row per (judge, rid) present, with
    columns from the request table plus: parsed, valid, correct (invalid = 0), correct_cc
    (invalid = NaN), correct_half (invalid = 0.5), accepted / chose_A (valid rows only, else NaN),
    reg_outcome in {dialect, msa, tie, invalid}.
    """
    meta, items = load_sample()
    req = pd.DataFrame(build_requests(meta, items)).drop(columns=["prompt"])
    rdir = C.MOCK_RESULTS_DIR if mock else C.RESULTS_DIR
    rows, errs = [], {}
    for judge in C.MODELS:
        p, e = rdir / f"{judge}.jsonl", rdir / f"{judge}.errors.jsonl"
        if p.exists():
            rows += [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
        errs[judge] = ({json.loads(line)["rid"] for line in e.open(encoding="utf-8") if line.strip()}
                       if e.exists() else set())
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit(f"No results in {rdir}")
    res = res.drop_duplicates(["judge", "rid"], keep="last")
    res["parsed"] = res["raw"].map(parse)  # always re-parse raw text with the current, documented rule
    orphans = res[~res.rid.isin(req.rid)].groupby("judge").size().to_dict()
    df = res[["judge", "rid", "parsed"]].merge(req, on="rid", how="inner", validate="many_to_one")
    allowed = df["condition"].map(lambda c: ALLOWED.get(c, ALLOWED_DEFAULT))
    df["valid"] = [p in a for p, a in zip(df["parsed"], allowed)]
    df.loc[~df["valid"], "parsed"] = None

    target = pd.Series(np.nan, index=df.index, dtype=object)
    pair = df["condition"].isin(PAIR_CONDS)
    target[pair] = df.loc[pair, "good_slot"]
    sv = df["condition"] == "solve"
    target[sv] = df.loc[sv, "gold_letter"]
    pt = df["condition"] == "point"
    target[pt] = np.where(df.loc[pt, "cand"] == "gold", "CORRECT", "INCORRECT")
    hit = (df["parsed"] == target).astype(float)
    df["correct"] = np.where(pair | sv | pt, hit, np.nan)
    df["correct_cc"] = np.where(df["valid"], df["correct"], np.nan)
    df["correct_half"] = np.where(df["valid"], df["correct"], 0.5)
    df.loc[~(pair | sv | pt), ["correct_cc", "correct_half"]] = np.nan
    df["accepted"] = np.where(df["valid"], (df["parsed"] == "CORRECT").astype(float), np.nan)
    df["chose_A"] = np.where(df["valid"], (df["parsed"] == "A").astype(float), np.nan)
    reg = df["condition"].isin(REG_CONDS)
    other = np.where(df["good_slot"] == "A", "B", "A")
    df["reg_outcome"] = np.where(reg, np.select(
        [df["parsed"] == df["good_slot"], df["parsed"] == other, df["parsed"] == "TIE"],
        ["dialect", "msa", "tie"], "invalid"), None)

    coverage = coverage_report(req, df, errs, orphans)
    return meta, items, req, df, coverage


def coverage_report(req, df, errs, orphans):
    """Coverage per judge x condition: present / expected rows and transport errors still pending.

    ``complete`` is True only if every judge with any rows has every expected request of every
    condition. Missing rows are treated as missing data downstream (never as wrong).
    """
    expected = req.groupby("condition").size().to_dict()
    rid_cond = req.set_index("rid")["condition"]
    by = {}
    incomplete = []
    for j in [j for j in C.MODELS if j in set(df.judge)]:
        dj = df[df.judge == j]
        have = dj.groupby("condition").size().to_dict()
        pend = rid_cond.reindex(list(errs.get(j, set()) - set(dj.rid))).dropna().value_counts().to_dict()
        by[j] = {}
        for c, n in expected.items():
            got = int(have.get(c, 0))
            by[j][c] = {"present": got, "expected": int(n), "frac": got / n,
                        "transport_pending": int(pend.get(c, 0))}
            if got < n:
                incomplete.append(f"{j}/{c} {got}/{n}")
    blocking = [c for c in incomplete
                if not (tuple(c.split(" ")[0].split("/", 1)) in COVERAGE_WAIVERS and c.split(" ")[1].startswith("0/"))]
    return {"complete": not incomplete, "complete_except_waivers": not blocking, "blocking_cells": blocking,
            "waivers": sorted("/".join(w) for w in COVERAGE_WAIVERS), "incomplete_cells": incomplete, "by_judge": by,
            "orphan_rows": {k: int(v) for k, v in orphans.items()}}


def _warn_coverage(cov):
    if cov["complete"]:
        return
    bar = "!" * 78
    msg = [bar, "!!! COVERAGE INCOMPLETE: results below use the rows present; missing = missing data.",
           f"!!! {len(cov['incomplete_cells'])} judge x condition cells incomplete, e.g.:"]
    msg += [f"!!!   {c}" for c in cov["incomplete_cells"][:12]]
    msg += ["!!! metrics.json['coverage_complete'] = False. Do NOT report these numbers as final.", bar]
    print("\n".join(msg), file=sys.stderr)


# ============================================================================ key matrices
def kmat(sub, col, keys, cols=C.VARIETIES):
    """keys x varieties DataFrame of per-key means of ``col`` (NaN where no row)."""
    if sub.empty:
        m = pd.DataFrame(np.nan, index=keys, columns=cols)
    else:
        m = sub.groupby(["key", "variety"])[col].mean().unstack().reindex(index=keys, columns=cols)
    m.index.name, m.columns.name = "key", "variety"
    return m.astype(float)


def omat(sub, col, unit, cols=C.VARIETIES):
    """(key, unit) x varieties order/candidate-level matrix of ``col`` (NaN where no row or NaN value)."""
    if sub.empty:
        return pd.DataFrame(columns=list(cols), dtype=float)
    m = sub.pivot_table(index=["key", unit], columns="variety", values=col, aggfunc="mean", dropna=False)
    return m.reindex(columns=list(cols)).astype(float)


def kmatched(Oa, Ob, keys, cols=C.VARIETIES, ref=None):
    """Order-matched key matrices (A, B) for paired gaps (DR3a): d_k = mean over units u present on
    *both* sides of (a_ku - b_ku), so A[c] / B[c] are per-key means of the a / b side over the units
    where both are non-NaN. ``Ob[c]`` is the reference for column c, or ``Ob[ref]`` for every c
    (e.g. ref="MSA" for gaps vs MSA within one condition). A[c] - B[c] is then exactly the
    order-matched per-key difference, and A/B can be masked and averaged like any key matrix."""
    idx = Oa.index.union(Ob.index)
    Oa, Ob = Oa.reindex(idx), Ob.reindex(idx)
    A, B = {}, {}
    for c in cols:
        a = Oa[c] if c in Oa else pd.Series(np.nan, index=idx)
        b = Ob[ref if ref is not None else c] if (ref or c) in Ob else pd.Series(np.nan, index=idx)
        ok = a.notna() & b.notna()
        g = lambda x: x.where(ok).groupby(level="key").mean().reindex(keys)  # noqa: E731
        A[c], B[c] = g(a), g(b)
    A, B = pd.DataFrame(A, index=keys), pd.DataFrame(B, index=keys)
    for m in (A, B):
        m.index.name, m.columns.name = "key", "variety"
    return A.astype(float), B.astype(float)


def mpair(sa, sb, col, keys, unit, cols=C.VARIETIES, ref=None):
    """Shorthand: order-matched (A, B) from two row sets (``sb`` may be ``sa``, with ref="MSA")."""
    return kmatched(omat(sa, col, unit, cols), omat(sb, col, unit, cols), keys, cols, ref)


def _mask(M, mask):
    """Apply a per-key boolean vector or a keys x cols boolean frame to a key matrix."""
    if isinstance(mask, pd.DataFrame):
        return M.where(mask.reindex(index=M.index, columns=M.columns).fillna(False).to_numpy(bool))
    return M.where(np.repeat(np.asarray(mask, bool)[:, None], M.shape[1], 1))


def gap_block(AB, W, unit_rows=None, unit="order"):
    """Gaps vs MSA from order-matched key matrices ``AB = (A, B)`` (``mpair(rows, rows, ..., ref="MSA")``):
    cells for every non-MSA variety + dialect mean, d_k = mean over orders of (correct_v - correct_MSA).

    Returns {"cells": {v: paired_summary}, "primary": paired_summary of the per-key mean over the
    5 dialects vs MSA}. If ``unit_rows`` (order- or candidate-level rows with column ``correct``) is
    given, each cell also gets the order-level exact McNemar (sensitivity only).
    """
    A, Bm = AB
    out = {"cells": {}}
    for v in C.VARIETIES:
        if v == "MSA":
            continue
        out["cells"][v] = S.paired_summary(A[v], Bm[v], W)
        if unit_rows is not None:
            out["cells"][v]["mcnemar_order_level"] = _mcnemar(unit_rows, v, unit)
    a, b = paired_mean_over(A, Bm, C.DIALECTS)
    out["primary"] = S.paired_summary(a, b, W)
    if unit_rows is not None:
        out["primary"]["mcnemar_order_level_stacked"] = _mcnemar(unit_rows, C.DIALECTS, unit)
    return out


def _mcnemar(rows, v, unit):
    vs = [v] if isinstance(v, str) else list(v)
    m = rows[rows.variety == "MSA"][["key", unit, "correct"]]
    d = rows[rows.variety.isin(vs)][["key", unit, "variety", "correct"]].merge(m, on=["key", unit], suffixes=("", "_msa"))
    return S.mcnemar_exact(d.correct.to_numpy(float), d.correct_msa.to_numpy(float))


def _subset(M, keys_sub):
    return M.reindex(keys_sub)


def _discordance(rows_a, rows_b, unit, on=("variety",)):
    """Order-level discordance rate and design effect deff = m * Var(d_k) / Var(e_ko) (DR3f)."""
    k = ["key", unit, *on]
    m = rows_a[k + ["correct"]].merge(rows_b[k + ["correct"]], on=k, suffixes=("_a", "_b"))
    if m.empty:
        return {"discordance": None, "deff": None}
    m["e"] = m.correct_a - m.correct_b
    dk = m.groupby(["key", *on]).e.mean()
    per = m.groupby(["key", *on]).size().mean()
    ve = m.e.var(ddof=1)
    return {"discordance": float((m.e != 0).mean()),
            "deff": float(per * dk.var(ddof=1) / ve) if ve and ve > 0 else None}


# ============================================================================ main analysis
def analyze(mock=False, allow_incomplete=False):
    """Run the full pre-specified analysis and write metrics.json, tables and figures.

    Real (non-mock) runs **assert coverage** before any test (DR4): if any judge x condition cell is
    incomplete (beyond the pre-specified ``COVERAGE_WAIVERS``, e.g. Opus pair_ar deferred and
    entirely absent) the run stops and writes nothing, unless ``allow_incomplete=True`` (CLI
    ``--allow-incomplete``), in which case metrics are written but marked non-final.
    Mock runs only warn (missing rows are planted on purpose).
    """
    meta, items, req, df, cov = load(mock)
    _warn_coverage(cov)
    if not mock and not cov["complete_except_waivers"] and not allow_incomplete:
        raise SystemExit("ABORT (DR4): coverage incomplete in " + ", ".join(cov["blocking_cells"][:20])
                         + (" ..." if len(cov["blocking_cells"]) > 20 else "")
                         + ". Finish/re-request the missing rows, or pass --allow-incomplete for a "
                           "non-final run.")
    keys = sorted(meta["keys"])
    vkeys = sorted(meta["verbose_keys"])
    W = S.stratified_weights(keys)
    W128 = S.stratified_weights(vkeys)
    judges = [j for j in C.MODELS if j in set(df["judge"])]
    flagged_ok = meta.get("flagged_keys_in_sample") is not None
    flagged = set(meta.get("flagged_keys_in_sample") or [])
    ifm_note = ("flagged_keys_in_sample unavailable (verified subset failed to load): ifm_unflagged == "
                "primary and ifm_flagged_only is empty; the IFM sensitivity is NOT informative")
    if not flagged_ok:
        print(f"WARNING: {ifm_note}", file=sys.stderr)
    unflag = np.array([k not in flagged for k in keys])
    feat = pd.read_parquet(C.DATA_DIR / "item_features.parquet")
    feat["cands_same"] = feat.gold_same_as_msa & feat.distr_same_as_msa
    cands_same = feat.pivot(index="key", columns="variety", values="cands_same").reindex(keys).fillna(False).astype(bool)
    pairs = pair_table(items, C.DIALECTS)
    distr_letter = items[items.dialect == "MSA"].set_index("key")["distractor"].map(lambda i: LETTERS[i])

    out = {"schema_version": "A1.1", "mock": bool(mock), "coverage_complete": cov["complete"],
           "coverage_complete_except_waivers": cov["complete_except_waivers"],
           "allow_incomplete_override": bool(allow_incomplete and not cov["complete_except_waivers"]),
           "coverage": cov,
           "design": {"n_keys": len(keys), "n_subset_keys": len(vkeys), "B": B, "n_flips": S.N_FLIPS,
                      "seed": S.SEED, "delta": S.DELTA, "judges": judges,
                      "register_pairs": {d: {"total": int((pairs.variety == d).sum()),
                                             "placebo": int(((pairs.variety == d) & pairs.placebo).sum())}
                                         for d in C.DIALECTS},
                      "cands_same_as_msa": {d: int(cands_same[d].sum()) for d in C.DIALECTS}},
           "invalid": {}, "accuracy": {}, "solve": {}, "H1": {}, "H5": {}, "H3": {}, "H4": {},
           "eng_context": {}, "sensitivity": {}, "strata": {}, "competence": {}, "pointwise": {},
           "register": {}, "position": {}, "stability": {}, "mde": {}}

    for j in judges:
        dj = df[df.judge == j]
        cond = {c: dj[dj.condition == c] for c in sorted(set(req.condition))}
        en, sv, po, ar = cond["pair_en"], cond["solve"], cond["point"], cond["pair_ar"]
        M = {col: kmat(en, col, keys) for col in ("correct", "correct_cc", "correct_half")}
        G = {col: mpair(en, en, col, keys, "order", ref="MSA") for col in M}  # order-matched vs MSA
        Msv = kmat(sv, "correct", keys)
        Mpo = kmat(po, "correct", keys)

        out["invalid"][j] = invalid_rates(cond, keys, W, W128, vkeys)
        out["accuracy"][j] = {v: S.mean_summary(M["correct"][v], W) for v in C.VARIETIES}
        out["solve"][j] = {v: S.mean_summary(Msv[v], W) for v in C.VARIETIES}

        # ---- H1/H2 (pair_en) and sensitivity
        g = gap_block(G["correct"], W, unit_rows=en, unit="order")
        out["H1"][j] = g
        out["eng_context"][j] = {"pair_en": g["cells"]["ENG"]}
        sens = {"complete_case": gap_block(G["correct_cc"], W)["primary"],
                "invalid_half": gap_block(G["correct_half"], W)["primary"],
                "ifm_unflagged": gap_block(tuple(_mask(x, unflag) for x in G["correct"]), W)["primary"],
                "ifm_flagged_only": gap_block(tuple(_mask(x, ~unflag) for x in G["correct"]), W)["primary"],
                "ifm_flagged_keys": {"n_in_sample": int((~unflag).sum()), "source_available": flagged_ok,
                                     **({} if flagged_ok else {"note": ifm_note})},
                "mcnemar_order_level_stacked": g["primary"].get("mcnemar_order_level_stacked")}
        out["sensitivity"][j] = sens

        # ---- DR7 strata
        out["strata"][j] = strata(G["correct"], cands_same, W)

        # ---- H5 pointwise
        gp = gap_block(mpair(po, po, "correct", keys, "cand", ref="MSA"), W, unit_rows=po, unit="cand")
        out["H5"][j] = gp
        out["eng_context"][j]["point"] = gp["cells"]["ENG"]
        pg, pw_ = po[po.cand == "gold"], po[po.cand == "wrong"]
        Ag, Aw = kmat(pg, "accepted", keys), kmat(pw_, "accepted", keys)
        out["pointwise"][j] = {
            "accept_gold": {v: S.mean_summary(Ag[v], W) for v in C.VARIETIES},
            "accept_wrong": {v: S.mean_summary(Aw[v], W) for v in C.VARIETIES},
            "accept_wrong_gap": {v: S.paired_summary(Aw[v], Aw["MSA"], W, scale=False) for v in C.VARIETIES if v != "MSA"},
            "accept_gold_gap": {v: S.paired_summary(Ag[v], Ag["MSA"], W, scale=False) for v in C.VARIETIES if v != "MSA"},
            "complete_case": gap_block(mpair(po, po, "correct_cc", keys, "cand", ref="MSA"), W)["primary"],
            "invalid_half": gap_block(mpair(po, po, "correct_half", keys, "cand", ref="MSA"), W)["primary"]}

        # ---- H3 rubric language
        if ar.empty:
            out["H3"][j] = {"note": "no pair_ar rows (deferred or not yet run)"}
        else:
            Aar, Ben = mpair(ar, en, "correct", keys, "order")  # order-matched pair_ar vs pair_en
            a, b = paired_mean_over(Aar, Ben, C.VARIETIES)
            out["H3"][j] = {"primary": S.paired_summary(a, b, W),
                            "cells": {v: S.paired_summary(Aar[v], Ben[v], W) for v in C.VARIETIES}}

        # ---- H4 repetition (128-key subset, W128)
        out["H4"][j] = repetition(cond, en, vkeys, W128)

        # ---- DR5 competence
        sl = sv.groupby(["key", "variety"]).parsed.first().unstack().reindex(index=keys, columns=C.VARIETIES)
        out["competence"][j] = competence_analysis(
            M["correct"], Msv, sl, distr_letter, en[["key", "variety", "order", "correct"]], keys, W,
            C.VARIETIES, C.DIALECTS, en_matched=G["correct"])

        # ---- DR6 register
        out["register"][j] = register_analysis(dj[dj.condition.isin(REG_CONDS)], pairs, keys, W, C.DIALECTS)

        # ---- position (valid rows only)
        out["position"][j] = position(en, keys, W)

        # ---- stability, noise floor
        out["stability"][j] = stability(en, cond["pair_en_rep"], vkeys, W128)

        # ---- MDE per condition
        out["mde"][j] = mde_block(out, j, cond)

    # ---- Holm families (A1, DR3b)
    fam = {}
    fam["H1"] = S.holm([out["H1"][j]["primary"] for j in judges])
    S.holm([out["H1"][j]["primary"] for j in judges], p_field="p_less", out_field="p_holm_H2")
    fam["H1_secondary"] = S.holm([out["H1"][j]["cells"][v] for j in judges for v in C.DIALECTS])
    fam["H5"] = S.holm([out["H5"][j]["primary"] for j in judges])
    fam["H5_secondary"] = S.holm([out["H5"][j]["cells"][v] for j in judges for v in C.DIALECTS])
    h3 = [j for j in judges if "primary" in out["H3"][j]]
    fam["H3"] = S.holm([out["H3"][j]["primary"] for j in h3])
    fam["H3_secondary"] = S.holm([out["H3"][j]["cells"][v] for j in h3 for v in C.VARIETIES])
    h4 = [j for j in judges if "primary" in out["H4"][j]]
    S.holm([out["H4"][j]["primary"] for j in h4])
    S.holm([out["H4"][j]["primary"] for j in h4], p_field="p_less", out_field="p_holm_less")
    # DR5 headline decomposition: per-judge means (one test per judge) and judge x dialect cells, Holm within each.
    for part in ("solved_both", "did"):
        S.holm([out["competence"][j][part]["primary"] for j in judges])
        S.holm([out["competence"][j][part]["cells"][v] for j in judges for v in C.DIALECTS])
    for j in judges:
        for s in (out["H1"][j]["primary"], out["H5"][j]["primary"],
                  out["competence"][j]["solved_both"]["primary"], out["competence"][j]["did"]["primary"]):
            S.classify(s)
        if j in h3:
            S.classify(out["H3"][j]["primary"])
        if j in h4:
            # Class uses the two-sided p_holm (A1/DR3e: "7 tests, Holm", no direction pinned); the
            # directional "lowers accuracy" reading is p_holm_less. OPEN: pin which is confirmatory.
            S.classify(out["H4"][j]["primary"])
        out["strata"][j]["dialect_ranking"] = _ranking(out["strata"][j]["dialect_candidates"]["cells"])
    nj, nc = len(judges), len(judges) * len(C.DIALECTS)
    out["holm_families"] = {"H1": f"{nj} judges, two-sided, primary (field p_holm)",
                            "H2": f"{nj} judges, one-sided dialect<MSA (field p_holm_H2 on H1 primary)",
                            "H1_secondary": f"{nc} judge x dialect cells", "H5": f"{nj} judges", "H5_secondary": f"{nc} cells",
                            "H3": f"{len(h3)} judges with pair_ar", "H3_secondary": f"{7 * len(h3)} cells",
                            "H4": f"{len(h4)} judges; class uses p_holm (two-sided); p_holm_less = one-sided "
                                  "'repetition lowers accuracy' (directional reading, also Holm-adjusted)",
                            "competence": f"solved_both / did: {nj} judge means and {nc} cells, separately",
                            "outside": "ENG-MSA (eng_context), strata, register, position, stability"}
    out["drivers"] = drivers(df, judges)

    adir = (C.MOCK_RESULTS_DIR if mock else C.RESULTS_DIR) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "metrics.json").write_text(json.dumps(_clean(out), indent=1))
    write_tables(out, judges, adir / "tables")
    write_figures(out, judges, adir / "figures")
    print(json.dumps({"coverage_complete": cov["complete"],
                      "coverage_frac": {j: {c: round(x["frac"], 3) for c, x in v.items()} for j, v in cov["by_judge"].items()}},
                     indent=1))
    _warn_coverage(cov)
    return out


# ============================================================================ components
def invalid_rates(cond, keys, W, W128, vkeys):
    """Invalid share per condition x variety (rows present only), with a key-level sign-flip test of
    the dialect - MSA difference where the condition has an MSA arm."""
    out = {}
    for c, r in cond.items():
        if r.empty:
            continue
        sub = c in SUBSET_CONDS
        ks, w = (vkeys, W128) if sub else (keys, W)
        I = kmat(r.assign(inv=(~r.valid).astype(float)), "inv", ks)
        out[c] = {}
        for v in C.VARIETIES:
            n = int((r.variety == v).sum())
            if n == 0:
                continue
            e = {"rate": float(1 - r[r.variety == v].valid.mean()), "n_rows": n}
            if v != "MSA" and "MSA" in set(r.variety):
                e["p_vs_msa"] = S.signflip(I[v] - I["MSA"])
            out[c][v] = e
    return out


def strata(AB, cands_same, W):
    """DR7: gaps vs MSA separately for keys whose both candidates are identical to MSA
    ("question_only") and the rest ("dialect_candidates"); dialect mean per stratum.
    ``AB`` = order-matched (A, B) key matrices vs MSA (see :func:`kmatched`)."""
    A, Bm = AB
    out = {}
    for name, mask in (("question_only", cands_same), ("dialect_candidates", ~cands_same)):
        cells = {v: S.paired_summary(A[v].where(mask[v].to_numpy()), Bm[v].where(mask[v].to_numpy()), W)
                 for v in C.DIALECTS}
        a, b = paired_mean_over(A, Bm, C.DIALECTS, mask={v: mask[v].to_numpy() for v in C.DIALECTS})
        out[name] = {"cells": cells, "primary": S.paired_summary(a, b, W)}
    return out


def _ranking(cells):
    """Dialects ordered from largest drop (most negative gap) to smallest; dialect-candidate stratum."""
    return [v for v, _ in sorted(((v, s["est"]) for v, s in cells.items() if s.get("est") is not None),
                                 key=lambda t: t[1])]


def repetition(cond, en, vkeys, W128):
    """H4 on the 128-key subset: acc(verbose) - acc(pair_en) averaged over the 7 varieties per key
    (negative = repeating the distractor lowers accuracy); verbose_gold mirror; and the signed
    repetition bias 0.5 * (acc(verbose_gold) - acc(verbose)) (positive = repetition favours the
    repeated candidate). All differences are order-matched (DR3a).

    The class uses the two-sided Holm p (``p_holm``); the directional reading of H4 ("repetition
    lowers accuracy") is the one-sided ``p_holm_less``. Which is confirmatory must be pinned in the
    spec before freezing (a one-sided class would call an opposite-sign effect "inconclusive")."""
    vb, vg = cond["verbose"], cond["verbose_gold"]
    if vb.empty:
        return {"note": "no verbose rows"}
    Ob = omat(en[en.key.isin(set(vkeys))], "correct", "order")
    Ovb, Ovg = omat(vb, "correct", "order"), omat(vg, "correct", "order")
    Avb, Bb = kmatched(Ovb, Ob, vkeys)
    a, b = paired_mean_over(Avb, Bb, C.VARIETIES)
    out = {"primary": S.paired_summary(a, b, W128),
           "cells": {v: S.paired_summary(Avb[v], Bb[v], W128) for v in C.VARIETIES}}
    if not vg.empty:
        Avg, Bg = kmatched(Ovg, Ob, vkeys)
        a2, b2 = paired_mean_over(Avg, Bg, C.VARIETIES)
        out["gold_repeated"] = {"primary": S.paired_summary(a2, b2, W128),
                                "cells": {v: S.paired_summary(Avg[v], Bg[v], W128) for v in C.VARIETIES}}
        g, w = paired_mean_over(*kmatched(Ovg, Ovb, vkeys), C.VARIETIES)
        out["repetition_bias"] = S.diff_summary(0.5 * (g - w), W128)
        # Cross-variety part of H4 is exploratory: dialect-mean minus MSA difference of the effect,
        # with the effect formed per order (verbose_o - pair_en_o) and then matched to MSA by order.
        Odv = Ovb.reindex(Ovb.index.union(Ob.index)) - Ob.reindex(Ovb.index.union(Ob.index))
        a3, b3 = paired_mean_over(*kmatched(Odv, Odv, vkeys, ref="MSA"), C.DIALECTS)
        out["effect_dialect_minus_msa"] = S.paired_summary(a3, b3, W128, scale=False)
    return out


def position(en, keys, W):
    """Position bias on valid rows: P(choose A) - 0.5 and order inconsistency (same slot chosen in
    both orders, among (key, variety) with both orders valid), per variety and overall."""
    v_ = en[en.valid]
    PA = kmat(v_, "chose_A", keys)
    w = v_.pivot_table(index=["key", "variety"], columns="order", values="correct", aggfunc="first")
    if {"good_first", "good_second"} <= set(w.columns):
        inc = (w["good_first"] != w["good_second"]).astype(float).where(w.notna().all(1))
        I = inc.unstack().reindex(index=keys, columns=C.VARIETIES)
    else:
        I = pd.DataFrame(np.nan, index=keys, columns=C.VARIETIES)
    out = {v: {"pA_minus_half": S.mean_summary(PA[v] - 0.5, W),
               "inconsistency": S.mean_summary(I[v], W)} for v in C.VARIETIES}
    out["overall"] = {"pA_minus_half": S.mean_summary(S.nanmean_rows(PA.to_numpy()) - 0.5, W),
                      "inconsistency": S.mean_summary(S.nanmean_rows(I.to_numpy()), W)}
    return out


def stability(en, rep, vkeys, W128):
    """Run-to-run stability (pair_en vs pair_en_rep, identical prompts) on the 128-key subset.

    Agreement = same parsed label (invalid counts as its own label). Per key: mean over orders;
    overall: mean over varieties. noise_floor = 1 - stability. Also the MSA->dialect flip rate in
    pair_en on the same keys and its excess over the noise floor (dialect flips beyond run-to-run
    flips; paired by key, W128).
    """
    if rep.empty:
        return {"note": "no pair_en_rep rows"}
    e = en[en.key.isin(vkeys)].assign(lab=lambda x: x.parsed.fillna("INVALID"))
    r = rep.assign(lab=lambda x: x.parsed.fillna("INVALID"))
    m = e[["key", "variety", "order", "lab"]].merge(r[["key", "variety", "order", "lab"]],
                                                     on=["key", "variety", "order"], suffixes=("", "_rep"))
    A = kmat(m.assign(agree=(m.lab == m.lab_rep).astype(float)), "agree", vkeys)
    overall = S.nanmean_rows(A.to_numpy())
    st = S.mean_summary(overall, W128)
    out = {"by_variety": {v: S.mean_summary(A[v], W128) for v in C.VARIETIES}, "overall": st,
           "noise_floor": {"est": None if st["est"] is None else 1 - st["est"],
                           "ci95": [None if x is None else 1 - x for x in st["ci95"][::-1]]}}
    ms = e[e.variety == "MSA"][["key", "order", "lab"]]
    f = e[e.variety.isin(C.DIALECTS)].merge(ms, on=["key", "order"], suffixes=("", "_msa"))
    F = kmat(f.assign(flip=(f.lab != f.lab_msa).astype(float)), "flip", vkeys, cols=C.DIALECTS)
    flip = S.nanmean_rows(F.to_numpy())
    out["dialect_flip_rate"] = S.mean_summary(flip, W128)
    out["flip_excess_over_noise"] = S.paired_summary(flip, 1 - overall, W128, scale=False)
    return out


def mde_block(out, j, cond):
    """DR3f: MDE (alpha .05 two-sided, 80% power; and at the Holm-worst alpha .05/7) per condition's
    primary test, with the order-level discordance rate and the design effect of the key unit."""
    def entry(s, disc=None):
        if s is None or s.get("n_keys", 0) == 0:
            return None
        e = {"n_keys": s["n_keys"], "sd_d": s.get("sd_d"), "mde80": S.mde(s.get("sd_d"), s["n_keys"]),
             "mde80_alpha_over_7": S.mde(s.get("sd_d"), s["n_keys"], alpha=0.05 / 7)}
        if disc:
            e.update(disc)
        return e
    en, po, ar, vb = cond["pair_en"], cond["point"], cond["pair_ar"], cond["verbose"]
    msa = lambda r: r[r.variety == "MSA"].drop(columns="variety")  # noqa: E731
    dia = lambda r: r[r.variety.isin(C.DIALECTS)]  # noqa: E731
    res = {"pair_en_dialect_gap": entry(out["H1"][j]["primary"], _discordance(dia(en), msa(en), "order", on=())),
           "point_dialect_gap": entry(out["H5"][j]["primary"], _discordance(dia(po), msa(po), "cand", on=())),
           "pair_en_cell_median": entry({"n_keys": out["H1"][j]["cells"]["UAE"]["n_keys"],
                                         "sd_d": float(np.nanmedian([out["H1"][j]["cells"][v]["sd_d"] or np.nan for v in C.DIALECTS]))}),
           "did_dialect": entry(out["competence"][j]["did"]["primary"])}
    if "primary" in out["H3"][j]:
        res["rubric_lang"] = entry(out["H3"][j]["primary"], _discordance(ar, en, "order"))
    if "primary" in out["H4"][j]:
        res["repetition"] = entry(out["H4"][j]["primary"], _discordance(vb, en[en.key.isin(set(vb.key))], "order"))
    pm = out["register"][j]["pooled"].get("main")
    if pm:
        res["reg_forced_vs_half"] = entry(pm["forced_msa"])
    return res


# ============================================================================ drivers (exploratory)
def drivers(df, judges):
    """Exploratory: do dialect errors concentrate on items that drift further from MSA?

    Unit: (judge, key, dialect, order) in pair_en where the same judge was correct on the MSA
    version with the same order. Outcome: wrong on the dialect version ("flip"). Two binomial GEEs
    (exchangeable, clustered by key): (1) z-scored word Jaccard of the dialect question vs MSA,
    (2) z-scored character similarity q_char_sim (DR7 dose-response); both with gold-string-identical
    -to-MSA, both-candidates-identical-to-MSA, variety and judge. Flip rates by tercile of each
    overlap measure, and by q_char_sim tercile within the both-candidates-identical stratum.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    feat = pd.read_parquet(C.DATA_DIR / "item_features.parquet")
    en = df[(df.condition == "pair_en") & df.judge.isin(judges)]
    msa = en[en.variety == "MSA"][["judge", "key", "order", "correct"]].rename(columns={"correct": "msa_ok"})
    d = en[en.variety.isin(C.DIALECTS)].merge(msa, on=["judge", "key", "order"]).merge(feat, on=["key", "variety"])
    d = d[d.msa_ok == 1].assign(flip=lambda x: 1 - x.correct)
    if d.empty or d.flip.nunique() < 2:
        return {"n": int(len(d)), "note": "no variation in flips"}
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    d = d.assign(jacc_z=z(d.q_word_jacc), char_z=z(d.q_char_sim), gold_same=d.gold_same_as_msa.astype(int),
                 cands_same=(d.gold_same_as_msa & d.distr_same_as_msa).astype(int))
    res = {"n": int(len(d))}
    for name, col in (("overlap", "q_word_jacc"), ("char_sim", "q_char_sim")):
        t = pd.qcut(d[col].rank(method="first"), 3, labels=["low", "mid", "high"])
        res[f"flip_rate_by_{name}_tercile"] = d.groupby(t, observed=True).flip.mean().to_dict()
    qs = d[d.cands_same == 1]
    if len(qs) > 30:
        t = pd.qcut(qs.q_char_sim.rank(method="first"), 3, labels=["low", "mid", "high"])
        res["flip_rate_by_char_sim_tercile_question_only"] = qs.groupby(t, observed=True).flip.mean().to_dict()
    for name, x in (("gee", "jacc_z"), ("gee_char", "char_z")):
        try:
            m = smf.gee(f"flip ~ {x} + gold_same + cands_same + variety + judge", groups="key", data=d,
                        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = m.conf_int()
            res[name] = {k: {"or": float(np.exp(m.params[k])), "lo": float(np.exp(ci.loc[k, 0])),
                             "hi": float(np.exp(ci.loc[k, 1])), "p": float(m.pvalues[k])}
                         for k in (x, "gold_same", "cands_same")}
        except Exception as e:  # report rather than hide a failed fit
            res[f"{name}_error"] = f"{type(e).__name__}: {e}"
    return res


def _clean(o):
    """JSON-safe copy: numpy scalars -> Python, NaN/inf -> None, tuple -> list, non-str keys -> str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o
