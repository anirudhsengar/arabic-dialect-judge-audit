"""Planted-effects simulator: writes mock judge outputs with KNOWN ground truth to results_mock/.

Independent of ajd/analyze.py by design (written from experiment_spec.md, amendment A1). It uses only
ajd.conditions.build_requests and ajd.data.load_sample, so request IDs match the real run.

Usage (from experiment/):
    uv run --project .. python tests/simulate.py [--defer-opus-pair-ar]

Outputs (only ever in results_mock/):
    results_mock/<judge>.jsonl         rows {rid, judge, model, params, raw, parsed, meta, ts}
    results_mock/<judge>.errors.jsonl  transport errors for the ~0.5% of requests left missing
    results_mock/ground_truth.json     planted parameters + realised values from the simulated rows

Randomness: every draw is a hash of (SEED, judge, ...), so the output does not depend on loop order.

Generative model (pairwise conditions: pair_en, pair_ar, pair_en_rep, verbose, verbose_gold)
  Per (judge, key, variety, rubric group) a latent state is drawn and shared by both orders:
    know     -> picks the good candidate in both orders
    confused -> picks the bad candidate in both orders
    unsure   -> each order independently picks slot A with prob pA (0.5 unless position bias)
  pair_en_rep shares pair_en's state (only the unsure coin flips are redrawn).
  Latent uniforms are shared across varieties of a key with prob rho (correlated difficulty).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajd import config as C  # noqa: E402
from ajd.conditions import build_requests  # noqa: E402
from ajd.data import load_sample  # noqa: E402

SEED = 20261024
OUT = C.MOCK_RESULTS_DIR
assert OUT.name == "results_mock" and OUT != C.RESULTS_DIR
LETTERS = "ABCD"
INVALID_RATE = 0.01
MISSING_RATE = 0.005
ALT_FORMAT_RATE = 0.02  # valid but non-canonical text (reasoning prefix, "Candidate A"), tests the parser
PAIR_CONDS = {"pair_en", "pair_ar", "pair_en_rep", "verbose", "verbose_gold"}
REG_CONDS = {"reg_tie", "reg_forced", "reg_forced_msaq"}
V = C.VARIETIES


# ---------------------------------------------------------------- deterministic uniforms
def U(*parts) -> float:
    h = hashlib.blake2b(json.dumps((SEED,) + parts, ensure_ascii=False).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def coupled(rho, *parts):
    """Uniform shared across varieties of a key with prob rho; parts[-1] must be the variety."""
    if U(*parts, "mix") < rho:
        return U(*parts[:-1], "shared")
    return U(*parts, "own")


# ---------------------------------------------------------------- independent normaliser (DR6)
_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks, tatweel
# Hamza seats ؤ/ئ are NOT folded: DR6's quoted counts (KSA 2, Σ279) do not fold them, and folding is a
# substantive (dialectal) choice rather than orthography. Whitespace is removed entirely (DR6).
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})


def norm(s: str) -> str:
    s = _DIAC.sub("", unicodedata.normalize("NFKC", s)).translate(_MAP)
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P") and not ch.isspace()).casefold()


# ---------------------------------------------------------------- judge profiles
def flat(x):
    return {v: x for v in V}


# solve: P(solved) per variety, rho_s = cross-variety coupling of the solve latent.
# pair: function (judge, key, variety, solved) -> (p_know, p_confused); rho_p coupling of pair latent.
# pA: P(choose A | unsure). rep: repetition penalty (pp) on the repeated candidate.
# point: P(accept gold), P(accept distractor) per variety.
# reg: per (substantive/placebo) P(TIE) in reg_tie; P(MSA | non-tie) for reg_tie / reg_forced / reg_forced_msaq.
def _pair_const(know, confused=0.0):
    return lambda v, solved: (know, confused)


REG_NEUTRAL = {"tie_sub": 0.40, "tie_placebo": 0.90, "msa_tie": 0.5, "msa_forced": 0.5, "msa_msaq": 0.5,
               "msa_forced_placebo": 0.5}

PROFILES = {
    # No dialect effect anywhere: every variety identical, ~85% accuracy in every condition.
    "haiku-5.5": dict(solve=flat(0.85), rho_s=0.85, pair=_pair_const(0.70), rho_p=0.85, pA=0.5, rep=0.0,
                      point_gold=flat(0.85), point_wrong=flat(0.15), reg=REG_NEUTRAL),
    # Competence-inherited gap: MAG fails solve on 15% more keys (nested: MSA-unsolved ⊂ MAG-unsolved);
    # pairwise is correct iff solved (errors only on unsolved keys) -> raw gap -15pp, solved-both gap 0, DiD 0.
    "opus-5.5": dict(solve={**flat(0.90), "MAG": 0.75}, rho_s=1.0,
                     pair=lambda v, solved: (1.0, 0.0) if solved else (0.0, 1.0), rho_p=1.0, pA=0.5, rep=0.0,
                     point_gold=flat(0.92), point_wrong=flat(0.08), reg=REG_NEUTRAL),
    # Judging-specific gap: EGY pairwise 8pp lower than MSA on every solve stratum, solve unaffected.
    # acc = know + 0.5*unsure; solved: MSA .95 / EGY .87, unsolved: MSA .65 / EGY .57 -> DiD = -8pp.
    "deepseek-flash": dict(solve=flat(0.85), rho_s=0.85,
                           pair=lambda v, solved: ((0.90 if solved else 0.30) - (0.16 if v == "EGY" else 0.0), 0.0),
                           rho_p=1.0, pA=0.5, rep=0.0,
                           point_gold=flat(0.88), point_wrong=flat(0.12), reg=REG_NEUTRAL),
    # Position bias: P(A | unsure) = 0.70, unsure = 60% -> P(A) - 0.5 = +0.12; repetition penalty 20pp.
    "gemini-3.8-flash": dict(solve=flat(0.80), rho_s=0.85, pair=_pair_const(0.40), rho_p=0.85, pA=0.70, rep=0.20,
                             point_gold=flat(0.85), point_wrong=flat(0.15), reg=REG_NEUTRAL),
    # Register + leniency (see ground_truth.json["gpt-6.1-sol"]).
    "gpt-6.1-sol": dict(solve=flat(0.88), rho_s=0.85, pair=_pair_const(0.76), rho_p=0.85, pA=0.5, rep=0.0,
                      point_gold=flat(0.90), point_wrong={**flat(0.05), "SYR": 0.25},
                      reg={"tie_sub": 0.30, "tie_placebo": 0.99, "msa_tie": 0.75, "msa_forced": 0.75,
                           "msa_msaq": 0.50, "msa_forced_placebo": 0.5}),
    # Noise judges: no planted effect.
    "gpt-6-luna": dict(solve=flat(0.82), rho_s=0.8, pair=_pair_const(0.64), rho_p=0.8, pA=0.5, rep=0.0,
                       point_gold=flat(0.82), point_wrong=flat(0.18), reg=REG_NEUTRAL),
    "gemini-4-argon": dict(solve=flat(0.90), rho_s=0.8, pair=_pair_const(0.80), rho_p=0.8, pA=0.5, rep=0.0,
                           point_gold=flat(0.90), point_wrong=flat(0.10), reg=REG_NEUTRAL),
    "deepseek-v4-pro": dict(solve=flat(0.86), rho_s=0.8, pair=_pair_const(0.72), rho_p=0.8, pA=0.5, rep=0.0,
                            point_gold=flat(0.86), point_wrong=flat(0.14), reg=REG_NEUTRAL),
}
assert set(PROFILES) == set(C.MODELS), set(C.MODELS) ^ set(PROFILES)


# ---------------------------------------------------------------- simulation
def solved(j, key, v):
    p = PROFILES[j]
    return coupled(p["rho_s"], j, "solve", key, v) < p["solve"][v]


def pair_state(j, grp, key, v):
    p = PROFILES[j]
    know, conf = p["pair"](v, solved(j, key, v))
    u = coupled(p["rho_p"], j, grp, key, v)
    return "know" if u < know else ("confused" if u < know + conf else "unsure")


def simulate_one(j, q, by, placebo):
    """Returns the canonical label the judge outputs (A/B/C/D/TIE/correct/incorrect)."""
    p, cond, key, v = PROFILES[j], q["condition"], q["key"], q["variety"]
    r = by[(key, v)]
    if cond == "solve":
        if solved(j, key, v):
            return LETTERS[r.answer]
        others = [i for i in range(4) if i not in (r.answer, r.distractor)]
        i = r.distractor if U(j, "solve-wrong", key, v) < 0.5 else others[int(U(j, "solve-other", key, v) * 2)]
        return LETTERS[i]
    if cond == "point":
        acc = p["point_gold"][v] if q["cand"] == "gold" else p["point_wrong"][v]
        u = coupled(0.8, j, "point", q["cand"], key, v)
        return "correct" if u < acc else "incorrect"
    good, bad = q["good_slot"], ("B" if q["good_slot"] == "A" else "A")
    if cond in PAIR_CONDS:
        grp = "pair_ar" if cond == "pair_ar" else "pair_en"
        st = pair_state(j, grp, key, v)
        if st == "know":
            choice = good
        elif st == "confused":
            choice = bad
        else:
            # coin coupled across varieties like the latent state (rep re-draws it: run-to-run noise)
            choice = "A" if coupled(p["rho_p"], j, "unsure", cond, key, q["order"], v) < p["pA"] else "B"
        if cond in ("verbose", "verbose_gold") and p["rep"]:
            # Repetition penalty: P(choose repeated candidate) drops by exactly rep (in expectation).
            repeated = good if cond == "verbose_gold" else bad
            know, conf = p["pair"](v, solved(j, key, v))
            # order-averaged base P(choose repeated); switching with rep/p0 in both orders drops it by rep
            p0 = (know if repeated == good else conf) + (1 - know - conf) * 0.5
            if choice == repeated and U(j, "rep", cond, key, v, q["order"]) < p["rep"] / p0:
                choice = bad if repeated == good else good
        return choice
    # Register conditions: "good" slot = dialect-register gold, the other slot = MSA gold.
    reg, is_plc = p["reg"], placebo[(key, v)]
    msa_slot, dia_slot = bad, good
    if cond == "reg_tie":
        if U(j, "tie", key, v, q["order"]) < (reg["tie_placebo"] if is_plc else reg["tie_sub"]):
            return "TIE"
        pm = 0.5 if is_plc else reg["msa_tie"]
    elif cond == "reg_forced":
        pm = reg["msa_forced_placebo"] if is_plc else reg["msa_forced"]
    else:  # reg_forced_msaq
        pm = reg["msa_forced_placebo"] if is_plc else reg["msa_msaq"]
    u = coupled(0.7, j, cond, key, v, q["order"])  # key-level preference shared across orders w.p. 0.7
    return msa_slot if u < pm else dia_slot


def render(j, q, label):
    """Raw text: canonical JSON, occasionally a valid non-canonical form, ~1% invalid."""
    field = {"solve": "answer", "point": "verdict"}.get(q["condition"], "winner")
    u = U(j, "fmt", q["rid"])
    if u < INVALID_RATE:
        bad = ['{"%s": ' % field, "I cannot determine this.", '{"%s": "maybe"}' % field, "", "```json\n{"]
        return bad[int(U(j, "badkind", q["rid"]) * len(bad))]
    if u < INVALID_RATE + ALT_FORMAT_RATE:
        if field == "winner" and label in "AB" and len(label) == 1:
            return 'Comparing both candidates. {"winner": "Candidate %s"}' % label
        return 'Reasoning first. {"%s": "%s"}' % (field, label)
    return '{"%s": "%s"}' % (field, label)


_PARSE = re.compile(r'"(?:winner|answer|verdict)"\s*:\s*"(?:candidate\s+)?(A|B|C|D|TIE|correct|incorrect)"', re.I)


def parse(text):
    found = _PARSE.findall(text or "")
    return found[-1].upper() if found else None


def is_correct(q, parsed):
    """Pre-specified scoring: invalid = wrong. None for register conditions (no correct answer)."""
    c = q["condition"]
    if c in REG_CONDS:
        return None
    if parsed is None:
        return 0
    if c == "solve":
        return int(parsed == q["gold_letter"])
    if c == "point":
        return int(parsed == ("CORRECT" if q["cand"] == "gold" else "INCORRECT"))
    return int(parsed == q["good_slot"])


# ---------------------------------------------------------------- realised ground truth
def realised(rows_by_judge, reqs_by_rid, placebo, verbose_keys):
    import pandas as pd
    recs = []
    for j, rows in rows_by_judge.items():
        for row in rows:
            q = reqs_by_rid[row["rid"]]
            recs.append({"judge": j, "condition": q["condition"], "key": q["key"], "variety": q["variety"],
                         "order": q.get("order"), "cand": q.get("cand"), "good_slot": q.get("good_slot"),
                         "parsed": row["parsed"], "correct": is_correct(q, row["parsed"])})
    df = pd.DataFrame(recs)
    out = {}
    for j in rows_by_judge:
        d = df[df.judge == j]
        acc = {c: {v: round(float(g.correct.mean()), 4) for v, g in dc.groupby("variety")}
               for c, dc in d[~d.condition.isin(REG_CONDS)].groupby("condition")}
        # per-key accuracies (mean over orders / candidates)
        pk = d[~d.condition.isin(REG_CONDS)].groupby(["condition", "variety", "key"]).correct.mean()
        J = {"accuracy_invalid_as_wrong": acc,
             "invalid_rate": round(float(d.parsed.isna().mean()), 4), "n_rows": int(len(d))}
        dialect_gap = {}
        for c in ("pair_en", "pair_ar", "point"):
            if c not in pk.index.get_level_values(0):
                continue
            m = pk[c]["MSA"]
            dialect_gap[c] = {v: round(float((pk[c][v] - m).dropna().mean()), 4) for v in C.DIALECTS + ["ENG"]}
            dialect_gap[c]["mean_5_dialects"] = round(float(np.mean([dialect_gap[c][v] for v in C.DIALECTS])), 4)
        J["gap_vs_msa_keylevel"] = dialect_gap
        # DR5: solved-in-both gap and DiD (pair_en)
        s = pk["solve"]
        did, solved_both = {}, {}
        for v in C.DIALECTS + ["ENG"]:
            keys = s[v].index.intersection(s["MSA"].index).intersection(pk["pair_en"][v].index) \
                .intersection(pk["pair_en"]["MSA"].index)
            sv, sm = s[v][keys], s["MSA"][keys]
            pv, pm = pk["pair_en"][v][keys], pk["pair_en"]["MSA"][keys]
            both = (sv == 1) & (sm == 1)
            solved_both[v] = round(float((pv - pm)[both].mean()), 4)
            did[v] = round(float(((pv - pm) - (sv - sm)).mean()), 4)
        J["pair_en_gap_solved_both"] = solved_both
        J["pair_en_did"] = did
        # position bias & inconsistency (valid rows only)
        pe = d[(d.condition == "pair_en") & d.parsed.isin(["A", "B"])]
        J["pair_en_P_A_minus_half"] = round(float((pe.parsed == "A").mean() - 0.5), 4)
        wide = pe.pivot_table(index=["key", "variety"], columns="order", values="parsed", aggfunc="first").dropna()
        J["pair_en_order_inconsistency"] = round(float((wide["good_first"] == wide["good_second"]).mean()), 4)
        # repetition (128-key subset, averaged over varieties)
        sub = d[d.key.isin(verbose_keys)]
        base = sub[sub.condition == "pair_en"].correct.mean()
        J["repetition"] = {"pair_en_subset_acc": round(float(base), 4),
                           "verbose_minus_pair_en": round(float(sub[sub.condition == "verbose"].correct.mean() - base), 4),
                           "verbose_gold_minus_pair_en": round(float(sub[sub.condition == "verbose_gold"].correct.mean() - base), 4)}
        # stability
        a = d[d.condition == "pair_en"].set_index(["key", "variety", "order"]).parsed
        b = d[d.condition == "pair_en_rep"].set_index(["key", "variety", "order"]).parsed
        ix = a.index.intersection(b.index)
        J["stability_pair_en_rep_agreement"] = round(float((a[ix] == b[ix]).mean()), 4)
        # pointwise acceptance
        pt = d[d.condition == "point"]
        J["point_accept_rate"] = {cand: {v: round(float((g.parsed == "CORRECT").mean()), 4)
                                         for v, g in pc.groupby("variety")} for cand, pc in pt.groupby("cand")}
        # register (valid rows)
        reg = {}
        for c in sorted(REG_CONDS):
            dc = d[(d.condition == c) & d.parsed.notna()].copy()
            dc["placebo"] = [placebo[(k, v)] for k, v in zip(dc.key, dc.variety)]
            dc["msa"] = (dc.parsed != dc.good_slot) & (dc.parsed != "TIE")
            dc["tie"] = dc.parsed == "TIE"
            res = {}
            for name, g in (("substantive", dc[~dc.placebo]), ("placebo", dc[dc.placebo])):
                nt = g[~g.tie]
                res[name] = {"n_rows": int(len(g)), "tie_rate": round(float(g.tie.mean()), 4),
                             "P_msa_given_nontie": round(float(nt.msa.mean()), 4),
                             "P_msa_given_nontie_by_dialect": {v: round(float(x.msa.mean()), 4)
                                                               for v, x in nt.groupby("variety")}}
            reg[c] = res
        J["register"] = reg
        out[j] = J
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defer-opus-pair-ar", action="store_true",
                    help="omit opus-5.5 pair_ar entirely (spec A1 defers it); off by default")
    args = ap.parse_args()

    meta, items = load_sample()
    reqs = build_requests(meta, items)
    by = {(r.key, r.dialect): r for r in items.itertuples()}
    placebo = {}
    for key in meta["keys"]:
        m = by[(key, "MSA")]
        for d in C.DIALECTS:
            r = by[(key, d)]
            placebo[(key, d)] = norm(r.choices[r.answer]) == norm(m.choices[m.answer])
    reg_pairs = {(q["key"], q["variety"]) for q in reqs if q["condition"] == "reg_forced"}
    plc_counts = {d: sum(placebo[(k, v)] for k, v in reg_pairs if v == d) for d in C.DIALECTS}
    pair_counts = {d: sum(1 for k, v in reg_pairs if v == d) for d in C.DIALECTS}

    OUT.mkdir(exist_ok=True)
    for f in list(OUT.glob("*.jsonl")):
        f.unlink()
    if (OUT / "analysis").exists():
        shutil.rmtree(OUT / "analysis")

    rows_by_judge, missing = {}, {}
    for j, cfg in C.MODELS.items():
        rows, errs = [], []
        for q in reqs:
            if args.defer_opus_pair_ar and j == "opus-5.5" and q["condition"] == "pair_ar":
                continue
            if U(j, "missing", q["rid"]) < MISSING_RATE:
                errs.append({"rid": q["rid"], "error": "APIStatusError: 529 overloaded (simulated)"})
                continue
            label = simulate_one(j, q, by, placebo)
            raw = render(j, q, label)
            rows.append({"rid": q["rid"], "judge": j, "model": cfg["model"], "params": cfg["params"],
                         "raw": raw, "parsed": parse(raw),
                         "meta": {"served_model": cfg["model"], "usage": None, "simulated": True},
                         "ts": "2026-10-24T00:00:00+00:00"})
        random.Random(f"{SEED}:{j}").shuffle(rows)  # real files are in completion order, not request order
        with (OUT / f"{j}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (OUT / f"{j}.errors.jsonl").open("w", encoding="utf-8") as f:
            for e in errs:
                f.write(json.dumps(e) + "\n")
        rows_by_judge[j] = rows
        missing[j] = len(errs)
        print(f"{j:18s} rows={len(rows)} missing={len(errs)} invalid={sum(r['parsed'] is None for r in rows)}")

    reqs_by_rid = {q["rid"]: q for q in reqs}
    gt = {
        "seed": SEED, "n_requests_per_judge": len(reqs),
        "invalid_rate_planted": INVALID_RATE, "missing_rate_planted": MISSING_RATE,
        "missing_per_judge": missing, "opus_pair_ar_deferred": args.defer_opus_pair_ar,
        "scoring": "invalid = wrong; missing rows excluded (transport errors logged in <judge>.errors.jsonl)",
        "placebo": {"rule": "independent normaliser: NFKC, strip harakat/Quranic marks/tatweel, alif/hamza "
                            "variants->ا, ة->ه, ى->ي, , drop Unicode punctuation and whitespace, casefold",
                    "orthography_only_pairs": plc_counts, "total_register_pairs": pair_counts,
                    "orthography_only_keys": sorted(f"{k}|{v}" for (k, v) in reg_pairs if placebo[(k, v)])},
        "planted": {
            "haiku-5.5": {"effect": "none", "all_varieties_accuracy": 0.85,
                          "expect": "primary pair_en/point dialect gaps equivalent (90% CI within ±2pp), not significant"},
            "opus-5.5": {"effect": "competence-inherited gap in MAG",
                         "solve": {"MAG": 0.75, "other": 0.90},
                         "mechanism": "pairwise correct iff solved; solve latent nested so every MSA-unsolved key is "
                                      "also MAG-unsolved",
                         "expect": {"pair_en_gap_MAG": -0.15, "pair_en_gap_solved_both_MAG": 0.0,
                                    "pair_en_did_MAG": 0.0, "other_dialects_gap": 0.0}},
            "deepseek-flash": {"effect": "judging-specific gap in EGY", "solve": 0.85,
                               "pair_en_acc": {"solved": {"MSA": 0.95, "EGY": 0.87}, "unsolved": {"MSA": 0.65, "EGY": 0.57}},
                               "expect": {"pair_en_gap_solved_both_EGY": -0.08, "pair_en_did_EGY": -0.08,
                                          "solve_gap_EGY": 0.0, "other_dialects_gap": 0.0}},
            "gemini-3.8-flash": {"effect": "position bias + repetition penalty",
                                 "P_unsure": 0.60, "P_A_given_unsure": 0.70,
                                 "expect": {"P_A_minus_half": 0.12, "order_inconsistency": 0.60 * (0.7**2 + 0.3**2),
                                            "verbose_minus_pair_en": 0.20, "verbose_gold_minus_pair_en": -0.20,
                                            "pair_en_acc": 0.70, "no_dialect_gap": True},
                                 "note": "position bias planted in pairwise-correctness conditions only, not in reg_*"},
            "gpt-6.1-sol": {"effect": "register preference + leniency",
                          "register": {"reg_forced_P_msa_substantive": 0.75, "reg_forced_msaq_P_msa_substantive": 0.50,
                                       "reg_tie_tie_rate_substantive": 0.30, "reg_tie_P_msa_given_nontie_substantive": 0.75,
                                       "placebo_tie_rate": 0.99, "placebo_P_msa_forced": 0.50},
                          "register_pattern": "MSA PREFERENCE that appears only when the question is in dialect: "
                                              "dialect-question 0.75 vs MSA-question 0.50. Pure register matching would "
                                              "be the opposite (dialect question -> P(MSA) < 0.5; MSA question -> > 0.5). "
                                              "So reg_forced - reg_forced_msaq = +0.25 toward MSA (anti-matching).",
                          "point_accept_distractor": {"SYR": 0.25, "other": 0.05}, "point_accept_gold": 0.90,
                          "expect": {"point_gap_SYR": -0.10, "point_gap_other": 0.0}},
            "gpt-6-luna": {"effect": "none (noise)"}, "gemini-4-argon": {"effect": "none (noise)"},
        },
        "realised": realised(rows_by_judge, reqs_by_rid, placebo, set(meta["verbose_keys"])),
    }
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=1, ensure_ascii=False))
    print(f"orthography-only register pairs: {plc_counts}; wrote {OUT/'ground_truth.json'}")


if __name__ == "__main__":
    main()
