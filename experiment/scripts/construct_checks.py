"""Construct checks for the judging-vs-competence analysis (post hoc). Reads only stored outputs.

Inputs: results/<judge>.jsonl (via ajd.analyze.load, read-only), data/sample_items.parquet.
No API calls. All 8 judges included (GPT-6.1 Sol complete). Output: results/analysis/construct_checks.json.

(1) Chance-corrected difference-in-differences. Under know-or-guess, a comprehension loss dk lowers
    4-option solve accuracy by 0.75*dk and 2-option pair accuracy by 0.5*dk, so the expected raw
    pairwise gap is 2/3 of the solve gap. Per key and dialect v we rescale both gaps to the
    "knowledge" scale: pair gap / 0.5 (from (acc-.5)/.5) and solve gap / 0.75 (from (acc-.25)/.75),
    and take DiD_cc = 2*(pair_v - pair_MSA) - (4/3)*(solve_v - solve_MSA), mean over the five
    dialects (order-matched pair side as in metrics.json competence.did). Same bootstrap weights
    (domain-stratified, B=2000, seed 2026) and sign-flip test as metrics.json. Negative = judging
    degrades MORE than comprehension (as measured by solving) predicts. Holm over the 8 judges
    (field p_holm), sign-flip p_two.
(2) Lucky-guess artefact in SB (solved-both residual).
    (a) Analytic know-or-guess bound: k_v = (s_v - .25)/.75; among keys solved in v, P(known) = k_v/s_v;
        an unknown key is judged at 0.5, a known key at most 1.0. With no extra judging cost the
        expected SB residual is at most 0.5*(k_v/s_v - k_MSA/s_MSA) (in pp, mean over dialects).
        This is an upper bound on the artefact under this model (known keys judged perfectly).
        SB minus this bound is bootstrapped with the same weights (bound recomputed per replicate).
    (b) Stricter competence proxies: SB recomputed on keys solved in ENG, MSA and the dialect, and on
        keys solved in all seven varieties.
(3) Option-referential candidates. Keys whose ENG gold or distractor matches REGEX (refers to other
    options: "all/none/both/neither/either of the above/these/options", bare letter lists such as
    "A and B", "Both b and c", bare "Both") are dropped; H1 primary, H5 primary, pointwise gold /
    distractor acceptance change and SB primary are recomputed on the remaining keys.
"""
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
REGEX = (r"\b(all|none|both|neither|either)\s+(of\s+)?(the\s+)?(above|these|those|options|answers|choices)\b"
         r"|\b(all|none)\s+the\s+above\b"
         r"|^\(?[a-d]\)?(\s*,?\s*\(?[a-d]\)?)*\s*(,|and|or|&)\s*\(?[a-d]\)?\s*(only)?\s*\.?$"
         r"|^(both|only|neither|either)\s+\(?[a-d]\)?\b"
         r"|^(both|neither|either)\.?$")
PAT = re.compile(REGEX, re.I)

meta, items, req, df, cov = A.load(False)
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)
judges = [j for j in C.MODELS if j in set(df.judge) and j not in PENDING]
D = C.DIALECTS

eng = items[items.dialect == "ENG"].set_index("key").reindex(keys)
flag_gold = np.array([bool(PAT.search(r.choices[r.answer].strip())) for r in eng.itertuples()])
flag_dist = np.array([bool(PAT.search(r.choices[r.distractor].strip())) for r in eng.itertuples()])
flag = flag_gold | flag_dist
keep = ~flag


def mask(M, m):
    return A._mask(M, m)


def sb(EA, EB, Msv, extra=None, keymask=None):
    msa = Msv["MSA"].to_numpy(float) == 1
    both = {}
    for v in D:
        b = (Msv[v].to_numpy(float) == 1) & msa
        if extra is not None:
            b &= extra
        if keymask is not None:
            b &= keymask
        both[v] = b
    a, b = paired_mean_over(EA, EB, D, mask=both)
    s = S.paired_summary(a, b, W)
    return {k: s[k] for k in ("est", "ci95", "n_keys")} | {"logodds_diff": s["logodds_diff"]}


out = {"description": __doc__, "regex": REGEX, "judges_excluded_pending": sorted(PENDING),
       "option_referential": {"n_keys_flagged": int(flag.sum()), "n_gold_flagged": int(flag_gold.sum()),
                              "n_distractor_flagged": int(flag_dist.sum()), "n_keys": len(keys),
                              "flagged_keys": [k for k, f in zip(keys, flag) if f]},
       "judges": {}}

for j in judges:
    dj = df[df.judge == j]
    en, sv, po = (dj[dj.condition == c] for c in ("pair_en", "solve", "point"))
    EA, EB = A.mpair(en, en, "correct", keys, "order", ref="MSA")
    Msv = A.kmat(sv, "correct", keys)
    r = {}

    # (1) chance-corrected DiD
    did_raw = np.column_stack([((EA[v] - EB[v]) - (Msv[v] - Msv["MSA"])).to_numpy(float) for v in D])
    did_cc = np.column_stack([(2 * (EA[v] - EB[v]) - (4 / 3) * (Msv[v] - Msv["MSA"])).to_numpy(float) for v in D])
    s_raw, s_cc = S.diff_summary(S.nanmean_rows(did_raw), W), S.diff_summary(S.nanmean_rows(did_cc), W)
    pa, pb = paired_mean_over(EA, EB, D)
    sa, sb_ = paired_mean_over(Msv, Msv["MSA"].to_numpy(float), D)
    r["did"] = {"raw": {k: s_raw[k] for k in ("est", "ci95", "p_two")},
                "chance_corrected": {k: s_cc[k] for k in ("est", "ci95", "ci90", "p_two")},
                "pair_gap_pp": float(np.nanmean(pa - pb)), "solve_gap_pp": float(np.nanmean(sa - sb_)),
                "predicted_pair_gap_from_solve_2_3": float(2 / 3 * np.nanmean(sa - sb_))}

    # (2a) know-or-guess bound on SB artefact; (2b) stricter proxy
    sr = {v: float(np.nanmean(Msv[v].to_numpy(float))) for v in C.VARIETIES}
    kf = lambda s: max(s - .25, 0) / .75 / s  # noqa: E731  P(known | solved)
    art = {v: 0.5 * (kf(sr[v]) - kf(sr["MSA"])) for v in D}
    # bootstrap of SB minus the bound (same weights; bound recomputed from bootstrap solve rates)
    msa = Msv["MSA"].to_numpy(float) == 1
    both = {v: (Msv[v].to_numpy(float) == 1) & msa for v in D}
    a, b = paired_mean_over(EA, EB, D, mask=both)
    bsb = S.boot_mean(W, a) - S.boot_mean(W, b)
    bs = {v: S.boot_mean(W, Msv[v].to_numpy(float)) for v in C.VARIETIES}
    bkf = lambda s: np.clip(s - .25, 0, None) / .75 / s  # noqa: E731
    bart = np.mean([0.5 * (bkf(bs[v]) - bkf(bs["MSA"])) for v in D], axis=0)
    mean_art = float(np.mean(list(art.values())))
    r["sb_artifact_bound"] = {"per_dialect": art, "mean": mean_art, "solve_rates": sr,
                              "sb_minus_bound": {"est": float(np.nanmean(a - b) - mean_art),
                                                 "ci95": S._ci(bsb - bart)},
                              "bound_share_of_sb": float(mean_art / np.nanmean(a - b))}
    r["sb"] = sb(EA, EB, Msv)
    all7 = np.all(np.column_stack([Msv[v].to_numpy(float) == 1 for v in C.VARIETIES]), axis=1)
    r["sb_strict_all_seven"] = sb(EA, EB, Msv, extra=all7)
    eng_solved = Msv["ENG"].to_numpy(float) == 1
    r["sb_strict_eng_msa_dialect"] = sb(EA, EB, Msv, extra=eng_solved)

    # (3) option-referential exclusion
    h1 = lambda m: S.paired_summary(*paired_mean_over(mask(EA, m), mask(EB, m), D), W)  # noqa: E731
    PA, PB = A.mpair(po, po, "correct", keys, "cand", ref="MSA")
    pg, pw = po[po.cand == "gold"], po[po.cand == "wrong"]
    Ag, Aw = A.kmat(pg, "accepted", keys), A.kmat(pw, "accepted", keys)
    allm = np.ones(len(keys), bool)
    res = {}
    for name, m in (("all", allm), ("excl_option_referential", keep)):
        g1 = h1(m)
        g5 = S.paired_summary(*paired_mean_over(mask(PA, m), mask(PB, m), D), W)
        ga = paired_mean_over(mask(Ag, m), Ag["MSA"].where(m).to_numpy(float), D)
        wa = paired_mean_over(mask(Aw, m), Aw["MSA"].where(m).to_numpy(float), D)
        res[name] = {"H1": {k: g1[k] for k in ("est", "ci95", "n_keys")},
                     "H5": {k: g5[k] for k in ("est", "ci95", "n_keys")},
                     "gold_accept_change": float(np.nanmean(ga[0] - ga[1])),
                     "dist_accept_change": float(np.nanmean(wa[0] - wa[1])),
                     "SB": sb(EA, EB, Msv, keymask=m)}
    r["option_referential"] = res
    out["judges"][j] = r

S.holm([out["judges"][j]["did"]["chance_corrected"] for j in judges])
(ROOT / "results/analysis/construct_checks.json").write_text(json.dumps(out, indent=1, default=float))
print("flagged keys:", int(flag.sum()), "gold", int(flag_gold.sum()), "dist", int(flag_dist.sum()))
for j, r in out["judges"].items():
    d, o = r["did"], r["option_referential"]
    print(f"{j:17s} pair={100*d['pair_gap_pp']:+.1f} solve={100*d['solve_gap_pp']:+.1f} "
          f"pred2/3={100*d['predicted_pair_gap_from_solve_2_3']:+.1f} DiDraw={100*d['raw']['est']:+.1f} "
          f"DiDcc={100*d['chance_corrected']['est']:+.1f} [{100*d['chance_corrected']['ci95'][0]:+.1f},"
          f"{100*d['chance_corrected']['ci95'][1]:+.1f}] p={d['chance_corrected']['p_two']:.3f} "
          f"holm={d['chance_corrected']['p_holm']:.3f} ci90=[{100*d['chance_corrected']['ci90'][0]:+.1f},{100*d['chance_corrected']['ci90'][1]:+.1f}]")
    bb = r["sb_artifact_bound"]; s7 = r["sb_strict_all_seven"]
    print(f"{'':17s} SB-bound={100*bb['sb_minus_bound']['est']:+.1f} [{100*bb['sb_minus_bound']['ci95'][0]:+.1f},"
          f"{100*bb['sb_minus_bound']['ci95'][1]:+.1f}] share={bb['bound_share_of_sb']:.2f} "
          f"all7={100*s7['est']:+.1f} [{100*s7['ci95'][0]:+.1f},{100*s7['ci95'][1]:+.1f}] n={s7['n_keys']} LO={s7['logodds_diff']['est']:+.2f} {s7['logodds_diff']['ci95']}")
    print(f"{'':17s} SB={100*r['sb']['est']:+.1f} (n={r['sb']['n_keys']}) bound={100*r['sb_artifact_bound']['mean']:+.2f} "
          f"strict={100*r['sb_strict_eng_msa_dialect']['est']:+.1f} [{100*r['sb_strict_eng_msa_dialect']['ci95'][0]:+.1f},"
          f"{100*r['sb_strict_eng_msa_dialect']['ci95'][1]:+.1f}] n={r['sb_strict_eng_msa_dialect']['n_keys']} "
          f"LO={r['sb_strict_eng_msa_dialect']['logodds_diff']['est']:+.2f}")
    for nm in ("all", "excl_option_referential"):
        x = o[nm]
        print(f"{'':17s} {nm:24s} H1={100*x['H1']['est']:+.2f} H5={100*x['H5']['est']:+.2f} "
              f"gold={100*x['gold_accept_change']:+.2f} dist={100*x['dist_accept_change']:+.2f} SB={100*x['SB']['est']:+.2f}")
