"""Planted-truth recovery check: runs ajd.analyze.analyze(mock=True) on results_mock/ (written by
tests/simulate.py) and asserts each planted effect is recovered. Prints PASS/FAIL per check, INFO for
context that is reported but not asserted.

Usage (from experiment/):
    uv run --project .. python tests/simulate.py
    uv run --project .. python tests/check.py            # runs the mock analysis, then checks
    uv run --project .. python tests/check.py --no-run   # checks the existing results_mock/analysis/metrics.json

Never touches results/: analyze is only called with mock=True.
Written against metrics.json schema_version "A1.1"; a missing key is a FAIL that names the path.
Exit code = number of FAILs.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajd import config as C  # noqa: E402

MOCK = C.MOCK_RESULTS_DIR
ALPHA = 0.05
D = C.DIALECTS
V = C.VARIETIES
JUDGES = list(C.MODELS)
RESULTS = []


class Missing(Exception):
    pass


def get(m, path):
    node = m
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            raise Missing(path)
        node = node[part]
    return node


def check(name, fn):
    """fn() -> (ok, detail). A missing key or an exception counts as FAIL."""
    try:
        ok, detail = fn()
    except Missing as e:
        ok, detail = False, f"missing key {e}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def info(name, fn):
    try:
        detail = fn()
    except Missing as e:
        detail = f"missing key {e}"
    print(f"INFO  {name}  [{detail}]")


def near(x, target, tol):
    return x is not None and abs(x - target) <= tol


def f3(x):
    return "None" if x is None else f"{x:+.3f}"


def est(m, path):
    return get(m, path + "/est")


def sig(m, path, field="p_holm"):
    return get(m, path + "/" + field) < ALPHA


def run_checks(m, gt):
    R = gt["realised"]
    check("schema_version is A1.1 and mock=True",
          lambda: (m.get("schema_version") == "A1.1" and m.get("mock") is True,
                   f"{m.get('schema_version')}, mock={m.get('mock')}"))

    def avg(path_fmt):
        return sum(est(m, path_fmt.format(v=v)) for v in V) / len(V)

    # ------------------------------------------------------------ coverage / invalids (DR4)
    def cov(j):
        def f():
            byc = get(m, f"coverage/by_judge/{j}")
            present = sum(c["present"] for c in byc.values())
            pending = sum(c.get("transport_pending", 0) for c in byc.values())
            exp_present = gt["n_requests_per_judge"] - gt["missing_per_judge"][j]
            exp_pending = gt["missing_per_judge"][j]
            if gt.get("opus_pair_ar_deferred") and j == "opus-5.5":
                exp_present -= 14336
            return (present == exp_present and pending == exp_pending,
                    f"present {present}/{exp_present}, transport_pending {pending}/{exp_pending}")
        return f
    for j in JUDGES:
        check(f"{j}: coverage counts present rows and logged transport errors", cov(j))
    check("coverage flagged incomplete (~0.5% missing planted)",
          lambda: (get(m, "coverage_complete") is False, f"coverage_complete={get(m, 'coverage_complete')}"))

    def inv(j):
        def f():
            rates = [c[v]["rate"] for c in get(m, f"invalid/{j}").values() for v in c if "rate" in c[v]]
            mean = sum(rates) / len(rates)
            return near(mean, 0.01, 0.004), f"mean over cond×variety {mean:.4f} ({len(rates)} cells)"
        return f
    for j in JUDGES:
        check(f"{j}: invalid rate ≈ 1% (reported per judge × variety × condition)", inv(j))

    def inv_tests():
        ps = [c[v]["p_vs_msa"] for j in JUDGES for c in get(m, f"invalid/{j}").values() for v in c
              if "p_vs_msa" in c[v]]
        k = sum(p < ALPHA for p in ps)
        return k / len(ps) <= 0.10, f"{k}/{len(ps)} invalid-rate cells differ from MSA at α=.05 (planted uniform)"
    check("invalid-rate vs MSA tests ~ nominal", inv_tests)

    # ------------------------------------------------------------ haiku: null everywhere
    j = "haiku-5.5"
    for fam in ("H1", "H5"):
        check(f"{j}: {fam} primary classified 'equivalent' (90% CI within ±2pp, DR3c)",
              lambda fam=fam: (get(m, f"{fam}/{j}/primary/class") == "equivalent",
                               f"est {f3(est(m, f'{fam}/{j}/primary'))}, ci90 "
                               f"{[round(x, 4) for x in get(m, f'{fam}/{j}/primary/ci90')]}"))
        check(f"{j}: {fam} no Holm-significant dialect cell (secondary family)",
              lambda fam=fam: (not any(sig(m, f"{fam}/{j}/cells/{v}") for v in D),
                               ", ".join(f"{v}{f3(est(m, f'{fam}/{j}/cells/{v}'))}" for v in D)))
    check(f"{j}: H4 repetition effect not significant",
          lambda: (not sig(m, f"H4/{j}/primary"), f"est {f3(est(m, f'H4/{j}/primary'))}"))
    check(f"{j}: competence DiD primary not significant",
          lambda: (not sig(m, f"competence/{j}/did/primary"), f"est {f3(est(m, f'competence/{j}/did/primary'))}"))
    info(f"{j}: H3 rubric-language primary (no effect planted; pair_ar latent drawn independently of pair_en)",
         lambda: f"class {get(m, f'H3/{j}/primary/class')}, est {f3(est(m, f'H3/{j}/primary'))}")

    # ------------------------------------------------------------ opus: competence-inherited MAG gap
    j = "opus-5.5"
    check(f"{j}: raw pair_en MAG gap ≈ -15pp, Holm-significant (H1 secondary)",
          lambda: (near(est(m, f"H1/{j}/cells/MAG"), -0.15, 0.035) and sig(m, f"H1/{j}/cells/MAG"),
                   f"{f3(est(m, f'H1/{j}/cells/MAG'))}, p_holm {get(m, f'H1/{j}/cells/MAG/p_holm'):.2g}"))
    check(f"{j}: H1 primary (mean of 5 dialects ≈ -3pp) classified 'different'",
          lambda: (get(m, f"H1/{j}/primary/class") == "different" and near(est(m, f"H1/{j}/primary"), -0.03, 0.01),
                   f"{f3(est(m, f'H1/{j}/primary'))}, {get(m, f'H1/{j}/primary/class')}"))
    check(f"{j}: other dialects no H1 gap",
          lambda: (all(abs(est(m, f"H1/{j}/cells/{v}")) < 0.02 and not sig(m, f"H1/{j}/cells/{v}")
                       for v in D if v != "MAG"),
                   ", ".join(f"{v}{f3(est(m, f'H1/{j}/cells/{v}'))}" for v in D if v != "MAG")))
    check(f"{j}: solve MAG − MSA ≈ -15pp",
          lambda: (near(est(m, f"solve/{j}/MAG") - est(m, f"solve/{j}/MSA"), -0.15, 0.035),
                   f3(est(m, f"solve/{j}/MAG") - est(m, f"solve/{j}/MSA"))))
    check(f"{j}: MAG gap on solved-in-both keys ≈ 0, not Holm-significant (DR5a)",
          lambda: (near(est(m, f"competence/{j}/solved_both/cells/MAG"), 0.0, 0.02)
                   and not sig(m, f"competence/{j}/solved_both/cells/MAG"),
                   f"{f3(est(m, f'competence/{j}/solved_both/cells/MAG'))}, "
                   f"p_holm {get(m, f'competence/{j}/solved_both/cells/MAG/p_holm'):.3f}"))
    check(f"{j}: MAG DiD ≈ 0, not Holm-significant (DR5b)",
          lambda: (near(est(m, f"competence/{j}/did/cells/MAG"), 0.0, 0.03)
                   and not sig(m, f"competence/{j}/did/cells/MAG"),
                   f"{f3(est(m, f'competence/{j}/did/cells/MAG'))}, "
                   f"p_holm {get(m, f'competence/{j}/did/cells/MAG/p_holm'):.3f}"))
    check(f"{j}: DiD primary not 'different' (no judge-specific effect claimed)",
          lambda: (get(m, f"competence/{j}/did/primary/class") != "different",
                   get(m, f"competence/{j}/did/primary/class")))
    check(f"{j}: P(judge wrong | solved) ≈ 0 in MSA and MAG (DR5c)",
          lambda: (all(est(m, f"competence/{j}/p_wrong_given/{v}/solved") < 0.03 for v in ("MSA", "MAG")),
                   ", ".join(f"{v} {est(m, f'competence/{j}/p_wrong_given/{v}/solved'):.3f}" for v in ("MSA", "MAG"))))

    # ------------------------------------------------------------ deepseek: judging-specific EGY gap
    j = "deepseek-flash"
    check(f"{j}: raw pair_en EGY gap ≈ -8pp, Holm-significant",
          lambda: (near(est(m, f"H1/{j}/cells/EGY"), -0.08, 0.035) and sig(m, f"H1/{j}/cells/EGY"),
                   f3(est(m, f"H1/{j}/cells/EGY"))))
    check(f"{j}: other dialects no H1 gap",
          lambda: (all(not sig(m, f"H1/{j}/cells/{v}") for v in D if v != "EGY"),
                   ", ".join(f"{v}{f3(est(m, f'H1/{j}/cells/{v}'))}" for v in D if v != "EGY")))
    check(f"{j}: solve unaffected (|EGY − MSA| < 3pp)",
          lambda: (abs(est(m, f"solve/{j}/EGY") - est(m, f"solve/{j}/MSA")) < 0.03,
                   f3(est(m, f"solve/{j}/EGY") - est(m, f"solve/{j}/MSA"))))
    check(f"{j}: EGY gap on solved-in-both keys ≈ -8pp, Holm-significant (DR5a)",
          lambda: (near(est(m, f"competence/{j}/solved_both/cells/EGY"), -0.08, 0.035)
                   and sig(m, f"competence/{j}/solved_both/cells/EGY"),
                   f3(est(m, f"competence/{j}/solved_both/cells/EGY"))))
    check(f"{j}: EGY DiD ≈ -8pp, Holm-significant (DR5b; simulator-realised {R[j]['pair_en_did']['EGY']:+.3f})",
          lambda: (near(est(m, f"competence/{j}/did/cells/EGY"), -0.08, 0.035)
                   and sig(m, f"competence/{j}/did/cells/EGY"),
                   f3(est(m, f"competence/{j}/did/cells/EGY"))))
    check(f"{j}: DiD of other dialects not Holm-significant",
          lambda: (all(not sig(m, f"competence/{j}/did/cells/{v}") for v in D if v != "EGY"),
                   ", ".join(f"{v}{f3(est(m, f'competence/{j}/did/cells/{v}'))}" for v in D if v != "EGY")))
    check(f"{j}: DiD primary classified 'different'",
          lambda: (get(m, f"competence/{j}/did/primary/class") == "different",
                   f3(est(m, f"competence/{j}/did/primary"))))
    check(f"{j}: EGY effect in both strata (question-only and dialect-candidate; planted in both)",
          lambda: (all(near(est(m, f"strata/{j}/{s}/cells/EGY"), -0.08, 0.04)
                       for s in ("question_only", "dialect_candidates")),
                   ", ".join(f"{s} {f3(est(m, f'strata/{j}/{s}/cells/EGY'))}"
                             for s in ("question_only", "dialect_candidates"))))
    check(f"{j}: dialect ranking puts EGY first",
          lambda: (get(m, f"strata/{j}/dialect_ranking")[0] == "EGY", str(get(m, f"strata/{j}/dialect_ranking"))))

    # ------------------------------------------------------------ gemini-3.8-flash: position + repetition
    j = "gemini-3.8-flash"
    check(f"{j}: position bias P(A) − 0.5 ≈ +0.12 (simulator-realised {R[j]['pair_en_P_A_minus_half']:+.3f})",
          lambda: (near(avg(f"position/{j}/{{v}}/pA_minus_half"), 0.12, 0.03),
                   f3(avg(f"position/{j}/{{v}}/pA_minus_half"))))
    check(f"{j}: order inconsistency ≈ 0.348 (= 0.6 × (0.7² + 0.3²))",
          lambda: (near(avg(f"position/{j}/{{v}}/inconsistency"), 0.348, 0.04),
                   f3(avg(f"position/{j}/{{v}}/inconsistency"))))
    check("other judges: |P(A) − 0.5| < 0.03",
          lambda: (all(abs(avg(f"position/{k}/{{v}}/pA_minus_half")) < 0.03 for k in JUDGES if k != j),
                   ", ".join(f"{k}{f3(avg(f'position/{k}/{{v}}/pA_minus_half'))}" for k in JUDGES if k != j)))
    check(f"{j}: H4 primary (distractor repeated → acc +20pp) 'different'",
          lambda: (near(est(m, f"H4/{j}/primary"), 0.20, 0.05) and get(m, f"H4/{j}/primary/class") == "different",
                   f"{f3(est(m, f'H4/{j}/primary'))}, p_holm {get(m, f'H4/{j}/primary/p_holm'):.2g}"))
    check(f"{j}: gold repeated → acc −20pp",
          lambda: (near(est(m, f"H4/{j}/gold_repeated/primary"), -0.20, 0.05),
                   f3(est(m, f"H4/{j}/gold_repeated/primary"))))
    check(f"{j}: signed repetition_bias ≈ −0.20 (repetition penalty, not length preference)",
          lambda: (near(est(m, f"H4/{j}/repetition_bias"), -0.20, 0.05), f3(est(m, f"H4/{j}/repetition_bias"))))
    check("other judges: H4 primary not Holm-significant, |repetition_bias| < 4pp",
          lambda: (all(not sig(m, f"H4/{k}/primary") and abs(est(m, f"H4/{k}/repetition_bias")) < 0.04
                       for k in JUDGES if k != j),
                   ", ".join(f"{k}{f3(est(m, f'H4/{k}/repetition_bias'))}" for k in JUDGES if k != j)))
    check(f"{j}: no dialect gap (H1 primary not 'different')",
          lambda: (get(m, f"H1/{j}/primary/class") != "different", f3(est(m, f"H1/{j}/primary"))))
    check(f"{j}: stability ≈ 0.75 (= 1 − 0.6 × 0.42); opus ≥ 0.95",
          lambda: (near(avg(f"stability/{j}/by_variety/{{v}}"), 0.748, 0.04)
                   and avg("stability/opus-5.5/by_variety/{v}") >= 0.95,
                   f"{j} {avg(f'stability/{j}/by_variety/{{v}}'):.3f}, "
                   f"opus {avg('stability/opus-5.5/by_variety/{v}'):.3f}"))

    # ------------------------------------------------------------ gpt-6.1-sol: register, placebo, leniency
    j = "gpt-6.1-sol"
    P = f"register/{j}/pooled"
    check(f"{j}: substantive reg_forced P(MSA) ≈ 0.75, ≠ 0.5",
          lambda: (near(est(m, f"{P}/main/forced_msa"), 0.75, 0.04)
                   and get(m, f"{P}/main/forced_msa/p_vs_half") < ALPHA, f3(est(m, f"{P}/main/forced_msa"))))
    check(f"{j}: substantive reg_forced_msaq P(MSA) ≈ 0.50",
          lambda: (near(est(m, f"{P}/main/msaq_msa"), 0.50, 0.04), f3(est(m, f"{P}/main/msaq_msa"))))
    check(f"{j}: mirror contrast forced − msaq ≈ +0.25, significant (MSA preference under a dialect "
          f"question; pure register matching would give the opposite pattern)",
          lambda: (near(est(m, f"{P}/main/forced_minus_msaq"), 0.25, 0.05)
                   and get(m, f"{P}/main/forced_minus_msaq/p_two") < ALPHA,
                   f3(est(m, f"{P}/main/forced_minus_msaq"))))
    check(f"{j}: per-dialect substantive P(MSA) ≈ 0.75 in every dialect (incl. UAE after placebo removal)",
          lambda: (all(near(est(m, f"register/{j}/by_dialect/{d}/main/forced_msa"), 0.75, 0.06) for d in D),
                   ", ".join(f"{d} {est(m, f'register/{j}/by_dialect/{d}/main/forced_msa'):.3f}" for d in D)))
    check(f"{j}: substantive reg_tie tie rate ≈ 0.30",
          lambda: (near(est(m, f"{P}/main/tie"), 0.30, 0.04), f3(est(m, f"{P}/main/tie"))))
    check(f"{j}: P(MSA | non-tie) in reg_tie ≈ 0.75 (UAE substantive)",
          lambda: (near(est(m, f"register/{j}/by_dialect/UAE/main/msa_given_nontie"), 0.75, 0.06),
                   f3(est(m, f"register/{j}/by_dialect/UAE/main/msa_given_nontie"))))
    check(f"{j}: placebo (orthography-only) TIE ≥ 0.95 and forced P(MSA) ≈ 0.5 (DR6)",
          lambda: (est(m, f"{P}/placebo/tie") >= 0.95 and near(est(m, f"{P}/placebo/forced_msa"), 0.5, 0.08)
                   and get(m, f"{P}/placebo/forced_msa/p_vs_half") >= ALPHA,
                   f"tie {est(m, f'{P}/placebo/tie'):.3f}, forced_msa {est(m, f'{P}/placebo/forced_msa'):.3f}"))
    check(f"{j}: length-parity logit P(MSA) ≈ 0.75 forced / 0.50 msaq (no length effect planted)",
          lambda: (near(get(m, f"register/{j}/logit/reg_forced/p_msa_at_parity_avg"), 0.75, 0.05)
                   and near(get(m, f"register/{j}/logit/reg_forced_msaq/p_msa_at_parity_avg"), 0.50, 0.05),
                   f"{get(m, f'register/{j}/logit/reg_forced/p_msa_at_parity_avg'):.3f} / "
                   f"{get(m, f'register/{j}/logit/reg_forced_msaq/p_msa_at_parity_avg'):.3f}"))
    check("other judges: substantive reg_forced P(MSA) ≈ 0.5, |contrast| < 5pp",
          lambda: (all(near(est(m, f"register/{k}/pooled/main/forced_msa"), 0.5, 0.04)
                       and abs(est(m, f"register/{k}/pooled/main/forced_minus_msaq")) < 0.05
                       for k in JUDGES if k != j),
                   ", ".join(f"{k} {est(m, f'register/{k}/pooled/main/forced_msa'):.3f}" for k in JUDGES if k != j)))

    def placebo_counts():
        mine = gt["placebo"]["orthography_only_pairs"]
        theirs = {d: get(m, f"design/register_pairs/{d}/placebo") for d in D}
        ok = all(abs(mine[d] - theirs[d]) <= max(8, 0.1 * mine[d]) for d in D)
        return ok, (f"analysis {theirs} (Σ{sum(theirs.values())}) vs independent normaliser {mine} "
                    f"(Σ{sum(mine.values())}); DR6 quoted 279")
    check("placebo membership agrees with independent normaliser (±max(8, 10%) per dialect)", placebo_counts)
    DR6_COUNTS = {"UAE": 213, "EGY": 36, "MAG": 23, "SYR": 5, "KSA": 2}  # notes/design_review.md, DR6
    check("placebo counts equal DR6's quoted counts exactly (Σ279)",
          lambda: ({d: get(m, f"design/register_pairs/{d}/placebo") for d in D} == DR6_COUNTS,
                   str({d: get(m, f"design/register_pairs/{d}/placebo") for d in D})))

    check(f"{j}: leniency — accept distractor SYR ≈ 0.25, MSA ≈ 0.05",
          lambda: (near(est(m, f"pointwise/{j}/accept_wrong/SYR"), 0.25, 0.04)
                   and near(est(m, f"pointwise/{j}/accept_wrong/MSA"), 0.05, 0.03),
                   f"SYR {est(m, f'pointwise/{j}/accept_wrong/SYR'):.3f}, "
                   f"MSA {est(m, f'pointwise/{j}/accept_wrong/MSA'):.3f}"))
    check(f"{j}: accept_gold unaffected (all varieties ≈ 0.89)",
          lambda: (all(near(est(m, f"pointwise/{j}/accept_gold/{v}"), 0.89, 0.04) for v in V),
                   ", ".join(f"{v} {est(m, f'pointwise/{j}/accept_gold/{v}'):.3f}" for v in V)))
    check(f"{j}: H5 SYR pointwise gap ≈ −10pp, Holm-significant; other dialects not",
          lambda: (near(est(m, f"H5/{j}/cells/SYR"), -0.10, 0.03) and sig(m, f"H5/{j}/cells/SYR")
                   and not any(sig(m, f"H5/{j}/cells/{v}") for v in D if v != "SYR"),
                   f"SYR {f3(est(m, f'H5/{j}/cells/SYR'))}"))
    info(f"{j}: accept_wrong_gap SYR (leniency reported separately per H5)",
         lambda: f3(est(m, f"pointwise/{j}/accept_wrong_gap/SYR")))

    # ------------------------------------------------------------ noise judges, family-wise control, H2
    for k in ("gpt-6-luna", "gemini-4-argon", "gpt-6.1-sol"):
        info(f"{k}: no planted pair_en dialect effect",
             lambda k=k: (f"H1 primary {f3(est(m, f'H1/{k}/primary'))} class {get(m, f'H1/{k}/primary/class')}; "
                          f"Holm-sig cells {[v for v in D if sig(m, f'H1/{k}/cells/{v}')] or 'none'}; "
                          f"simulator-realised {R[k]['gap_vs_msa_keylevel']['pair_en']['mean_5_dialects']:+.4f}"))
    check("H1 primary Holm-significant exactly for opus-5.5 and deepseek-flash",
          lambda: ({k for k in JUDGES if sig(m, f"H1/{k}/primary")} == {"opus-5.5", "deepseek-flash"},
                   str(sorted(k for k in JUDGES if sig(m, f"H1/{k}/primary")))))
    check("H2 (one-sided dialect < MSA) rejects for opus-5.5 and deepseek-flash",
          lambda: (all(get(m, f"H1/{k}/primary/p_holm_H2") < ALPHA for k in ("opus-5.5", "deepseek-flash")),
                   ", ".join(f"{k} {get(m, f'H1/{k}/primary/p_holm_H2'):.2g}" for k in ("opus-5.5", "deepseek-flash"))))


def run_unit_checks(m):
    """Pure-function checks on synthetic inputs (no I/O, never a non-mock analysis)."""
    import numpy as np
    import pandas as pd
    from ajd import analyze as A
    from ajd.register import placebo_key

    check("placebo_key: spacing-only and hamza-alif/diacritic differences are placebo; ئ/ؤ seats are not",
          lambda: (placebo_key("أوغير") == placebo_key("أو غير") and placebo_key("على20") == placebo_key("على 20")
                   and placebo_key("إلى،") == placebo_key("الي") and placebo_key("دائمًا") != placebo_key("دايمًا"), ""))

    def matched():
        # key k1: MSA both orders, UAE only order 2. Unmatched d = 1 - 0.5 = 0.5; order-matched d = 1 - 1 = 0.
        rows = pd.DataFrame([("k1", "MSA", "o1", 0.0), ("k1", "MSA", "o2", 1.0), ("k1", "UAE", "o2", 1.0),
                             ("k2", "MSA", "o1", 1.0), ("k2", "UAE", "o1", 0.0), ("k2", "MSA", "o2", 1.0),
                             ("k2", "UAE", "o2", 0.0)], columns=["key", "variety", "order", "correct"])
        Am, Bm = A.mpair(rows, rows, "correct", ["k1", "k2"], "order", ref="MSA")
        d = (Am["UAE"] - Bm["UAE"]).tolist()
        return d == [0.0, -1.0] and Bm["MSA"].tolist() == [0.5, 1.0], f"d_k={d}"
    check("kmatched: d_k is order-matched (mean over shared orders of v - MSA), DR3a", matched)

    def waivers():
        req = pd.DataFrame({"rid": ["a", "b", "c"], "condition": ["pair_en", "pair_ar", "pair_ar"]})
        df_ok = pd.DataFrame({"judge": ["opus-5.5"], "rid": ["a"], "condition": ["pair_en"]})
        df_part = pd.DataFrame({"judge": ["opus-5.5"] * 2, "rid": ["a", "b"], "condition": ["pair_en", "pair_ar"]})
        df_other = pd.DataFrame({"judge": ["gpt-6.1-sol"], "rid": ["a"], "condition": ["pair_en"]})
        # The real waiver set is empty (the Opus pair_ar cell was completed); test the mechanism with it set.
        saved, A.COVERAGE_WAIVERS = A.COVERAGE_WAIVERS, {("opus-5.5", "pair_ar")}
        try:
            c1, c2, c3 = (A.coverage_report(req, d, {}, {}) for d in (df_ok, df_part, df_other))
        finally:
            A.COVERAGE_WAIVERS = saved
        ok = (c1["complete"] is False and c1["complete_except_waivers"] is True
              and c2["complete_except_waivers"] is False and c3["complete_except_waivers"] is False)
        return ok, f"waived-absent {c1['complete_except_waivers']}, partial {c2['complete_except_waivers']}, non-waived {c3['complete_except_waivers']}"
    check("coverage assertion: only an entirely absent Opus pair_ar is waived (DR4)", waivers)
    check("mock metrics record coverage_complete_except_waivers = False (planted missing rows)",
          lambda: (get(m, "coverage_complete_except_waivers") is False, ""))
    check("H4 class uses the field named in holm_families (two-sided p_holm); p_holm_less present",
          lambda: ("class uses p_holm" in get(m, "holm_families/H4")
                   and all(get(m, f"H4/{k}/primary/holm_reject") == (get(m, f"H4/{k}/primary/p_holm") < ALPHA)
                           and get(m, f"H4/{k}/primary/p_holm_less") is not None
                           for k in JUDGES if "primary" in get(m, f"H4/{k}")), ""))
    check("IFM flagged-key source recorded and available (118 keys in sample)",
          lambda: (all(get(m, f"sensitivity/{k}/ifm_flagged_keys/source_available") is True for k in JUDGES),
                   str({get(m, f"sensitivity/{k}/ifm_flagged_keys/n_in_sample") for k in JUDGES})))
    _ = np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true", help="skip analyze(mock=True); use existing metrics.json")
    args = ap.parse_args()
    gt_path = MOCK / "ground_truth.json"
    if not gt_path.exists():
        raise SystemExit("run tests/simulate.py first")
    if not args.no_run:
        from ajd.analyze import analyze
        with contextlib.redirect_stdout(io.StringIO()):  # analyze dumps metrics to stdout
            analyze(mock=True)  # mock only: writes results_mock/analysis, never results/
    m = json.loads((MOCK / "analysis" / "metrics.json").read_text())
    gt = json.loads(gt_path.read_text())
    print("================ planted-truth checks ================")
    run_checks(m, gt)
    print("================ unit checks ================")
    run_unit_checks(m)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\nSUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} PASS, {n_fail} FAIL")
    sys.exit(min(n_fail, 255))


if __name__ == "__main__":
    main()
