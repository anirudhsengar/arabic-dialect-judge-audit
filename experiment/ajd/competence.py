"""Judging vs competence decomposition (spec amendment A1 / DR5), the headline analysis.

``pair_en`` is a two-option MCQ with judge framing, so a dialect gap in judging may just inherit the
judge's own (in)ability to solve the dialect item. Per judge x variety (v != MSA):

(a) gap vs MSA on keys the same model solved in *both* v and MSA;
(b) per-key difference-in-differences (pair_v - pair_MSA) - (solve_v - solve_MSA), sign-flip test;
(c) P(judge wrong | solved) and P(judge wrong | not solved) per variety, with CIs;
(d) "not solved" split into "picked our distractor" vs "picked another wrong option" vs invalid
    output (keys whose solve row is missing are excluded);
    only the first is directly comparable with the two-option pairwise item;
(e) binomial GEE correct ~ variety * solved (exchangeable, clustered by key), MSA as reference.

A judge-specific dialect effect is claimed only if (a)/(b) show it. Per-key units: pair correctness
is the mean over the two orders; solve correctness is a single row (invalid = not solved).
"""
import numpy as np
import pandas as pd

from . import stats as S


def _where(mask, x):
    return np.where(mask, np.asarray(x, float), np.nan)


def paired_mean_over(Ma, Mb, cols, mask=None):
    """Key vectors (a, b) for a paired comparison averaged over ``cols``.

    Ma[c] is compared with Mb[c] (or with Mb itself if Mb is a 1-D vector, e.g. MSA). A cell (key, c)
    enters only if both sides are present and ``mask[c]`` (optional, boolean per key) is True; each
    key's a/b are the means over its usable cells, so the estimate is the per-key mean of d over
    the averaged columns.
    """
    A, Bm = [], []
    for c in cols:
        a = np.asarray(Ma[c], float)
        b = np.asarray(Mb[c] if isinstance(Mb, pd.DataFrame) else Mb, float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if mask is not None:
            ok &= np.asarray(mask[c], bool)
        A.append(_where(ok, a)); Bm.append(_where(ok, b))
    return S.nanmean_rows(np.column_stack(A)), S.nanmean_rows(np.column_stack(Bm))


def _gee(rows, varieties):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    r = rows.dropna(subset=["correct", "solved"])
    if len(r) < 100 or r.correct.nunique() < 2 or r.solved.nunique() < 2:
        return {"n": int(len(r)), "note": "too few rows or no variation"}
    # Categorical with MSA first = treatment coding with MSA as reference (no patsy C(): the config
    # module is imported as C in analyze.py and would shadow it).
    cats = ["MSA"] + [v for v in varieties if v != "MSA" and v in set(r.variety)]
    r = r.assign(variety=pd.Categorical(r.variety, categories=cats), solved=r.solved.astype(float))
    try:
        m = smf.gee("correct ~ variety * solved", groups="key", data=r, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    except Exception as e:  # report rather than hide a failed fit
        return {"n": int(len(r)), "error": f"{type(e).__name__}: {e}"}
    ci = m.conf_int()
    terms = {k: {"or": float(np.exp(m.params[k])), "lo": float(np.exp(ci.loc[k, 0])),
                 "hi": float(np.exp(ci.loc[k, 1])), "p": float(m.pvalues[k])}
             for k in m.params.index if k != "Intercept"}
    inter = [k for k in m.params.index if ":solved" in k]
    out = {"n": int(len(r)), "n_keys": int(r.key.nunique()), "terms": terms}
    if inter:
        try:
            R = np.zeros((len(inter), len(m.params)))
            for i, k in enumerate(inter):
                R[i, list(m.params.index).index(k)] = 1
            w = m.wald_test(R, scalar=True)
            out["interaction_wald"] = {"stat": float(w.statistic), "df": len(inter), "p": float(w.pvalue)}
        except Exception as e:
            out["interaction_wald_error"] = f"{type(e).__name__}: {e}"
    return out


def competence_analysis(en, solve, solve_letter, distr_letter, en_rows, keys, W, varieties, dialects,
                        en_matched=None):
    """DR5 (a)-(e) for one judge.

    en, solve: DataFrames keys x varieties of per-key pair_en accuracy and solve correctness.
    solve_letter: keys x varieties of the parsed solve answer (None if invalid/missing).
    distr_letter: Series key -> letter of our fixed distractor (same index in every variety).
    en_rows: order-level pair_en rows (key, variety, order, correct) for the GEE.
    en_matched: optional order-matched (A, B) key matrices of pair_en vs MSA (A[v] - B[v] = mean over
    orders present on both sides of correct_v - correct_MSA, DR3a); used for the pair-side gap in
    (a) and (b). Without it the unmatched en[v] - en["MSA"] is used.
    Returns {"solved_both": {cells, primary}, "did": {cells, primary}, "p_wrong_given": {...},
    "not_solved_split": {...}, "gee": {...}}. Holm across judges is applied by the caller.
    """
    non_msa = [v for v in varieties if v != "MSA"]
    msa_s = solve["MSA"].to_numpy(float)
    both = {v: (solve[v].to_numpy(float) == 1) & (msa_s == 1) for v in non_msa}
    if en_matched is not None:
        EA, EB = en_matched
    else:
        EA, EB = en, pd.DataFrame({v: en["MSA"] for v in en.columns}, index=en.index)
    out = {"solved_both": {"cells": {}}, "did": {"cells": {}}}
    for v in non_msa:
        out["solved_both"]["cells"][v] = S.paired_summary(_where(both[v], EA[v]), _where(both[v], EB[v]), W)
        out["did"]["cells"][v] = S.diff_summary((EA[v] - EB[v]) - (solve[v] - solve["MSA"]), W)
    a, b = paired_mean_over(EA, EB, dialects, mask=both)
    out["solved_both"]["primary"] = S.paired_summary(a, b, W)
    did = np.column_stack([((EA[v] - EB[v]) - (solve[v] - solve["MSA"])).to_numpy(float) for v in dialects])
    out["did"]["primary"] = S.diff_summary(S.nanmean_rows(did), W)

    pw = {}
    for v in varieties:
        s = solve[v].to_numpy(float)
        wrong = 1 - en[v].to_numpy(float)
        ws, wn = _where(s == 1, wrong), _where(s == 0, wrong)
        pw[v] = {"solved": S.mean_summary(ws, W), "not_solved": S.mean_summary(wn, W),
                 "solve_rate": S.mean_summary(s, W)}
        if v != "MSA":
            wm = _where(msa_s == 1, 1 - en["MSA"].to_numpy(float))
            bd = S.boot_mean(W, ws) - S.boot_mean(W, wm)
            pw[v]["solved_minus_msa"] = {"est": S._f(np.nanmean(ws) - np.nanmean(wm)) if np.any(~np.isnan(ws)) and np.any(~np.isnan(wm)) else None,
                                         "ci95": S._ci(bd)}
    out["p_wrong_given"] = pw

    split = {}
    dl = distr_letter.reindex(keys).to_numpy(object)
    for v in varieties:
        lt = solve_letter[v].to_numpy(object)
        s = solve[v].to_numpy(float)
        grp = {"picked_distractor": (s == 0) & (lt == dl),
               "picked_other": (s == 0) & (lt != dl) & pd.notna(lt),
               "invalid": (s == 0) & pd.isna(lt)}  # missing solve rows are NaN, not 0
        split[v] = {g: {"n_keys": int(m.sum()), "judge_acc": S.mean_summary(_where(m, en[v]), W)}
                    for g, m in grp.items()}
    out["not_solved_split"] = split

    long = solve.rename_axis(index="key", columns="variety").reset_index().melt(
        id_vars="key", var_name="variety", value_name="solved")
    rows = en_rows.merge(long, on=["key", "variety"], how="left")
    out["gee"] = _gee(rows, varieties)
    return out
