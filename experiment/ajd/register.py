"""Register probe clean-up (spec amendment A1 / DR6).

The register conditions pit the gold option as rendered by the dialect translators against the same
gold option as rendered by the MSA translators (question in dialect d for reg_tie/reg_forced, in MSA
for reg_forced_msaq). Many pairs differ only in orthography or punctuation. We therefore:

1. normalise both strings (:func:`normalise`) and call a pair a **placebo** pair if the normalised
   strings are equal once all whitespace is removed (:func:`placebo_key`; spacing differences such
   as أو غير / أوغير are orthographic, DR6) (expect TIE ~ 1 and P(MSA) ~ 0.5); all other pairs are **main** pairs;
2. report, per judge x dialect x {placebo, main}: tie rate, P(MSA | non-tie) in reg_tie, P(MSA) in
   reg_forced, P(MSA) in reg_forced_msaq and the paired contrast reg_forced - reg_forced_msaq
   (negative = the judge prefers the candidate matching the question's register rather than MSA
   per se);
3. fit a key-clustered logistic regression chose_msa ~ log_len_ratio + char_sim + dialect on main
   pairs and report P(MSA) at length parity (log_len_ratio = 0, char_sim at its mean);
4. compare dialects on the common key subset where all five dialects have a main pair.

Per-key unit: the MSA share over the valid rows of a key's two orders (invalid rows excluded and
reported separately). Tests against 0.5 are sign-flip tests on (share - 0.5).
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from . import stats as S

_HAMZA_ALIF = re.compile("[أإآٱ]")          # أ إ آ ٱ -> ا
_MARKS = re.compile("[ً-ْٰـ]")              # harakat, tanwin, shadda, sukun, dagger alif, tatweel
_PUNCT = re.compile(r"[،؛؟٪-٭۔!-/:-@\[-`{-~«»“”‘’…–—]")
_WS = re.compile(r"\s+")


def normalise(s: str) -> str:
    """Arabic orthographic normaliser used to define placebo register pairs.

    Steps: hamza/alif variants (أ إ آ ٱ) -> ا; ta marbuta ة -> ه; alif maqsura ى -> ي; strip
    harakat/tanwin/shadda/sukun (U+064B-U+0652), dagger alif (U+0670) and tatweel (U+0640); remove
    ASCII and Arabic punctuation (، ؛ ؟ ٪ ٫ ٬ ٭ ۔ and quotes/dashes); collapse whitespace; strip.
    Latin letters and digits are left unchanged (case is kept).
    """
    s = _HAMZA_ALIF.sub("ا", s)
    s = s.replace("ة", "ه").replace("ى", "ي")
    s = _MARKS.sub("", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def placebo_key(s: str) -> str:
    """Comparison key for the placebo rule: :func:`normalise` with all whitespace removed, so pairs
    differing only in spacing (e.g. على20 / على 20) count as orthography-only. hamza seats ئ / ؤ are
    deliberately NOT folded (DR6 counts; folding them is a substantive choice, see design_review)."""
    return _WS.sub("", normalise(s))


def pair_table(items, dialects) -> pd.DataFrame:
    """One row per (key, dialect) register pair, mirroring conditions.build_requests' exclusion rule.

    Columns: key, variety, placebo (:func:`placebo_key` equal), char_sim (difflib ratio of the
    normalised strings), log_len_ratio = log(len MSA / len dialect) of the raw stripped strings
    (positive = MSA candidate longer), len_msa, len_dia.
    """
    by = {(r.key, r.dialect): r for r in items.itertuples()}
    rows = []
    for key in sorted({k for k, _ in by}):
        m = by[(key, "MSA")]
        mg = m.choices[m.answer].strip()
        for d in dialects:
            r = by[(key, d)]
            dg = r.choices[r.answer].strip()
            if dg == mg:
                continue
            nm, nd = normalise(mg), normalise(dg)
            rows.append({"key": key, "variety": d, "placebo": placebo_key(mg) == placebo_key(dg),
                         "char_sim": SequenceMatcher(None, nd, nm, autojunk=False).ratio(),
                         "log_len_ratio": float(np.log(max(1, len(mg)) / max(1, len(dg)))),
                         "len_msa": len(mg), "len_dia": len(dg)})
    return pd.DataFrame(rows)


def _kv(sub, col, keys):
    """Per-key mean of ``col`` over rows (NaN rows ignored), aligned to ``keys``."""
    if sub.empty:
        return np.full(len(keys), np.nan)
    return sub.groupby("key")[col].mean().reindex(keys).to_numpy(float)


def _vs_half(x, W):
    """Mean share with CI, sign-flip p vs 0.5 (field p_vs_half), sd of (x - .5) and its MDE."""
    s = S.mean_summary(x, W)
    d = np.asarray(x, float) - 0.5
    s["p_vs_half"] = S.signflip(d)
    s["sd_d"] = float(np.nanstd(d, ddof=1)) if s["n_keys"] > 1 else None
    s["mde80"] = S.mde(s["sd_d"], s["n_keys"])
    return s


def _cell(dj, pairs_d, keys, W):
    """All register quantities for one judge x dialect x pair-set (rows already restricted)."""
    ks = set(pairs_d.key)
    t = dj[(dj.condition == "reg_tie") & dj.key.isin(ks)]
    f = dj[(dj.condition == "reg_forced") & dj.key.isin(ks)]
    q = dj[(dj.condition == "reg_forced_msaq") & dj.key.isin(ks)]
    tv, fv, qv = (x[x.valid] for x in (t, f, q))
    is_tie = _kv(tv.assign(v=(tv.reg_outcome == "tie").astype(float)), "v", keys)
    msa_nt = _kv(tv[tv.reg_outcome != "tie"].assign(v=(tv.reg_outcome == "msa").astype(float)), "v", keys)
    msa_f = _kv(fv.assign(v=(fv.reg_outcome == "msa").astype(float)), "v", keys)
    msa_q = _kv(qv.assign(v=(qv.reg_outcome == "msa").astype(float)), "v", keys)
    out = {"n_pairs": len(ks),
           "tie": S.mean_summary(is_tie, W),
           "msa_given_nontie": _vs_half(msa_nt, W),
           "forced_msa": _vs_half(msa_f, W),
           "msaq_msa": _vs_half(msa_q, W),
           "forced_minus_msaq": S.paired_summary(msa_f, msa_q, W, scale=False),
           "invalid": {c: (float(1 - x.valid.mean()) if len(x) else None)
                       for c, x in (("reg_tie", t), ("reg_forced", f), ("reg_forced_msaq", q))},
           "n_rows": {"reg_tie": len(t), "reg_forced": len(f), "reg_forced_msaq": len(q)}}
    return out, {"tie": is_tie, "forced": msa_f, "msaq": msa_q}


def _logit_parity(dj, pairs, condition, dialects):
    """Key-clustered logistic chose_msa ~ log_len_ratio + char_sim + variety on main pairs.

    Unit: valid row (both orders). Cluster-robust (by key) covariance. Returns odds ratios for the
    two covariates and predicted P(MSA) at length parity (log_len_ratio = 0, char_sim = mean over
    main pairs) per dialect, with 95% CIs from the robust covariance.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    r = dj[(dj.condition == condition) & dj.valid].merge(pairs[~pairs.placebo], on=["key", "variety"])
    if len(r) < 50 or r.reg_outcome.nunique() < 2:
        return {"n": int(len(r)), "note": "too few rows"}
    r = r.assign(chose_msa=(r.reg_outcome == "msa").astype(float),
                 dialect=pd.Categorical(r.variety, categories=[d for d in dialects if d in set(r.variety)]))
    groups = pd.factorize(r.key)[0]
    try:
        m = smf.glm("chose_msa ~ log_len_ratio + char_sim + dialect", data=r,
                    family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:  # report rather than hide a failed fit
        return {"n": int(len(r)), "error": f"{type(e).__name__}: {e}"}
    ci = m.conf_int()
    res = {"n": int(len(r)), "n_keys": int(r.key.nunique()),
           "coef": {k: {"or": float(np.exp(m.params[k])), "lo": float(np.exp(ci.loc[k, 0])),
                        "hi": float(np.exp(ci.loc[k, 1])), "p": float(m.pvalues[k])}
                    for k in ("log_len_ratio", "char_sim")}}
    grid = pd.DataFrame({"dialect": pd.Categorical(list(r.dialect.cat.categories), categories=r.dialect.cat.categories),
                         "log_len_ratio": 0.0, "char_sim": float(r.char_sim.mean())})
    pr = m.get_prediction(grid).summary_frame(alpha=0.05)
    res["p_msa_at_parity"] = {d: {"est": float(pr["mean"].iloc[i]),
                                  "ci95": [float(pr["mean_ci_lower"].iloc[i]), float(pr["mean_ci_upper"].iloc[i])]}
                              for i, d in enumerate(grid.dialect)}
    res["p_msa_at_parity_avg"] = float(pr["mean"].mean())
    res["p_msa_raw"] = float(r.chose_msa.mean())
    return res


def register_analysis(dj, pairs, keys, W, dialects):
    """Full DR6 register analysis for one judge's rows ``dj``.

    Returns {"by_dialect": {d: {"placebo": cell, "main": cell}}, "pooled": {"placebo", "main"},
    "logit": {"reg_forced", "reg_forced_msaq"}, "common_keys": {...}} where each ``cell`` is the
    dict documented in :func:`_cell`. "pooled" averages, per key, the MSA shares over the dialects in
    which that key has a pair of the given type (key-level unit; sign-flip vs 0.5).
    """
    out = {"by_dialect": {}, "pooled": {}}
    vecs = {"placebo": {}, "main": {}}
    for d in dialects:
        pd_ = pairs[pairs.variety == d]
        dd = dj[dj.variety == d]
        out["by_dialect"][d] = {}
        for name, sel in (("placebo", pd_.placebo), ("main", ~pd_.placebo)):
            cell, v = _cell(dd, pd_[sel], keys, W)
            out["by_dialect"][d][name] = cell
            vecs[name][d] = v
    for name in ("placebo", "main"):
        if not vecs[name]:
            continue
        pool = {m: S.nanmean_rows(np.column_stack([vecs[name][d][m] for d in dialects])) for m in ("tie", "forced", "msaq")}
        out["pooled"][name] = {"n_pairs": int(sum(out["by_dialect"][d][name]["n_pairs"] for d in dialects)),
                               "tie": S.mean_summary(pool["tie"], W),
                               "forced_msa": _vs_half(pool["forced"], W),
                               "msaq_msa": _vs_half(pool["msaq"], W),
                               "forced_minus_msaq": S.paired_summary(pool["forced"], pool["msaq"], W, scale=False)}
    # Holm over the 5 per-dialect forced-vs-0.5 tests on main pairs (exploratory family, per judge).
    S.holm([out["by_dialect"][d]["main"]["forced_msa"] for d in dialects], p_field="p_vs_half")
    S.holm([out["by_dialect"][d]["main"]["forced_minus_msaq"] for d in dialects], p_field="p_two")
    out["logit"] = {c: _logit_parity(dj, pairs, c, dialects) for c in ("reg_forced", "reg_forced_msaq")}

    main = pairs[~pairs.placebo]
    cnt = main.groupby("key").variety.nunique()
    common = sorted(cnt[cnt == len(dialects)].index)
    mask = np.isin(np.array(keys), common)
    ck = {"n_keys": len(common)}
    for m in ("forced", "msaq", "tie"):
        ck[m] = {d: S.mean_summary(np.where(mask, vecs["main"][d][m], np.nan), W) for d in dialects}
    # Spread across dialects on common keys (max - min of the per-dialect forced P(MSA)), with CI.
    M = np.column_stack([np.where(mask, vecs["main"][d]["forced"], np.nan) for d in dialects])
    if mask.any() and not np.all(np.isnan(M)):
        boots = np.column_stack([S.boot_mean(W, M[:, i]) for i in range(M.shape[1])])
        est = np.nanmean(M, 0)
        ck["forced_spread"] = {"est": float(np.nanmax(est) - np.nanmin(est)),
                               "ci95": S._ci(np.nanmax(boots, 1) - np.nanmin(boots, 1))}
    out["common_keys"] = ck
    return out
