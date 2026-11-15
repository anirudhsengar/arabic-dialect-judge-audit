"""Statistical primitives for the pre-specified analysis (spec amendment A1, DR3/DR9).

Conventions used throughout:

* **Unit = key.** Every outcome is first reduced to one number per key (the mean over the rows that
  exist for that key: the two orders, the two pointwise candidates, ...). Keys with no row are NaN
  (transport errors are *missing*, not wrong). All vectors are aligned to a fixed, sorted key list.
* **Bootstrap (DR9).** Domain-stratified multinomial weights ``W`` (B x K): within each domain the
  32 keys are resampled with replacement, so every replicate keeps 32 keys per domain. One ``W`` over
  the 1,024 keys is shared by all judges, varieties and conditions; a separate ``W`` over the
  128-key subset is used for verbose / verbose_gold / pair_en_rep. A replicate's mean of a key
  vector ``x`` is ``W @ x0 / W @ mask`` (NaNs carry zero weight), so subsets (register pairs,
  strata, complete cases) are handled by masking, with the pairing kept.
* **Tests (DR3a).** Sign-flip permutation test on per-key paired differences ``d_k`` (10,000 random
  sign vectors, seed 2026; p = (1 + #extreme) / (1 + 10,000)). Order-level exact McNemar is kept only
  as a sensitivity check.
* **Scale (DR3d).** Every pp gap is reported with the error ratio (1 - acc_a) / (1 - acc_b) and the
  log-odds difference logit(acc_a) - logit(acc_b), with bootstrap CIs; accuracies are clipped to
  [0.5/n, 1 - 0.5/n] for these two transforms.
* **Equivalence (DR3c).** delta = 2 pp: "equivalent" if the 90% bootstrap CI lies inside
  (-delta, +delta); otherwise "different" if the Holm-adjusted test rejects at 0.05; otherwise
  "inconclusive". (Equivalence is checked first, exactly in the order A1 lists the classes; both
  underlying flags are stored so the other precedence can be read off.)
"""
from functools import lru_cache

import numpy as np
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests

B = 2000
N_FLIPS = 10_000
SEED = 2026  # equal to config.SEED; duplicated so this module has no dependency on the frozen config
DELTA = 0.02
ALPHA = 0.05


# --------------------------------------------------------------------------- bootstrap weights
def stratified_weights(keys, seed=SEED, b=B):
    """Domain-stratified multinomial bootstrap weights, shape (b, len(keys)).

    ``keys`` are ``"domain::qid"`` strings. Within each domain (in sorted domain order) the n_d keys
    get multinomial(n_d, 1/n_d) counts per replicate, so domain sizes are fixed (domains have equal
    weight by design). A fresh ``default_rng(seed)`` is used, so the weights depend only on the key
    list and the seed.
    """
    keys = list(keys)
    dom = np.array([k.split("::", 1)[0] for k in keys])
    rng = np.random.default_rng(seed)
    W = np.zeros((b, len(keys)), dtype=np.float64)
    for d in sorted(set(dom)):
        idx = np.flatnonzero(dom == d)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=b)
    return W


def boot_mean(W, x):
    """Bootstrap replicates of the NaN-aware mean of key vector ``x`` (length K) -> (B,)."""
    m = ~np.isnan(x)
    den = W @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (W @ np.where(m, x, 0.0)) / den, np.nan)


def _ci(boot, level=0.95):
    a = (1 - level) / 2
    b = boot[~np.isnan(boot)]
    if b.size == 0:
        return [None, None]
    lo, hi = np.percentile(b, [100 * a, 100 * (1 - a)])
    return [float(lo), float(hi)]


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def mean_summary(x, W):
    """Mean of a key vector with 95% bootstrap CI: {est, ci95, n_keys}."""
    x = np.asarray(x, float)
    n = int((~np.isnan(x)).sum())
    if n == 0:
        return {"est": None, "ci95": [None, None], "n_keys": 0}
    return {"est": float(np.nanmean(x)), "ci95": _ci(boot_mean(W, x)), "n_keys": n}


# --------------------------------------------------------------------------- tests
@lru_cache(maxsize=8)
def _signs(n, n_flips=N_FLIPS, seed=SEED):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(n_flips, n), dtype=np.int8) * 2 - 1).astype(np.float32)


def signflip(d, alternative="two-sided"):
    """Sign-flip permutation test of mean(d) = 0 on per-key differences (NaNs dropped).

    alternative: "two-sided", "less" (mean < 0) or "greater". Monte Carlo with 10,000 sign vectors
    from ``default_rng(2026)`` (the same vectors for every test with the same n, so results are
    deterministic). Returns None if no key has data.
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return None
    if np.all(d == 0):
        return 1.0
    obs = d.mean()
    perm = (_signs(d.size) @ d.astype(np.float32)) / d.size
    tol = 1e-9 * max(1.0, abs(obs))
    if alternative == "two-sided":
        k = np.sum(np.abs(perm) >= abs(obs) - tol)
    elif alternative == "less":
        k = np.sum(perm <= obs + tol)
    else:
        k = np.sum(perm >= obs - tol)
    return float((1 + k) / (1 + perm.size))


def mcnemar_exact(a, b):
    """Exact McNemar on aligned paired binary arrays (NaN pairs dropped). Sensitivity check only."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"p": 1.0 if n01 + n10 == 0 else float(binomtest(n10, n01 + n10, 0.5).pvalue),
            "n10": n10, "n01": n01, "n_pairs": int(ok.sum())}


def holm(summaries, p_field="p_two", out_field="p_holm"):
    """Holm-adjust ``p_field`` across a family of summary dicts in place (None entries skipped)."""
    live = [s for s in summaries if s is not None and s.get(p_field) is not None]
    if live:
        adj = multipletests([s[p_field] for s in live], method="holm")[1]
        for s, a in zip(live, adj):
            s[out_field] = float(a)
    return summaries


def classify(s, delta=DELTA, alpha=ALPHA, p_field="p_holm"):
    """Equivalence classification of a primary gap (see module docstring). Sets s['class']."""
    lo, hi = s.get("ci90", [None, None])
    s["within_margin"] = bool(lo is not None and -delta < lo and hi < delta)
    s["holm_reject"] = bool(s.get(p_field) is not None and s[p_field] < alpha)
    s["class"] = ("equivalent" if s["within_margin"] else
                  "different" if s["holm_reject"] else "inconclusive")
    return s


# --------------------------------------------------------------------------- paired summaries
def _logit(p):
    return np.log(p) - np.log1p(-p)


def mde(sd, n, alpha=ALPHA, power=0.8):
    """Minimum detectable mean difference for a key-level paired test (normal approximation)."""
    if not n or sd is None or not np.isfinite(sd):
        return None
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / np.sqrt(n))


def paired_summary(a, b, W, scale=True):
    """Paired comparison of key vectors a (condition/variety of interest) and b (reference).

    Keys where either side is NaN are dropped from both (complete pairs). Returns
    est = mean(a - b) with 95% and 90% bootstrap CIs, acc_a/acc_b, error ratio and log-odds
    difference (if ``scale``) with 95% CIs, sign-flip p (two-sided and one-sided "less"), n_keys,
    sd of d_k and the MDE (alpha .05 two-sided, 80% power).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = ~(np.isnan(a) | np.isnan(b))
    a, b = np.where(both, a, np.nan), np.where(both, b, np.nan)
    n = int(both.sum())
    if n == 0:
        return {"est": None, "n_keys": 0, "ci95": [None, None], "ci90": [None, None],
                "p_two": None, "p_less": None}
    d = a - b
    ba, bb = boot_mean(W, a), boot_mean(W, b)
    bd = ba - bb
    sd = float(np.nanstd(d, ddof=1)) if n > 1 else None
    out = {"est": float(np.nanmean(d)), "ci95": _ci(bd, .95), "ci90": _ci(bd, .90),
           "acc_a": float(np.nanmean(a)), "acc_b": float(np.nanmean(b)),
           "p_two": signflip(d, "two-sided"), "p_less": signflip(d, "less"),
           "n_keys": n, "sd_d": sd, "mde80": mde(sd, n)}
    if scale:
        eps = 0.5 / n
        cl = lambda p: np.clip(p, eps, 1 - eps)  # noqa: E731
        er = (1 - cl(out["acc_a"])) / (1 - cl(out["acc_b"]))
        ber = (1 - cl(ba)) / (1 - cl(bb))
        lo = _logit(cl(out["acc_a"])) - _logit(cl(out["acc_b"]))
        blo = _logit(cl(ba)) - _logit(cl(bb))
        out["error_ratio"] = {"est": _f(er), "ci95": _ci(ber)}
        out["logodds_diff"] = {"est": _f(lo), "ci95": _ci(blo)}
    return out


def diff_summary(d, W):
    """Summary of an already-formed per-key difference vector d (e.g. a difference-in-differences)."""
    d = np.asarray(d, float)
    n = int((~np.isnan(d)).sum())
    if n == 0:
        return {"est": None, "n_keys": 0, "ci95": [None, None], "ci90": [None, None],
                "p_two": None, "p_less": None}
    bd = boot_mean(W, d)
    sd = float(np.nanstd(d, ddof=1)) if n > 1 else None
    return {"est": float(np.nanmean(d)), "ci95": _ci(bd, .95), "ci90": _ci(bd, .90),
            "p_two": signflip(d, "two-sided"), "p_less": signflip(d, "less"),
            "n_keys": n, "sd_d": sd, "mde80": mde(sd, n)}


def nanmean_rows(mat):
    """Row-wise NaN mean of a (K, m) array without warnings; all-NaN rows -> NaN."""
    mat = np.asarray(mat, float)
    m = ~np.isnan(mat)
    cnt = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.where(m, mat, 0).sum(1) / cnt, np.nan)
