"""Pointwise (H5) gold / distractor acceptance figure and numbers, plus the pairwise error ratio (post hoc).

Covers the pointwise (H5) result and the pairwise error ratio (post hoc).

Reads stored outputs only: results/<judge>.jsonl via ajd.analyze.load (read-only; no API calls),
plus results/analysis/metrics.json and results/analysis/pointwise_sdt.json.

(1) Per judge: Delta Gold / Delta Dist = per-key mean over the 5 dialects of
    [acceptance(v) - acceptance(MSA)] over VALID pointwise outputs (ajd.competence.paired_mean_over,
    as pointwise_sdt.json gold_gap and pointwise_solved.json "all"), with domain-stratified key
    bootstrap 95% CIs (ajd.stats.stratified_weights, B=2000, seed 2026) and sign-flip p.
    Identity check against H5 pointwise accuracy gap (metrics.json H5.<j>.primary.est):
      (a) exact form: H5 counts an invalid output as wrong, so with G' = P(accept | gold, invalid =
          not accepted) and D' = P(not rejected | distractor, invalid = not rejected) the per-cell
          identity acc = (G' + 1 - D') / 2 gives H5 gap = (dG' - dD') / 2 exactly;
      (b) valid-only form: (dGold - dDist) / 2, which differs from H5 only through invalid outputs.
(2) Gold-acceptance gap minus solve gap (per-key difference-in-differences, recomputed exactly as
    pointwise_sdt.py and checked against pointwise_sdt.json): sign-flip two-sided p (10,000 flips,
    seed 2026; ajd.stats.signflip) Holm-adjusted across the 8 judges.
(3) H1 pairwise mean dialect gap error ratio (1-acc_dialect)/(1-acc_MSA): metrics.json
    H1.<j>.primary.error_ratio (collected, not recomputed).
Output: results/analysis/pointwise_figure.json; figure results/analysis/figures/pointwise_gold_dist.pdf
(judges ordered by MSA pair_en accuracy, metrics.json accuracy.<j>.MSA).
"""
import json
import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

M = json.loads((ROOT / "results/analysis/metrics.json").read_text())
SDT = json.loads((ROOT / "results/analysis/pointwise_sdt.json").read_text())
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
NAMES = {"gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-luna": "GPT-6 Luna", "opus-5.5": "Opus 5.5", "haiku-5.5": "Haiku 5.5",
         "gemini-4-argon": "Gemini 4 Argon", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek-flash": "DeepSeek Flash"}
KEEP = ("est", "ci95", "n_keys", "p_two")

meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)


def change(Mx):
    a, b = paired_mean_over(Mx, Mx["MSA"], DIALECTS)
    s = S.paired_summary(a, b, W, scale=False)
    return {k: s[k] for k in KEEP}


out = {"method": __doc__, "judges": {}}
for j in M["design"]["judges"]:
    dj = df[df.judge == j]
    po = dj[dj.condition == "point"]
    assert not po.duplicated(["key", "variety", "cand"]).any(), j
    pg, pw = po[po.cand == "gold"], po[po.cand == "wrong"]
    Ag, Aw = A.kmat(pg, "accepted", keys), A.kmat(pw, "accepted", keys)
    # invalid-as-wrong versions (exact H5 decomposition)
    Gp = A.kmat(pg, "correct", keys)                       # accepted gold, invalid = 0
    Dp = 1.0 - A.kmat(pw, "correct", keys)                 # not rejected distractor, invalid = 1
    Sv = A.kmat(dj[dj.condition == "solve"], "correct", keys)

    dG, dD = change(Ag), change(Aw)
    dGp, dDp = change(Gp), change(Dp)
    h5 = M["H5"][j]["primary"]["est"]
    exact = (dGp["est"] - dDp["est"]) / 2
    approx = (dG["est"] - dD["est"]) / 2

    did = np.column_stack([((Ag[v] - Ag["MSA"]) - (Sv[v] - Sv["MSA"])).to_numpy(float) for v in DIALECTS])
    did_s = S.diff_summary(S.nanmean_rows(did), W)
    ref = SDT["judges"][j]["gold_minus_solve_gap"]
    assert abs(did_s["est"] - ref["est"]) < 1e-12 and abs(did_s["p_two"] - ref["p_two"]) < 1e-12, j
    ref_g = SDT["judges"][j]["gold_gap"]
    assert abs(dG["est"] - ref_g["est"]) < 1e-12 and np.allclose(dG["ci95"], ref_g["ci95"]), j

    n_inv = int((~po.valid).sum())
    out["judges"][j] = {
        "display": NAMES[j],
        "msa_pairwise_accuracy": M["accuracy"][j]["MSA"]["est"],
        "delta_gold": dG, "delta_dist": dD,
        "identity": {"h5_gap": h5,
                     "half_dGold_minus_dDist_valid_only": approx,
                     "diff_valid_only_minus_h5": approx - h5,
                     "delta_gold_invalid_as_not_accepted": dGp["est"],
                     "delta_dist_invalid_as_not_rejected": dDp["est"],
                     "half_dGp_minus_dDp_exact": exact,
                     "diff_exact_minus_h5": exact - h5,
                     "n_invalid_pointwise_rows": n_inv, "n_pointwise_rows": int(len(po))},
        "gold_minus_solve_gap": {k: did_s[k] for k in KEEP},
        "pairwise_H1_mean_gap": M["H1"][j]["primary"]["est"],
        "pairwise_H1_error_ratio": M["H1"][j]["primary"]["error_ratio"],
    }

J = list(out["judges"])
adj = multipletests([out["judges"][j]["gold_minus_solve_gap"]["p_two"] for j in J], method="holm")[1]
for j, a in zip(J, adj):
    out["judges"][j]["gold_minus_solve_gap"]["p_holm8"] = float(a)
out["gold_minus_solve_holm_survivors_0.05"] = [j for j, a in zip(J, adj) if a < 0.05]
out["identity_max_abs_diff_exact"] = max(abs(out["judges"][j]["identity"]["diff_exact_minus_h5"]) for j in J)
(ROOT / "results/analysis/pointwise_figure.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

order = sorted(J, key=lambda j: -out["judges"][j]["msa_pairwise_accuracy"])
plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Liberation Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.5, 2.0))
x0 = np.arange(len(order))
for off, key, lab, col, mk in ((-0.12, "delta_gold", r"$\Delta$Gold acceptance", "#0072B2", "o"),
                               (+0.12, "delta_dist", r"$\Delta$Distractor acceptance", "#E69F00", "s")):
    e = np.array([out["judges"][j][key]["est"] for j in order]) * 100
    lo = np.array([out["judges"][j][key]["ci95"][0] for j in order]) * 100
    hi = np.array([out["judges"][j][key]["ci95"][1] for j in order]) * 100
    ax.errorbar(x0 + off, e, yerr=[e - lo, hi - e], fmt=mk, ms=4, lw=0.9, capsize=1.5, color=col, label=lab)
ax.axhline(0, color="black", lw=0.6)
for x in x0[:-1] + 0.5:
    ax.axvline(x, color="0.88", lw=0.5)
ax.set_xticks(x0, [NAMES[j].rsplit(" ", 1)[0] + "\n" + NAMES[j].rsplit(" ", 1)[1] for j in order], fontsize=8)
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylabel("Dialect $-$ MSA (pp)", fontsize=8)
ax.tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
(ROOT / "results/analysis/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results/analysis/figures/pointwise_gold_dist.pdf")

f = lambda s: f"{100*s['est']:+5.1f} [{100*s['ci95'][0]:+5.1f},{100*s['ci95'][1]:+5.1f}]"  # noqa: E731
for j in order:
    r = out["judges"][j]
    i = r["identity"]
    print(f"{j:17s} MSAacc={100*r['msa_pairwise_accuracy']:.1f} dG {f(r['delta_gold'])} dD {f(r['delta_dist'])} "
          f"H5={100*i['h5_gap']:+.2f} (dG-dD)/2={100*i['half_dGold_minus_dDist_valid_only']:+.2f} "
          f"exact={100*i['half_dGp_minus_dDp_exact']:+.2f} inv={i['n_invalid_pointwise_rows']} | "
          f"G-S {f(r['gold_minus_solve_gap'])} p={r['gold_minus_solve_gap']['p_two']:.4g} "
          f"holm={r['gold_minus_solve_gap']['p_holm8']:.4g} | ER={r['pairwise_H1_error_ratio']['est']:.3f} "
          f"[{r['pairwise_H1_error_ratio']['ci95'][0]:.3f},{r['pairwise_H1_error_ratio']['ci95'][1]:.3f}]")
print("Holm survivors:", out["gold_minus_solve_holm_survivors_0.05"], "max |exact-H5|:", out["identity_max_abs_diff_exact"])
