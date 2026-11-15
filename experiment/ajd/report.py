"""LaTeX tables and figures from the metrics dict produced by ``ajd.analyze.analyze``.

Compact headline tables for an 8-page paper (``\\input`` the ones you need):

* ``gaps_main.tex``    per judge: MSA accuracy, mean dialect gap [95% CI], error ratio, log-odds
                       difference, Holm p (H1, H2 / H5), equivalence class, ENG-MSA context;
                       pairwise (H1/H2) and pointwise (H5) panels.
* ``competence.tex``   DR5: raw gap, gap on solved-in-both keys, difference-in-differences,
                       P(wrong | solved), judge accuracy on not-solved items split by what was picked.
* ``register.tex``     DR6: placebo vs main pairs (tie rate, P(MSA | non-tie), forced P(MSA),
                       MSA-question P(MSA), contrast, P(MSA) at length parity).
* ``robustness.tex``   invalid rate, position, stability / noise floor, repetition (H4), rubric (H3),
                       and the primary-gap sensitivity checks (complete case, invalid = .5, IFM).
Repository/appendix tables: ``gaps_cells.tex``, ``strata.tex``, ``pointwise.tex``,
``register_common.tex``, ``mde.tex``, ``coverage.tex``.

KSA/UAE carry a dagger (Levantine-native translators, DR20); this only changes display names here.
"""
import numpy as np

from . import config as C

DAGGER = {"KSA", "UAE"}
CLASS_ABBR = {"equivalent": "equiv.", "different": "diff.", "inconclusive": "incon.", None: "--"}


def vname(v, short=False):
    n = C.VARIETY_NAMES[v][:3] if short else C.VARIETY_NAMES[v]
    return n + (r"$^\dagger$" if v in DAGGER else "")


def _g(s, k="est"):
    return None if not isinstance(s, dict) else s.get(k)


def pp(x, sign=True):
    return "--" if x is None else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}")


def ci(s, level="ci95"):
    c = (s or {}).get(level) or [None, None]
    return "--" if c[0] is None else f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def num(x, fmt="{:.2f}"):
    return "--" if x is None else fmt.format(x)


def pv(x):
    return "--" if x is None else ("<.001" if x < 0.001 else f"{x:.3f}")


def _tab(path, header, rows, align, panels=None):
    """Write a booktabs-free tabular. ``panels``: {row_index: title} inserts a full-width row."""
    lines = [r"\begin{tabular}{" + align + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    for i, r in enumerate(rows):
        if panels and i in panels:
            lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(header), panels[i]))
        lines.append(" & ".join(map(str, r)) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_tables(out, judges, d):
    """Write all tables to directory ``d`` (mock: results_mock/analysis/tables; real: results/analysis/tables)."""
    d.mkdir(parents=True, exist_ok=True)
    V, D = C.VARIETIES, C.DIALECTS

    # ---------------------------------------------------------------- main gaps (H1/H2, H5)
    rows, panels = [], {0: "Pairwise, English rubric (H1/H2)"}
    for j in judges:
        p, eng = out["H1"][j]["primary"], out["eng_context"][j]["pair_en"]
        rows.append([j, pp(p.get("acc_b"), False), f"{pp(p['est'])} {ci(p)}", num(_g(p.get("error_ratio"))),
                     num(_g(p.get("logodds_diff")), "{:+.2f}"), pv(p.get("p_holm")), pv(p.get("p_holm_H2")),
                     CLASS_ABBR[p.get("class")], pp(eng.get("est"))])
    panels[len(rows)] = "Pointwise (H5)"
    for j in judges:
        p, eng = out["H5"][j]["primary"], out["eng_context"][j]["point"]
        rows.append([j, pp(p.get("acc_b"), False), f"{pp(p['est'])} {ci(p)}", num(_g(p.get("error_ratio"))),
                     num(_g(p.get("logodds_diff")), "{:+.2f}"), pv(p.get("p_holm")), "--",
                     CLASS_ABBR[p.get("class")], pp(eng.get("est"))])
    _tab(d / "gaps_main.tex", ["Judge", "MSA", r"$\bar\Delta_{\mathrm{dial}}$ (pp) [95\% CI]", "ER",
                               r"$\Delta$logit", r"$p_{\mathrm{H}}$", r"$p_{\mathrm{H2}}$", "Class", "ENG"],
         rows, "lrlrrrrlr", panels)

    # ---------------------------------------------------------------- per-cell secondary gaps
    rows = []
    for j in judges:
        c = out["H1"][j]["cells"]
        rows.append([j] + [pp(c["ENG"]["est"])] + [
            pp(c[v]["est"]) + ("*" if (c[v].get("p_holm") or 1) < 0.05 else "") for v in D])
    _tab(d / "gaps_cells.tex", ["Judge", "ENG"] + [vname(v) for v in D], rows, "l" + "r" * (1 + len(D)))

    # ---------------------------------------------------------------- competence (DR5)
    rows = []
    for j in judges:
        cp = out["competence"][j]
        raw, sb, did = out["H1"][j]["primary"], cp["solved_both"]["primary"], cp["did"]["primary"]
        pw = cp["p_wrong_given"]
        pw_d = np.nanmean([_g(pw[v]["solved"]) if _g(pw[v]["solved"]) is not None else np.nan for v in D])
        split = cp["not_solved_split"]

        def split_acc(g):
            xs = [(split[v][g]["n_keys"], _g(split[v][g]["judge_acc"])) for v in D]
            xs = [(n, a) for n, a in xs if a is not None and n]
            return None if not xs else sum(n * a for n, a in xs) / sum(n for n, _ in xs)
        rows.append([j, pp(raw["est"]), f"{pp(sb['est'])} {ci(sb)}", f"{pp(did['est'])} {ci(did)}",
                     pv(did.get("p_holm")), CLASS_ABBR[sb.get("class")],
                     f"{pp(_g(pw['MSA']['solved']), False)}/{pp(None if np.isnan(pw_d) else pw_d, False)}",
                     f"{pp(split_acc('picked_distractor'), False)}/{pp(split_acc('picked_other'), False)}"])
    _tab(d / "competence.tex", ["Judge", r"$\bar\Delta$", r"$\bar\Delta$ | solved both", "DiD",
                                r"$p_{\mathrm{H}}$(DiD)", "Class", r"$P$(wrong|solved) MSA/dial",
                                "Acc.|not solved: distr./other"], rows, "lrllrlrr")

    # ---------------------------------------------------------------- register (DR6)
    rows = []
    for j in judges:
        rg = out["register"][j]["pooled"]
        pl, mn = rg.get("placebo", {}), rg.get("main", {})
        lg = out["register"][j]["logit"]["reg_forced"]
        nt = np.nanmean([_g(out["register"][j]["by_dialect"][x]["main"]["msa_given_nontie"]) or np.nan for x in D])
        ntp = np.nanmean([_g(out["register"][j]["by_dialect"][x]["placebo"]["msa_given_nontie"]) or np.nan for x in D])
        fq = mn.get("forced_minus_msaq", {})
        rows.append([j, pp(_g(pl.get("tie")), False), pp(None if np.isnan(ntp) else ntp, False),
                     pp(_g(mn.get("tie")), False), pp(None if np.isnan(nt) else nt, False),
                     pp(_g(mn.get("forced_msa")), False), pp(_g(mn.get("msaq_msa")), False),
                     f"{pp(fq.get('est'))} {ci(fq)}", pp(lg.get("p_msa_at_parity_avg"), False)])
    _tab(d / "register.tex", ["Judge", "Plc. tie", r"Plc. $P_{\mathrm{MSA}}|\neg$tie", "Tie",
                              r"$P_{\mathrm{MSA}}|\neg$tie", r"Forced $P_{\mathrm{MSA}}$", r"MSA-q $P_{\mathrm{MSA}}$",
                              r"$\Delta$ (forced$-$MSA-q)", "Parity"], rows, "lrrrrrrlr")

    rows = []
    for j in judges:
        ck = out["register"][j]["common_keys"]
        rows.append([j] + [f"{pp(_g(ck['forced'][x]), False)}/{pp(_g(ck['msaq'][x]), False)}" for x in D]
                    + [pp(_g(ck.get("forced_spread")), False)])
    _tab(d / "register_common.tex", ["Judge"] + [vname(x) for x in D] + ["Spread"], rows, "l" + "r" * (len(D) + 1))

    # ---------------------------------------------------------------- robustness & sensitivity
    rows = []
    for j in judges:
        inv = out["invalid"][j].get("pair_en", {})
        invr = np.mean([inv[v]["rate"] for v in inv]) if inv else None
        pos = out["position"][j]["overall"]
        st = out["stability"][j].get("overall", {})
        h4 = out["H4"][j]
        h3 = out["H3"][j].get("primary", {})
        sens = out["sensitivity"][j]
        rows.append([j, pp(invr, False), pp(_g(pos["inconsistency"]), False), pp(_g(pos["pA_minus_half"])),
                     f"{pp(_g(st), False)} {ci(st).replace('+', '')}",
                     pp(_g(h4.get("primary"))), pp(_g(h4.get("gold_repeated", {}).get("primary"))),
                     pp(_g(h3)), pp(_g(sens["complete_case"])), pp(_g(sens["invalid_half"])),
                     pp(_g(sens["ifm_unflagged"]))])
    _tab(d / "robustness.tex", ["Judge", "Inv.", "Incons.", r"$P(A)-.5$", r"Stability [95\% CI]",
                                r"Rep.\ wrong", r"Rep.\ gold", "MSA rubric", r"$\bar\Delta$ cc",
                                r"$\bar\Delta$ inv=.5", r"$\bar\Delta$ IFM"], rows, "lrrrlrrrrrr")

    # ---------------------------------------------------------------- strata (DR7)
    rows = []
    for j in judges:
        s = out["strata"][j]
        rows.append([j] + [f"{pp(s['question_only']['cells'][v]['est'])}/{pp(s['dialect_candidates']['cells'][v]['est'])}"
                           for v in D] + [" > ".join(C.VARIETY_NAMES[v][:3] for v in s.get("dialect_ranking", []))])
    _tab(d / "strata.tex", ["Judge"] + [vname(v) for v in D] + ["Ranking (drop)"], rows, "l" + "r" * len(D) + "l")

    # ---------------------------------------------------------------- pointwise leniency
    rows = []
    for j in judges:
        pw = out["pointwise"][j]
        rows.append([j] + [f"{pp(_g(pw['accept_gold'][v]), False)}/{pp(_g(pw['accept_wrong'][v]), False)}" for v in V])
    _tab(d / "pointwise.tex", ["Judge"] + [vname(v) for v in V], rows, "l" + "r" * len(V))

    # ---------------------------------------------------------------- MDE, coverage
    conds = ["pair_en_dialect_gap", "point_dialect_gap", "did_dialect", "rubric_lang", "repetition", "reg_forced_vs_half"]
    rows = [[j] + [pp((out["mde"][j].get(c) or {}).get("mde80"), False) for c in conds] for j in judges]
    _tab(d / "mde.tex", ["Judge", "pair\\_en", "point", "DiD", "rubric", "rep.", "reg."], rows, "l" + "r" * len(conds))

    cov = out["coverage"]["by_judge"]
    cnames = sorted({c for j in cov for c in cov[j]})
    rows = []
    for j in judges:
        inv = out["invalid"][j]
        rows.append([j] + [f"{pp(cov[j][c]['frac'], False)}/{pp(np.mean([x['rate'] for x in inv[c].values()]) if inv.get(c) else None, False)}"
                           for c in cnames])
    _tab(d / "coverage.tex", ["Judge"] + [c.replace("_", r"\_") for c in cnames], rows, "l" + "r" * len(cnames))


# ==================================================================== figures
def _dotplot(ax, judges, getter, cats, labels):
    """Judge-coloured dots with 95% CIs; ``getter(j, c)`` returns a summary dict {est, ci95}."""
    n = len(judges)
    for i, j in enumerate(judges):
        x = np.arange(len(cats)) + (i - (n - 1) / 2) * (0.8 / n)
        s = [getter(j, c) or {} for c in cats]
        e = np.array([np.nan if s_.get("est") is None else s_["est"] for s_ in s]) * 100
        lo = np.array([np.nan if not s_.get("ci95") or s_["ci95"][0] is None else s_["ci95"][0] for s_ in s]) * 100
        hi = np.array([np.nan if not s_.get("ci95") or s_["ci95"][1] is None else s_["ci95"][1] for s_ in s]) * 100
        ax.errorbar(x, e, yerr=[np.nan_to_num(e - lo), np.nan_to_num(hi - e)], fmt="o", ms=3, capsize=1.5, lw=0.8, label=j)
    ax.set_xticks(range(len(cats)), labels)


def write_figures(out, judges, d):
    """Write figures to ``d``: gap_vs_msa.pdf, competence.pdf, pointwise_register.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    vs = [v for v in C.VARIETIES if v != "MSA"]
    lab = lambda xs: [C.VARIETY_NAMES[v][:3] + ("†" if v in DAGGER else "") for v in xs]  # noqa: E731

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    _dotplot(ax, judges, lambda j, v: out["H1"][j]["cells"][v], vs, lab(vs))
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhspan(-2, 2, color="grey", alpha=0.12, lw=0)
    ax.set_ylabel("Pairwise acc. $-$ MSA (pp)")
    ax.legend(frameon=False, fontsize=6, ncol=len(judges), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(d / "gap_vs_msa.pdf"); plt.close(fig)

    # Competence decomposition: raw mean dialect gap, gap on solved-in-both keys, DiD.
    parts = [("raw", lambda j: out["H1"][j]["primary"]),
             ("solved both", lambda j: out["competence"][j]["solved_both"]["primary"]),
             ("DiD", lambda j: out["competence"][j]["did"]["primary"])]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    for i, (name, get) in enumerate(parts):
        x = np.arange(len(judges)) + (i - 1) * 0.25
        s = [get(j) for j in judges]
        e = np.array([np.nan if x_.get("est") is None else x_["est"] for x_ in s]) * 100
        lo = np.array([np.nan if x_["ci95"][0] is None else x_["ci95"][0] for x_ in s]) * 100
        hi = np.array([np.nan if x_["ci95"][1] is None else x_["ci95"][1] for x_ in s]) * 100
        ax.errorbar(x, e, yerr=[np.nan_to_num(e - lo), np.nan_to_num(hi - e)], fmt="os^"[i], ms=3.5, capsize=1.5, lw=0.8, label=name)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhspan(-2, 2, color="grey", alpha=0.12, lw=0)
    ax.set_xticks(range(len(judges)), judges, fontsize=6.5)
    ax.set_ylabel("Mean dialect $-$ MSA (pp)")
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(d / "competence.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    _dotplot(axes[0], judges, lambda j, v: out["pointwise"][j]["accept_wrong"][v], C.VARIETIES, lab(C.VARIETIES))
    axes[0].set_ylabel("Distractor accepted (%)"); axes[0].set_title("Pointwise leniency", fontsize=8)
    for ax, key, title in ((axes[1], "forced_msa", "Forced, dialect question"),
                           (axes[2], "msaq_msa", "Forced, MSA question")):
        _dotplot(ax, judges, lambda j, x, k=key: out["register"][j]["by_dialect"][x]["main"][k], C.DIALECTS, lab(C.DIALECTS))
        ax.axhline(50, color="grey", lw=0.6); ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=8)
    axes[1].set_ylabel("MSA candidate chosen (%)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, ncol=len(judges), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(d / "pointwise_register.pdf"); plt.close(fig)
