"""Adversarial variant of tests/simulate.py: plants two effects that tests/check.py does not cover, runs
ajd.analyze.analyze(mock=True) and asserts the A1 handling. Mock only: writes results_mock/ (overwriting
the standard mock data; re-run tests/simulate.py afterwards to restore it).

Planted (on the two "noise" judges, everything else identical to simulate.py):
  E1 gpt-6-luna      ENG-only gap: pair_en/pair_ar accuracy on ENG -20pp, all dialects == MSA.
                     A1: ENG - MSA is context outside every family -> H1/H2 primary ~0 (equals the mean
                     of the 5 dialect cells), ENG cell carries no Holm p, H1_secondary has 40 cells,
                     eng_context shows ~-0.20.
  E2 gemini-4-argon  Invalid-only difference: +20% invalid outputs in pair_en KSA, choices otherwise
                     unchanged. A1/DR4: primary counts invalid as wrong -> KSA cell ~-0.18; complete-case
                     KSA cell / primary ~0; invalid=0.5 in between; invalid rate KSA ~0.21 with
                     p_vs_msa < .05; other conditions and varieties unaffected.

Usage (from experiment/):  uv run --project .. python tests/simulate_extra.py
Exit code = number of FAILs.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate as SIM  # noqa: E402
from ajd import config as C  # noqa: E402

ENG_DROP = 0.40          # know 0.64 -> 0.24: acc 0.82 -> 0.62 (-20pp) on ENG
EXTRA_INVALID = 0.20
J_ENG, J_INV = "gpt-6-luna", "gemini-4-argon"

# E1: ENG-only knowledge drop (shared by pair_en / pair_ar / verbose; rep = 0 for this judge)
SIM.PROFILES[J_ENG]["pair"] = lambda v, solved: (0.64 - (ENG_DROP if v == "ENG" else 0.0), 0.0)

# E2: extra invalid outputs in pair_en KSA only
_render = SIM.render


def render(j, q, label):
    if j == J_INV and q["condition"] == "pair_en" and q["variety"] == "KSA" \
            and SIM.U(j, "extra-invalid", q["rid"]) < EXTRA_INVALID:
        return "I cannot determine this."
    return _render(j, q, label)


SIM.render = render

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]")


def main():
    assert C.MOCK_RESULTS_DIR.name == "results_mock"
    sys.argv = [sys.argv[0]]
    SIM.main()
    from ajd.analyze import analyze
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = analyze(mock=True)
    m = json.loads((C.MOCK_RESULTS_DIR / "analysis" / "metrics.json").read_text())
    check("mock flag set, metrics written under results_mock/analysis", m["mock"] is True)
    D = C.DIALECTS

    # ---------------- E1 ENG-only gap
    h1 = m["H1"][J_ENG]
    eng = h1["cells"]["ENG"]
    check("E1 ENG cell est ~ -0.20", abs(eng["est"] + 0.20) < 0.03, f"{eng['est']:+.4f}")
    check("E1 ENG cell itself significant (effect is real, p_two)", eng["p_two"] < 0.05, f"p_two={eng['p_two']}")
    check("E1 ENG cell carries no Holm-adjusted p (outside every family)",
          not any(k.startswith("p_holm") for k in eng), str(sorted(k for k in eng if k.startswith("p"))))
    check("E1 eng_context pair_en == H1 ENG cell", abs(m["eng_context"][J_ENG]["pair_en"]["est"] - eng["est"]) < 1e-12)
    pr = h1["primary"]
    mean5 = float(np.mean([h1["cells"][v]["est"] for v in D]))
    check("E1 H1 primary est == mean of the 5 dialect cells (ENG excluded)", abs(pr["est"] - mean5) < 1e-9,
          f"primary {pr['est']:+.4f}, mean5 {mean5:+.4f}")
    check("E1 H1 primary not rejected (p_holm >= .05) and H2 not rejected",
          pr["p_holm"] >= 0.05 and pr["p_holm_H2"] >= 0.05,
          f"p_holm={pr['p_holm']:.3f} p_holm_H2={pr['p_holm_H2']:.3f} class={pr['class']}")
    check("E1 H1 primary |est| < 0.02", abs(pr["est"]) < 0.02, f"{pr['est']:+.4f}")
    check("E1 H1_secondary family has 40 cells", "40" in m["holm_families"]["H1_secondary"],
          m["holm_families"]["H1_secondary"])
    n_holm = sum("p_holm" in m["H1"][j]["cells"][v] for j in m["H1"] for v in m["H1"][j]["cells"])
    check("E1 exactly 40 H1 cells Holm-adjusted across judges", n_holm == 40, str(n_holm))
    h3 = m["H3"][J_ENG]
    check("E1 H3 (pair_ar - pair_en) unaffected: ENG drop is in both arms", abs(h3["primary"]["est"]) < 0.02,
          f"{h3['primary']['est']:+.4f}")
    check("E1 point ENG unaffected", abs(m["eng_context"][J_ENG]["point"]["est"]) < 0.02,
          f"{m['eng_context'][J_ENG]['point']['est']:+.4f}")
    # competence solved_both / did use dialects only too
    sb = m["competence"][J_ENG]["solved_both"]["primary"]["est"]
    check("E1 competence solved_both primary ~0 (dialects only)", abs(sb) < 0.02, f"{sb:+.4f}")

    # ---------------- E2 invalid-only difference
    inv = m["invalid"][J_INV]["pair_en"]
    check("E2 pair_en KSA invalid rate ~0.21", abs(inv["KSA"]["rate"] - 0.208) < 0.02, f"{inv['KSA']['rate']:.4f}")
    check("E2 pair_en KSA invalid differs from MSA (p_vs_msa < .05)", inv["KSA"]["p_vs_msa"] < 0.05,
          f"p={inv['KSA']['p_vs_msa']}")
    others = [inv[v]["rate"] for v in C.VARIETIES if v != "KSA"]
    check("E2 other varieties' pair_en invalid ~1%", max(others) < 0.02, f"max {max(others):.4f}")
    oth_c = [m["invalid"][J_INV][c]["KSA"]["rate"] for c in m["invalid"][J_INV] if c != "pair_en"]
    check("E2 KSA invalid ~1% in every other condition", max(oth_c) < 0.025, f"max {max(oth_c):.4f}")
    g = m["H1"][J_INV]
    exp = -EXTRA_INVALID * 0.90  # acc 0.90 (know .8 + .5*.2); invalid replaces a correct pick w.p. .9
    check("E2 primary (invalid=wrong) KSA cell ~ -0.18", abs(g["cells"]["KSA"]["est"] - exp) < 0.03,
          f"{g['cells']['KSA']['est']:+.4f}")
    check("E2 primary dialect mean ~ -0.036", abs(g["primary"]["est"] - exp / 5) < 0.012,
          f"{g['primary']['est']:+.4f} class={g['primary']['class']} p_holm={g['primary']['p_holm']:.4g}")
    cc = m["sensitivity"][J_INV]["complete_case"]
    check("E2 complete-case primary ~0 and within ±2pp margin", abs(cc["est"]) < 0.01 and
          -0.02 < cc["ci90"][0] and cc["ci90"][1] < 0.02, f"{cc['est']:+.4f} ci90={cc['ci90']}")
    hf = m["sensitivity"][J_INV]["invalid_half"]
    exp_h = -EXTRA_INVALID * (0.90 - 0.5) / 5
    check("E2 invalid=0.5 primary between (~ -0.016)", abs(hf["est"] - exp_h) < 0.01
          and g["primary"]["est"] < hf["est"] < cc["est"] + 0.005, f"{hf['est']:+.4f}")
    po = m["H5"][J_INV]["cells"]["KSA"]["est"]
    check("E2 pointwise KSA unaffected", abs(po) < 0.02, f"{po:+.4f}")
    sv = m["solve"][J_INV]
    check("E2 solve KSA ~ solve MSA", abs(sv["KSA"]["est"] - sv["MSA"]["est"]) < 0.03,
          f"{sv['KSA']['est']:.4f} vs {sv['MSA']['est']:.4f}")

    # ---------------- untouched judges still behave (spot check)
    hk = m["H1"]["haiku-5.5"]["primary"]
    check("haiku-5.5 H1 still equivalent", hk["class"] == "equivalent", hk["class"])
    print(f"SUMMARY: {sum(RES)}/{len(RES)} PASS, {len(RES) - sum(RES)} FAIL")
    sys.exit(len(RES) - sum(RES))


if __name__ == "__main__":
    main()
