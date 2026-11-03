"""Prompt templates and construction of every judge request.

Each request is a dict with a stable `rid`; `expected` holds what a reliable judge should output.
Conditions:
  solve      - the model answers the MCQ itself (competence baseline, used to condition judge errors)
  pair_en    - pairwise judge, English rubric: gold vs distractor, both orders
  pair_ar    - same items, MSA rubric (only the instruction language changes)
  verbose    - English rubric, distractor repeated VERBOSE_REPEATS times (128-key subset)
  verbose_gold - mirror of verbose: the gold answer is repeated instead
  pair_en_rep - pair_en prompts re-sent unchanged on the 128-key subset (run-to-run stability)
  reg_forced_msaq - reg_forced pairs with the question shown in MSA instead of the dialect
  point      - pointwise: one candidate (gold or distractor), judge says correct/incorrect
  reg_tie    - question in dialect d; gold answer in d vs the same gold answer in MSA; TIE allowed
  reg_forced - same pair, forced choice (no TIE) -> direction of register preference
"""
import hashlib
import json

from . import config as C

LETTERS = "ABCD"

PAIR_EN = """You are an impartial judge. Below is a question and two candidate answers. Exactly one candidate answers the question correctly. Decide which one. Judge correctness only.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}} or {{"winner": "B"}}"""

# MSA rendering of PAIR_EN, revised for fidelity and approved by a native Arabic speaker (2026-10-25).
PAIR_AR = """أنت حَكَم محايد. فيما يلي سؤال وإجابتان مقترحتان. واحدة فقط من الإجابتين تجيب عن السؤال بشكل صحيح. حدِّد أيّهما. اعتمد في حكمك على الصحة وحدها.

السؤال:
{question}

الإجابة المقترحة A:
{a}

الإجابة المقترحة B:
{b}

أجب بصيغة JSON فقط: {{"winner": "A"}} أو {{"winner": "B"}}"""

REG_TIE = """You are an impartial judge. Below is a question and two candidate answers. Decide which candidate is the better answer to the question. If they are equally good, answer TIE.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}}, {{"winner": "B"}}, or {{"winner": "TIE"}}"""

REG_FORCED = """You are an impartial judge. Below is a question and two candidate answers. Decide which candidate is the better answer to the question. You must choose one.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}} or {{"winner": "B"}}"""

POINT = """You are an impartial judge. Below is a question and a candidate answer. Decide whether the candidate answers the question correctly. Judge correctness only.

Question:
{question}

Candidate answer:
{a}

Respond with JSON only: {{"verdict": "correct"}} or {{"verdict": "incorrect"}}"""

SOLVE = """Answer the following multiple-choice question.

Question:
{question}

{options}

Respond with JSON only: {{"answer": "A"}}, {{"answer": "B"}}, {{"answer": "C"}}, or {{"answer": "D"}}"""


def _rid(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:20]


def _pairs(cond, template, key, variety, question, good, bad, meta, good_label="gold"):
    """Two requests: good in slot A, then good in slot B."""
    out = []
    for order in ("good_first", "good_second"):
        a, b = (good, bad) if order == "good_first" else (bad, good)
        prompt = template.format(question=question, a=a, b=b)
        out.append({
            "rid": _rid(cond, key, variety, order, prompt), "condition": cond, "key": key,
            "variety": variety, "order": order, "prompt": prompt,
            "good_slot": "A" if order == "good_first" else "B", "good_label": good_label, **meta,
        })
    return out


def build_requests(meta, items) -> list[dict]:
    by = {(r.key, r.dialect): r for r in items.itertuples()}
    verbose_keys = set(meta["verbose_keys"])
    flagged = set(meta.get("flagged_keys_in_sample") or [])
    reqs = []
    for key in meta["keys"]:
        base = {"domain": by[(key, "ENG")].domain, "flagged": key in flagged}
        for v in C.VARIETIES:
            r = by[(key, v)]
            gold, wrong = r.choices[r.answer], r.choices[r.distractor]
            opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r.choices))
            prompt = SOLVE.format(question=r.question, options=opts)
            reqs.append({"rid": _rid("solve", key, v, prompt), "condition": "solve", "key": key,
                         "variety": v, "prompt": prompt, "gold_letter": LETTERS[r.answer], **base})
            reqs += _pairs("pair_en", PAIR_EN, key, v, r.question, gold, wrong, base)
            reqs += _pairs("pair_ar", PAIR_AR, key, v, r.question, gold, wrong, base)
            for label, cand in (("gold", gold), ("wrong", wrong)):
                prompt = POINT.format(question=r.question, a=cand)
                reqs.append({"rid": _rid("point", key, v, label, prompt), "condition": "point", "key": key,
                             "variety": v, "prompt": prompt, "cand": label, **base})
            if key in verbose_keys:
                def repeat(t):
                    return ". ".join([t.rstrip(". ")] * C.VERBOSE_REPEATS) + "."
                reqs += _pairs("verbose", PAIR_EN, key, v, r.question, gold, repeat(wrong), base)
                # Mirror: the gold answer is repeated instead, separating length bias from a repetition penalty.
                reqs += _pairs("verbose_gold", PAIR_EN, key, v, r.question, repeat(gold), wrong, base)
                # Identical prompt sent a second time: run-to-run stability under default sampling.
                reqs += [{**q, "condition": "pair_en_rep", "rid": _rid("rep", q["rid"])}
                         for q in _pairs("pair_en", PAIR_EN, key, v, r.question, gold, wrong, base)]
        msa = by[(key, "MSA")]
        msa_gold = msa.choices[msa.answer]
        for d in C.DIALECTS:
            r = by[(key, d)]
            dia_gold = r.choices[r.answer]
            if dia_gold.strip() == msa_gold.strip():
                continue  # identical strings carry no register contrast; counted in analysis
            # "good" = the dialect-register answer; the MSA answer is in the other slot.
            reqs += _pairs("reg_tie", REG_TIE, key, d, r.question, dia_gold, msa_gold, base, "dialect")
            reqs += _pairs("reg_forced", REG_FORCED, key, d, r.question, dia_gold, msa_gold, base, "dialect")
            # Same answer pair, question shown in MSA: separates "prefers MSA" from "prefers the
            # candidate matching the question's register".
            reqs += _pairs("reg_forced_msaq", REG_FORCED, key, d, msa.question, dia_gold, msa_gold, base, "dialect")
    assert len({q["rid"] for q in reqs}) == len(reqs), "rid collision"
    return reqs


def output_schema(condition: str) -> dict:
    """JSON schema each judge is constrained to (structured output), matching the prompt's format."""
    if condition == "solve":
        name, labels = "answer", ["A", "B", "C", "D"]
    elif condition == "point":
        name, labels = "verdict", ["correct", "incorrect"]
    elif condition == "reg_tie":
        name, labels = "winner", ["A", "B", "TIE"]
    else:
        name, labels = "winner", ["A", "B"]
    return {"type": "object", "properties": {name: {"type": "string", "enum": labels}},
            "required": [name], "additionalProperties": False}
