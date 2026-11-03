# Prompt templates

All templates are copied verbatim from `experiment/ajd/conditions.py` (generated from the module source, not retyped). `{question}`, `{a}`, `{b}`, `{options}` are filled with the DialectalArabicMMLU item text (bare option texts as candidates); `{{`/`}}` are Python `str.format` escapes for literal braces. Each prompt is sent as a single user message, with no system prompt from our side. The module defines 6 distinct templates, used by 10 conditions.

## Condition to template map

| Condition | Template | What changes | Keys |
|---|---|---|---|
| `solve` | `SOLVE` | Judge model answers the MCQ itself (competence baseline). `{options}` = four lines `A. ...` to `D. ...`. | 1,024 |
| `pair_en` | `PAIR_EN` | Gold vs distractor, both orders. | 1,024 |
| `pair_ar` | `PAIR_AR` | Same items as pair_en; only the rubric language changes (MSA). | 1,024 |
| `point` | `POINT` | One candidate (gold or distractor). | 1,024 |
| `verbose` | `PAIR_EN` | Distractor repeated 3x (see transformation below). | 128 |
| `verbose_gold` | `PAIR_EN` | Gold answer repeated 3x. | 128 |
| `pair_en_rep` | `PAIR_EN` | pair_en prompts re-sent unchanged (run-to-run stability). | 128 |
| `reg_tie` | `REG_TIE` | Question in dialect d; gold in d vs gold in MSA; TIE allowed. | 1,024 (pairs with identical strings excluded) |
| `reg_forced` | `REG_FORCED` | Same pairs as reg_tie, forced choice. | 1,024 (as reg_tie) |
| `reg_forced_msaq` | `REG_FORCED` | reg_forced pairs with the question shown in MSA. | 1,024 (as reg_tie) |

Pairwise conditions send each item in both orders (`good_first`: good answer in slot A; `good_second`: in slot B).

## `PAIR_EN`

```text
You are an impartial judge. Below is a question and two candidate answers. Exactly one candidate answers the question correctly. Decide which one. Judge correctness only.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}} or {{"winner": "B"}}
```

## `PAIR_AR`

```text
أنت حَكَم محايد. فيما يلي سؤال وإجابتان مقترحتان. واحدة فقط من الإجابتين تجيب عن السؤال بشكل صحيح. حدِّد أيّهما. اعتمد في حكمك على الصحة وحدها.

السؤال:
{question}

الإجابة المقترحة A:
{a}

الإجابة المقترحة B:
{b}

أجب بصيغة JSON فقط: {{"winner": "A"}} أو {{"winner": "B"}}
```

## `POINT`

```text
You are an impartial judge. Below is a question and a candidate answer. Decide whether the candidate answers the question correctly. Judge correctness only.

Question:
{question}

Candidate answer:
{a}

Respond with JSON only: {{"verdict": "correct"}} or {{"verdict": "incorrect"}}
```

## `SOLVE`

```text
Answer the following multiple-choice question.

Question:
{question}

{options}

Respond with JSON only: {{"answer": "A"}}, {{"answer": "B"}}, {{"answer": "C"}}, or {{"answer": "D"}}
```

## `REG_TIE`

```text
You are an impartial judge. Below is a question and two candidate answers. Decide which candidate is the better answer to the question. If they are equally good, answer TIE.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}}, {{"winner": "B"}}, or {{"winner": "TIE"}}
```

## `REG_FORCED`

```text
You are an impartial judge. Below is a question and two candidate answers. Decide which candidate is the better answer to the question. You must choose one.

Question:
{question}

Candidate A:
{a}

Candidate B:
{b}

Respond with JSON only: {{"winner": "A"}} or {{"winner": "B"}}
```

## Verbosity transformation (`verbose`, `verbose_gold`)

`VERBOSE_REPEATS = 3`.

```python
def repeat(t):
                    return ". ".join([t.rstrip(". ")] * C.VERBOSE_REPEATS) + ".
```

## Output schemas (structured output)

Every judge is constrained to the JSON schema below (DeepSeek: JSON mode, schema not enforced).

- `solve`: `{"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}}, "required": ["answer"], "additionalProperties": false}`
- `point`: `{"type": "object", "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]}}, "required": ["verdict"], "additionalProperties": false}`
- `reg_tie`: `{"type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B", "TIE"]}}, "required": ["winner"], "additionalProperties": false}`
- `all other conditions`: `{"type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B"]}}, "required": ["winner"], "additionalProperties": false}`

Parsing rule: last JSON verdict in the raw text; `"Candidate A"` is read as `A`. The analysis always re-parses the raw text.
