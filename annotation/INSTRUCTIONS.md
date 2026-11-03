# Annotation instructions

Thank you for helping. You will check Arabic test questions written in your dialect. Each row has one multiple-choice question, shown in English and in the dialect, and two candidate answers in the dialect. The sheet has about 30 rows. Please work alone and do not discuss items with other annotators.

## What you are doing

1. **Evaluate the linguistic and semantic validity of the item.** Your task is to check whether the dialect question says the same thing as the English question, whether it reads as natural dialect, and whether each candidate answer is correct or incorrect for the question.
2. **Do not judge any AI system.** You are not rating a chatbot or a model output. You are rating the test item itself.
3. **Compare with the English.** The English question is the reference. Read it first, then read the dialect question.
4. **Decide each candidate independently.** Rate candidate 1 and candidate 2 separately. Do not decide one from the other.
5. **Do not assume a candidate is correct.** Either candidate, both, or neither may be correct. Their order is random.

You may use a dictionary or look up the subject matter. If you cannot decide, use the "cannot determine" or "uncertain" codes rather than guessing, and explain in `notes`.

## Columns you fill in

| Column | Codes | Meaning |
|---|---|---|
| `question_fidelity` | `2` / `1` / `0` | Does the dialect question mean the same as the English question? **2** = same meaning; **1** = small change that does not change the correct answer; **0** = meaning changed, missing, or wrong, so the correct answer could change. |
| `dialect_naturalness` | `1`–`5` | How natural is the dialect question to a native speaker? **1** = very unnatural; **3** = understandable but awkward; **5** = fully natural. |
| `dialect_authenticity` | `yes` / `mixed` / `no` | Is the question written in your dialect? **yes** = clearly this dialect; **mixed** = this dialect mixed with MSA or another dialect; **no** = mostly MSA or another dialect. |
| `candidate_1_status`, `candidate_2_status` | `definitely_correct` / `definitely_incorrect` / `ambiguous` / `cannot_determine` | Is this candidate a correct answer to the question as written? **definitely_correct** = clearly correct; **definitely_incorrect** = clearly wrong; **ambiguous** = it could be read as correct or wrong (for example, the wording allows two readings); **cannot_determine** = you cannot decide (for example, you lack the subject knowledge or the text is unclear). |
| `grammatical_fit_1`, `grammatical_fit_2` | `yes` / `no` / `uncertain` | Does the candidate fit the question grammatically (for example, it completes the sentence with correct gender, number and form)? |
| `option_reference_problem` | `yes` / `no` | **yes** if a candidate refers to other options that you cannot see here (for example "all of the above", "A and B", "both"), so it cannot be judged on its own. |
| `confidence` | `1`–`5` | Your overall confidence in your answers for this row. **1** = guessing; **5** = certain. |
| `notes` | free text | Optional. Explain any 0, `no`, `ambiguous` or `cannot_determine`, and note anything odd. |
| `annotator_native_dialect` | free text | Your native dialect (for example "Egyptian, Cairo"). Fill in every row. |

## Practical points

- Do not change, delete or reorder rows or the first five columns.
- Use the codes exactly as written (lower case, underscores).
- Fill in every coded column for every row. Leave only `notes` empty if you have nothing to add.
- Save the file with the same name plus your annotator code, for example `annotation_EGY_A1.csv`, and return it.
