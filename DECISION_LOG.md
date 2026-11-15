# Decision log

Chronological record of design decisions, taken from the pre-specified experiment specification (frozen 2026-10-24) and its dated amendments, plus the judge-selection rule and the model inventory from `experiment/ajd/config.py`. Spec text is quoted verbatim except for (a) anonymization: the name of the third-party API reseller used for an earlier, archived set of runs is replaced by `[reseller]`; and (b) statements about what a provider's API can or cannot do, which are replaced by the exact settings used. The per-row access route (standard vs Batch API) is documented in `data/provenance/` and `RUN_MANIFEST.json`; see the release note at the end. The list of candidate models is in `MODEL_SELECTION_LOG.md`. Files referenced below that are not part of this release (`notes/`, `results_archive/`) hold internal review notes and archived, non-analysed outputs.

## Timeline

| Date | Entry | Summary |
|---|---|---|
| 2026-10-24 | Spec frozen | Design, conditions, hypotheses H1-H5, statistics and integrity rules fixed before the full run. |
| 2026-10-24 | Change log | Pilot-driven changes (JSON schema, `pair_en_rep`, `verbose_gold`, parsing rule, 1,024-key extension, pointwise condition) before any analysis of full-run data. |
| 2026-10-24 | A1 | Pre-analysis amendment from three design reviews: `reg_forced_msaq`, key-level sign-flip tests, Holm families, equivalence margin, competence controls. |
| 2026-10-24 | A1.1 | Analysis code frozen and validated on planted-effects simulators; H4 two-sided. |
| 2026-10-25 | A2 | Judge lineup fixed by an explicit selection rule (8 judges). |
| 2026-10-25 | A3 | All judges via providers' official APIs; MSA rubric revised and approved by a native speaker; earlier `pair_ar` rows archived. |
| 2026-10-25 | Editorial notes | Access routes aligned with `config.py`; request total 72,292; Holm family sizes 8/40. |
| Release | Provenance note | Per-row access routes from the logged provider metadata (corrects the route statements for Opus 5.5 and GPT-6 Luna). |

## Change log (before any analysis of full-run data)
- 2026-10-24 pilot (6 keys): judges think/explain by default and truncate under small token limits -> all judges constrained to a JSON schema (DeepSeek: JSON mode), per-judge reasoning settings fixed (Opus 5.5 run with adaptive thinking, effort low; Haiku 5.5 and DeepSeek with thinking disabled).
- 2026-10-24: OpenAI gpt-6 and Claude 5.5 judges run without a temperature parameter (provider-default sampling) -> added pair_en_rep to measure run-to-run stability.
- 2026-10-24: pilot showed ceiling under repeated-distractor -> added verbose_gold mirror to separate length bias from a repetition penalty (repetition follows Zheng et al.'s repetitive-list attack).
- Parsing rule: last JSON verdict in the text; `"Candidate A"` is read as `A` (Gemini 3.1 Pro sometimes ignores the enum). Analysis re-parses raw text.
- Judge set: large + small per provider. Gemini 3.1 Pro / 3.8 Flash added via [reseller] (run with thinking effort "low").
- 2026-10-24 (before any analysis of full-run data): sample extended to 1,024 keys for core conditions; pointwise condition added. Already-collected rows are unaffected (request IDs depend only on prompts).
- 2026-10-24 (editorial, no design change): spec text aligned with config.py and the extended sample (reasoning settings, 1,024-key bootstrap, 32 keys/domain, pointwise Holm family). H4's cross-variety part has no pre-specified test and is reported descriptively via CIs.

## Pre-analysis amendment A1 (2026-10-24, before any full-run outcome was examined)
Source: area-chair synthesis of three design reviews, `notes/design_review.md` (items cited as DR#). At this point ~25% of calls were collected and no full-run metric had been computed or viewed.
- New condition (DR2): `reg_forced_msaq` — reg_forced pairs with the question in MSA. Contrast with reg_forced separates "prefers MSA" from "prefers the candidate matching the question's register".
- Opus: `pair_ar` deferred until the native-speaker rubric check (DR1); if the rubric changes, `pair_ar` is re-run for all judges.
- Test unit = key (DR3a): per key, d_k = mean over orders of (correct_v − correct_MSA); exact sign-flip permutation test (10,000 flips, seed 2026). Same for rubric language, pointwise (mean over the two candidates) and register (per-key MSA share vs 0.5). Order-level McNemar kept only as a sensitivity check.
- Holm families (DR3b): primary H1/H2 = one test per judge on the mean dialect gap over the 5 dialects (7 tests; H2 one-sided: dialect < MSA); secondary = the 35 judge × dialect cells. ENG − MSA reported as context, outside every family. H3 primary = per-judge mean over 7 varieties (7 tests); H5 mirrors H1.
- Equivalence (DR3c): margin δ = ±2 pp; a primary gap is "equivalent" if its 90% key-bootstrap CI lies within ±δ, "different" if the Holm-adjusted test rejects, else "inconclusive".
- Scale (DR3d): report error ratio (1 − acc_v)/(1 − acc_MSA) and log-odds difference next to every pp gap.
- H4 (DR3e): confirmatory = per-judge repetition effect averaged over varieties (7 tests, Holm); cross-variety differences exploratory. MDE per condition reported (DR3f).
- Missing data (DR4): transport errors are missing, not wrong, and are re-requested; analysis asserts coverage and reports it. Invalid rates per judge × variety × condition; primary gaps also reported complete-case and with invalid = 0.5.
- Judging vs competence (DR5), headline analysis: per judge × variety (a) gap on keys solved in both the variety and MSA; (b) per-key difference-in-differences (pair_v − pair_MSA) − (solve_v − solve_MSA) with sign-flip test; (c) P(judge wrong | solved) per variety; (d) "not solved" split into "picked our distractor" vs "picked another option"; (e) GEE correct ~ variety × solved, clustered by key. A judge-specific dialect effect is claimed only if (a)/(b) show it.
- Register clean-up (DR6): normalised-string rule (hamza/alif variants, ta marbuta, alif maqsura, diacritics, tatweel, punctuation, whitespace); orthography-only pairs are a placebo (expect TIE ≈ 1, P(MSA) ≈ 0.5); main estimates on substantively different pairs; P(MSA | non-tie); key-clustered logistic regression of chose-MSA on log length ratio and character similarity with dialect factor, reporting P(MSA) at length parity; cross-dialect comparison on the common key subset.
- Strata (DR7): gaps reported separately for keys whose both candidates are identical to MSA (question-only effect) and the rest; dialect ranking only on the latter; dose-response on q_char_sim in drivers().
- Bootstrap (DR9): domain-stratified multinomial weights shared across judges/varieties/conditions; separate stratified weights for the 128-key subset; CI for stability.
- Judge comparisons are descriptive only (DR11): judges differ in reasoning mode, sampling and system prompt; per-judge noise floor = 1 − stability.

## Amendment A1.1 (2026-10-24, analysis code frozen; still before any full-run outcome was examined)
- Implementation: `ajd/analyze.py`, `stats.py`, `competence.py`, `register.py`, `report.py`; validated on an independent planted-effects simulator (`tests/simulate.py` + `check.py`: 71/71; `tests/simulate_extra.py`: 24/24).
- H4 is tested two-sided (the direction of a repetition effect is not assumed; the 6-key pilot hinted at the opposite direction, so a one-sided test would be inappropriate).
- Orthographic normaliser does not fold the hamza seats ؤ/ئ (conservative: fewer pairs treated as placebo).
- Classification precedence: "equivalent" (90% CI within ±2 pp) is checked before "different" (Holm rejection); both flags are stored.
- Reporting rule: every H1/H5 primary result is shown next to its complete-case estimate and per-variety invalid rates; a gap that disappears under complete-case is described as a difference in output validity, not in judging accuracy.

## Amendment A2 (2026-10-25): judge lineup by explicit rule
- Rule: per provider, the newest generally available model in the flagship/balanced tier (~$2-5 per M input tokens) plus its newest small model; the $10/$50 tier (GPT-6 Astra, Claude Fable 5.1) is excluded for cost.
- Lineup (8): gpt-6.1-sol, gpt-6-luna ([reseller] Responses); gemini-4-argon, gemini-3.8-flash ([reseller] chat); claude-opus-5-5, claude-haiku-5-5 (Anthropic API); deepseek-v4-pro, deepseek-flash (DeepSeek API; v4-pro is the only Pro model the API lists).
- Superseded and archived (not analysed): gpt-6-sol (replaced by gpt-6.1-sol, released 2026-09-29) and gemini-3.1-pro (replaced by gemini-4-argon, public 2026-10-13). deepseek-v4-pro rows collected on 2026-10-24 with identical settings are reused.
- Measured hidden prompt overhead on [reseller]: gpt-6.1-sol 4,216 instruction tokens (Codex); gemini-4-argon ~164 tokens ("Say OK." = 166 prompt tokens vs 2 on gemini-3.8-flash). Disclosed in the paper.
- This change was made after preliminary results for haiku/deepseek-flash/opus had been computed; it does not depend on them (it corrects model currency) and no analysis choices changed.

## Amendment A3 (2026-10-25): access routes and MSA rubric
- All judges are called through their providers' official APIs (OpenAI and Google via their Batch APIs; Anthropic and DeepSeek via their standard APIs). All [reseller] data is archived and not analysed. gemini-4-argon / gemini-3.8-flash: thinking_level "low"; gpt-6.1-sol: reasoning_effort "low"; gpt-6-luna: reasoning_effort "none".
- MSA rubric (PAIR_AR) revised for fidelity before any judge's pair_ar data was analysed: "answers the question correctly" restored, "judge" (حَكَم) for "evaluator", "correctness only" disambiguated. A native Arabic speaker reviewed the revised wording and approved it (meaning preserved; only stylistic shift: "candidate" -> "proposed"). pair_ar rows collected with the earlier wording (haiku, deepseek-flash, deepseek-v4-pro, and pilot rows) are archived in results_archive/pair_ar_old_wording and not analysed; pair_ar is run with the approved wording for all judges.

## Editorial note (2026-10-25, no design change)
- Access routes, aligned with `config.py`: Opus 5.5 was called via the Anthropic Message Batches API; Haiku 5.5 via the standard Anthropic API; DeepSeek via its standard API; OpenAI and Google via their Batch APIs (A3's "Anthropic ... via their standard APIs" applies to Haiku only).
- The per-judge request total after A1 (DR2, `reg_forced_msaq`) is 72,292, not 66,712.

## Editorial note (2026-10-25, no design change): Holm family sizes
- A1 states the Holm families for 7 judges (7 primary tests, 35 cells). When A2 fixed the lineup at 8 judges, every per-judge primary family became 8 tests and the H1/H5 secondary cell families 40 judge × dialect cells (H3 secondary: 56 judge × variety cells); the analysis code (`ajd/analyze.py`) and the paper use 8/40. Earlier entries are left unedited.

## Judge selection rule (from `config.py`)

```python
# Judges (selection rule): per provider, the newest generally available model in the flagship/balanced
# price tier (~$2-5 per M input tokens) plus its newest small model; the $10/$50 tier (GPT-6 Astra,
# Claude Fable 5.1) is excluded for cost. All judges are called through their providers' own APIs
# (standard or Batch API; the route of every row is in data/provenance/ and RUN_MANIFEST.json).
# Only the DeepSeek judges are run with temperature 0; the others are run without a temperature
# parameter (provider default), so run-to-run stability is measured by the pair_en_rep condition.
```

## Model inventory (from `config.py`)

Collection dates are UTC dates of the logged responses (row counts in parentheses). Every judge has all 72,292 requests.

| Judge | Model ID | Provider | API type | Params (exact) | Collected |
|---|---|---|---|---|---|
| `gpt-6.1-sol` | `gpt-6.1-sol` | OpenAI | OpenAI Batch API, `/v1/chat/completions` bodies, `response_format` json_schema (strict) | `{"reasoning_effort": "low", "max_completion_tokens": 2048}` | 2026-10-25 (72,292) |
| `gpt-6-luna` | `gpt-6-luna` | OpenAI | OpenAI `/v1/chat/completions`, `response_format` json_schema (strict); per-row route: standard 67,822, Batch API 20, not recorded 4,450 | `{"reasoning_effort": "none", "max_completion_tokens": 64}` | 2026-10-24 (4,450), 2026-10-25 (67,842) |
| `opus-5.5` | `claude-opus-5-5` | Anthropic | Anthropic Messages API (structured output); per-row route: standard 67,696, Message Batches API 4,334, not recorded 262 | `{"max_tokens": 1024, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}` | 2026-10-24 (57,956), 2026-10-25 (14,336) |
| `haiku-5.5` | `claude-haiku-5-5` | Anthropic | Anthropic Messages API (standard; structured output); not recorded 2,737 | `{"max_tokens": 64, "thinking": {"type": "disabled"}}` | 2026-10-24 (57,956), 2026-10-25 (14,336) |
| `gemini-4-argon` | `gemini-4-argon` | Google | Gemini Batch API (generateContent; `response_json_schema`) | `{"thinking_config": {"thinking_level": "low"}, "max_output_tokens": 4096}` | 2026-10-25 (72,292) |
| `gemini-3.8-flash` | `gemini-3.8-flash` | Google | Gemini Batch API (generateContent; `response_json_schema`) | `{"thinking_config": {"thinking_level": "low"}, "max_output_tokens": 4096}` | 2026-10-25 (72,292) |
| `deepseek-v4-pro` | `deepseek-v4-pro` | DeepSeek | DeepSeek chat completions (OpenAI-compatible), JSON mode (`json_object`; schema not enforced) | `{"temperature": 0, "max_tokens": 1024, "extra_body": {"thinking": {"type": "disabled"}}}` | 2026-10-24 (5,871), 2026-10-25 (66,421) |
| `deepseek-flash` | `deepseek-flash` | DeepSeek | DeepSeek chat completions (OpenAI-compatible), JSON mode (`json_object`; schema not enforced) | `{"temperature": 0, "max_tokens": 1024, "extra_body": {"thinking": {"type": "disabled"}}}` | 2026-10-24 (57,956), 2026-10-25 (14,336) |

Endpoints: Anthropic `https://api.anthropic.com`; OpenAI `https://api.openai.com/v1`; Google `https://generativelanguage.googleapis.com/`; DeepSeek `https://api.deepseek.com`. Only DeepSeek runs at temperature 0; the others are run without a temperature parameter (provider-default sampling). Reasoning settings used: Opus 5.5 adaptive thinking, effort low; Gemini 4 Argon and Gemini 3.8 Flash thinking_level low; GPT-6.1 Sol reasoning_effort low; GPT-6 Luna reasoning_effort none; Haiku 5.5 and both DeepSeek models thinking disabled.

## Release note: access routes from the logged provider metadata

Derived by `experiment/scripts/export_provenance.py` from the raw collection logs (per-row values in `data/provenance/`, totals in `RUN_MANIFEST.json`). A row counts as "Batch API" if it carries a provider batch ID, "standard" if it carries provider metadata without one, and "not recorded" if it was logged without provider metadata (all such rows are dated 2026-10-24).

| Judge | Standard API | Batch API | Not recorded |
|---|---|---|---|
| `gpt-6.1-sol` | 0 | 72,292 | 0 |
| `gpt-6-luna` | 67,822 | 20 | 4,450 |
| `opus-5.5` | 67,696 | 4,334 | 262 |
| `haiku-5.5` | 69,555 | 0 | 2,737 |
| `gemini-4-argon` | 0 | 72,292 | 0 |
| `gemini-3.8-flash` | 0 | 72,292 | 0 |
| `deepseek-v4-pro` | 66,421 | 0 | 5,871 |
| `deepseek-flash` | 72,030 | 0 | 262 |

The 2026-10-25 editorial note above ("Opus 5.5 was called via the Anthropic Message Batches API"; "OpenAI and Google via their Batch APIs") does not match these counts for Opus 5.5 and GPT-6 Luna, which were mostly collected through the standard API. Request parameters are identical across all rows of each judge, whichever route was used. The 4,450 GPT-6 Luna rows without provider metadata are identical (same request ID, timestamp and raw text) to rows in the archived official-API set from 2026-10-24.
