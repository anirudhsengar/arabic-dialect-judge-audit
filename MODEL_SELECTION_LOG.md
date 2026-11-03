# Model selection log

Every candidate judge model that appears in the experiment specification (frozen 2026-10-24), its
dated amendments A2 and A3 (see `DECISION_LOG.md`), and the internal design-review notes. Dates are
the dates of the specification entry or, where stated, of the logged responses (UTC). `[reseller]` is
a third-party API reseller that was used for some runs before amendment A3; all data collected
through it were archived and not analysed. Archived outputs are not part of this release.

"Outputs viewed" records only what the specification or notes document: "yes" where an entry
describes or reacts to that model's outputs, "not documented" where outputs exist but no entry
refers to them, and "no outputs" where the model was never called.

| # | Candidate | Provider / route | Considered | Outputs viewed before decision | Decision | Decision date | Reason (as recorded) |
|---|---|---|---|---|---|---|---|
| 1 | `claude-opus-5-5` | Anthropic, official API | 2026-10-24 (frozen spec, 7-judge set) | Preliminary full-run results computed before A2 (A2 entry) | Included | 2026-10-24; kept by A2 rule 2026-10-25 | Anthropic flagship/balanced tier |
| 2 | `claude-haiku-5-5` | Anthropic, official API | 2026-10-24 (frozen spec) | Yes: 6-key pilot (change log); preliminary full-run results computed before A2 | Included | 2026-10-24; kept by A2 rule 2026-10-25 | Anthropic small model |
| 3 | `gpt-6-sol` | OpenAI; official API (346 rows, 2026-10-24) then `[reseller]` (18,130 rows, 2026-10-24 to 2026-10-25) | 2026-10-24 (frozen spec) | Not documented | Excluded (superseded); data archived, not analysed | 2026-10-25 (A2) | Replaced by `gpt-6.1-sol` (released 2026-09-29) under the A2 currency rule |
| 4 | `gpt-6-luna` | OpenAI; official API (2026-10-24), `[reseller]` (18,367 rows, 2026-10-24 to 2026-10-25), official API again after A3 | 2026-10-24 (frozen spec) | Yes: 6-key pilot (change log) | Included; only official-API rows analysed, `[reseller]` rows archived | 2026-10-24; route fixed by A3 2026-10-25 | OpenAI small model |
| 5 | `gemini-3.1-pro` | Google via `[reseller]` (12,512 rows, 2026-10-24 to 2026-10-25) | 2026-10-24 (change log: added via `[reseller]`) | Yes: change log notes it sometimes ignores the output enum | Excluded (superseded); data archived, not analysed | 2026-10-25 (A2) | Replaced by `gemini-4-argon` (public 2026-10-13) under the A2 currency rule |
| 6 | `gemini-3.8-flash` | Google; `[reseller]` (13,213 rows, 2026-10-24 to 2026-10-25), then official Batch API after A3 | 2026-10-24 (change log: added via `[reseller]`) | Not documented | Included; only official-API rows analysed, `[reseller]` rows archived | 2026-10-24; route fixed by A3 2026-10-25 | Google small model |
| 7 | `deepseek-flash` | DeepSeek, official API | 2026-10-24 (frozen spec) | Yes: 6-key pilot (change log); preliminary full-run results computed before A2 | Included | 2026-10-24; kept by A2 rule 2026-10-25 | DeepSeek small model |
| 8 | `deepseek-v4-pro` | DeepSeek, official API | 2026-10-24 (first logged responses) | Not documented | Included; 2026-10-24 rows with identical settings reused | 2026-10-25 (A2) | Only Pro model listed by the DeepSeek API (A2) |
| 9 | `gpt-6.1-sol` | OpenAI; `[reseller]` (135 rows, 2026-10-25), then official Batch API after A3 | 2026-10-25 (A2) | `[reseller]` prompt-token overhead measured (A2); verdicts not documented | Included; only official-API rows analysed, `[reseller]` rows archived | 2026-10-25 (A2; route A3) | Newest generally available OpenAI flagship/balanced model |
| 10 | `gemini-4-argon` | Google; `[reseller]` (81 rows, 2026-10-25), then official Batch API after A3 | 2026-10-25 (A2) | `[reseller]` prompt-token overhead measured (A2); verdicts not documented | Included; only official-API rows analysed, `[reseller]` rows archived | 2026-10-25 (A2; route A3) | Newest generally available Google flagship/balanced model |
| 11 | GPT-6 Astra | OpenAI | 2026-10-25 (A2) | No outputs | Excluded | 2026-10-25 (A2) | $10/$50 price tier excluded for cost |
| 12 | Claude Fable 5.1 | Anthropic | 2026-10-25 (A2) | No outputs | Excluded | 2026-10-25 (A2) | $10/$50 price tier excluded for cost |
| 13 | ALLaM-7B, Jais or Nile-Chat (open-weight, local) | Local inference (vLLM) | 2026-10-24 (design review item 14, optional) | No outputs | Not run | Not recorded | Design review made it conditional on a local GPU; no decision entry exists |

Notes

- Amendment A3 (2026-10-25) moved all judges to the providers' official APIs. Which official route
  (standard or Batch API) served each analysed row is recorded per row in `data/provenance/` and
  summarised in `RUN_MANIFEST.json`.
- `[reseller]` row counts above are counts of archived response rows and include any pilot or
  test rows; none of them enter any analysis.
- `pair_ar` rows collected with the earlier MSA rubric wording (A3) are archived for `haiku-5.5`,
  `deepseek-flash`, `deepseek-v4-pro`, `gpt-6-luna` and `opus-5.5` and are not analysed; this is a
  prompt change, not a model-selection change.
