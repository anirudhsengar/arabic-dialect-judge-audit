# Do LLM judges grade dialectal Arabic reliably?

Code, prompts, judge outputs and analysis results for an audit of LLM-as-a-judge reliability on Arabic
dialects. A frozen, stratified sample of DialectalArabicMMLU (English, Modern Standard Arabic (MSA),
Emirati, Saudi, Egyptian, Syrian and Moroccan versions of the same items) is judged by eight closed
models from four providers under ten conditions: solving the item itself, pairwise gold-vs-distractor
with an English or an MSA rubric, pointwise correct/incorrect, two repetition probes, a run-to-run
repeat, and three register-preference probes (dialect vs MSA answer). A native-speaker check of a
sample of the dialect renderings is in `annotation/`.

## Contents

| Path | What it is |
|---|---|
| `experiment/ajd/` | Experiment package: sample (`data.py`), prompts and request construction (`conditions.py`), frozen constants and judge settings (`config.py`), API runners (`runner.py`, `batch.py`), analysis (`analyze.py`, `stats.py`, `competence.py`, `register.py`, `report.py`, `features.py`), command line (`__main__.py`) |
| `experiment/scripts/` | Post hoc analyses reported in the paper (read stored outputs only), the annotation sample builder and analysis, and release helpers |
| `experiment/audit/` | Independent re-implementation of the main metrics (does not import the analysis code) |
| `experiment/tests/` | Planted-effects simulators and recovery checks (mock data only) |
| `data/sample.json` | Sample definition: seed, dataset revision, alignment report, key lists, flagged keys |
| `data/sample_items.parquet` | The sampled DialectalArabicMMLU rows (all 7 varieties) with the chosen distractor |
| `data/item_features.parquet` | Per-item lexical-overlap features (`ajd/features.py`) |
| `data/outputs/` | Judge outputs, one row per request (see below) |
| `data/provenance/` | Per-request provenance: served model, time, API route, settings and prompt hashes (see below) |
| `results/analysis/` | All results reported in the paper (`metrics.json` and one JSON per post hoc analysis) and the pointwise figure |
| `annotation/` | Native-speaker check: blank and filled rating sheets, answer key, rater instructions |
| `PROMPTS.md` | All prompt templates, verbatim, with the condition-to-template map and output schemas |
| `DECISION_LOG.md` | Dated design decisions and amendments, judge-selection rule, model inventory |
| `MODEL_SELECTION_LOG.md` | Every candidate judge model considered, with dates, decision and reason |
| `RUN_MANIFEST.json` | Per-judge run summary: provider, route counts, requested and served models, run window, settings, client versions, checksums |

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/). The analysis was run with CPython 3.13
and uv 0.12 on Linux x86_64.

```bash
uv sync --frozen --python 3.13   # installs exactly the locked dependencies from uv.lock
```

Reproducing the analysis needs no API keys and makes no API calls. Re-running the judges needs
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` and `DEEPSEEK_API_KEY` in the environment or in
a `.env` file at the repository root.

## Reproducing the results

Run from `experiment/`:

```bash
cd experiment
uv run --project .. python scripts/unpack_outputs.py     # data/outputs/*.jsonl.gz -> results/<judge>.jsonl
uv run --project .. python -m ajd status                 # coverage per judge: expect 72292/72292
uv run --project .. python -m ajd analyze                # results/analysis/metrics.json, tables/, figures/
uv run --project .. python audit/independent_check.py    # results/analysis/independent_audit.json
uv run --project .. python scripts/<name>.py             # one post hoc analysis (see the table below)
```

`scripts/pointwise_figure.py` reads `pointwise_sdt.json`, and `scripts/gold_holm.py` reads
`pointwise_figure.json` and `identity_check.json`, so run those after the scripts they read.
`scripts/mmmlu_split.py` and `scripts/mmmlu_full_overlap.py` also need two files that are not
redistributed here: `openai/MMMLU` `test/mmlu_AR-XY.csv` (revision `325a01dc`) at
`data/mmmlu/test/mmlu_AR-XY.csv`, and the full DialectalArabicMMLU parquet (revision below) at
`data/raw/data/test-00000-of-00001.parquet`.

Before release, the full sequence above was re-run from a fresh copy of this repository: every value
in `results/analysis/` was reproduced exactly. The exception is `opus_mode_check.json`: the released file
was produced from the raw logs, and a re-run here reports its two failed-attempt counts as `null` and
the request settings as their hash (see below). Figure PDFs embed a creation date, so they are not
byte-identical across runs. `analyze` also writes tables and figures that the paper does not use to
`results/analysis/tables/` and `results/analysis/figures/`; they are not included here.

### Where each result comes from

| Paper | Script | Output (`results/analysis/`) |
|---|---|---|
| Main gaps, probes, competence, register, stability and robustness results; results tables | `python -m ajd analyze` | `metrics.json` |
| Example item table | `scripts/example_item.py` | `example_item.json` |
| Overlap with OpenAI MMMLU; matched vs unmatched keys | `scripts/mmmlu_full_overlap.py`, `scripts/mmmlu_split.py` | `mmmlu_full_overlap.json`, `mmmlu_split.json` |
| Register placebo pairs by kind of surface difference | `scripts/placebo_surface.py` | `placebo_surface.json` |
| Option-referential keys and construct checks | `scripts/construct_checks.py` | `construct_checks.json` |
| Native-speaker check | `scripts/annotation_analysis.py` | `annotation_validation.json` |
| Pointwise figure, gold-minus-solve gap, error ratios | `scripts/pointwise_figure.py` | `pointwise_figure.json`, `figures/pointwise_gold_dist.pdf` |
| Pointwise accuracy identity | `scripts/identity_check.py` | `identity_check.json` |
| Per-judge gold-acceptance tests (Holm) | `scripts/gold_holm.py` | `gold_holm.json` |
| Pointwise signal-detection split | `scripts/pointwise_sdt.py` | `pointwise_sdt.json` |
| Pointwise on keys solved in both varieties | `scripts/pointwise_solved.py` | `pointwise_solved.json` |
| Gold-same / gold-differs strata, native renderings only | `scripts/pointwise_strata.py` | `pointwise_strata.json` |
| Native (EGY, SYR, MAG) vs non-native (KSA, UAE) renderings | `scripts/native_primary.py` | `native_primary.json` |
| Opus standard vs Batch API rows | `scripts/opus_mode_check.py` | `opus_mode_check.json` |
| Independent check of the analysis | `audit/independent_check.py` | `independent_audit.json` |

## Re-running the experiment

```bash
cd experiment
uv run --project .. python -m ajd sample              # download the pinned dataset, check alignment, draw the frozen sample
uv run --project .. python -m ajd count               # requests per condition (per judge)
uv run --project .. python -m ajd pilot <judge> ...   # small fixed slice covering every condition and variety
uv run --project .. python -m ajd run <judge> ...     # live API, resumable (used for Haiku and DeepSeek)
uv run --project .. python -m ajd batch submit|collect|drive <judge> ...   # provider Batch APIs (Opus, GPT, Gemini)
uv run --project .. python -m ajd models              # model IDs reachable with each configured key
```

`pilot`, `run` and `analyze` accept `--mock` (random outputs written only to `results_mock/`).
`pilot`, `run` and `batch` accept `--conditions` to restrict to named conditions. Hosted models change
over time, so re-runs may not match the released outputs exactly.

Tests (mock data only; they never touch `results/`):

```bash
uv run --project .. python tests/simulate.py && uv run --project .. python tests/check.py
uv run --project .. python tests/simulate_extra.py
```

## Dataset

- Source: `MBZUAI/Dialectal-Arabic-MMLU` on the Hugging Face Hub, revision
  `d7ca5ac8dc2bed489ac00bbfaa3656c3a0bd8c85` (`data/test-00000-of-00001.parquet`), CC BY 4.0.
- Quality flags: `IFM/Dialectal-Arabic-MMLU_verified` (keys with any dialect row removed are flagged
  for a robustness subset).
- Sample: seed 2026, 32 domains x 8 = 256 base keys, plus 24 per domain (seed 2027), for 1,024 keys in
  the core conditions; a 128-key subset (first 4 base keys per domain) for the verbosity and repeat
  conditions. Distractor: one wrong option per key, chosen by SHA-256 of the key (same index in every
  variety). Constants are in `experiment/ajd/config.py`.

## Judges

Eight judges, 72,292 requests each, all complete. Exact IDs, providers, API types and parameters are
in `DECISION_LOG.md` (model inventory), `experiment/ajd/config.py` and `RUN_MANIFEST.json`. Every judge
was called through its provider's official API; the route (standard or Batch API) of each row is in
`data/provenance/` (Opus 5.5 and GPT-6 Luna were collected mostly through the standard API, with a
minority of rows through the Batch API; see the release note in `DECISION_LOG.md`). All judges are
constrained to a JSON output schema, except DeepSeek (JSON mode, schema not enforced). Only DeepSeek
runs at temperature 0; run-to-run stability of the others is measured by `pair_en_rep`.

## Judge outputs (`data/outputs/<judge>.jsonl.gz`)

One JSON object per request, last response per `rid` (no duplicates). Fields:

- `rid`: request ID, rebuilt deterministically from `data/sample.json` and `data/sample_items.parquet`
  by `ajd.conditions.build_requests`; this is how each row is joined to its condition, variety, key and
  expected answer
- `judge`: judge name (key of `config.MODELS`)
- `condition`, `variety`, `key`: condition name, variety code (ENG, MSA, UAE, KSA, EGY, SYR, MAG), item key `domain::qid`
- `order` (pairwise conditions): `good_first` or `good_second`
- `cand` (pointwise): `gold` or `wrong`
- `raw`: the model's raw output text
- `parsed`: the verdict extracted at collection time (`A`, `B`, `C`, `D`, `TIE`, `CORRECT`, `INCORRECT`, or `null`)

The analysis always re-parses `raw` with the documented rule. Request parameters are identical for
every row of a judge and are listed in the model inventory.

## Provenance (`data/provenance/<judge>.provenance.jsonl.gz`) and `RUN_MANIFEST.json`

One JSON object per request (same `rid`s as `data/outputs/`), produced from the raw collection logs by
`experiment/scripts/export_provenance.py`. Fields:

- `rid`; `requested_model` (model ID sent); `served_model` (model string returned by the provider)
- `created_at`: UTC time at which the client logged the response
- `finish_reason`: provider stop/finish reason where the client recorded one (Anthropic `stop_reason`,
  Gemini `finishReason`); `null` for OpenAI and DeepSeek, whose finish reason was not logged
- `api_route`: `batch` (row carries a provider batch ID), `standard` (provider metadata without a batch
  ID) or `null` (no provider metadata)
- `batch_id_sha256`: first 16 hex characters of SHA-256 of the provider batch ID (the ID itself is not released)
- `settings_sha256`: SHA-256 of `json.dumps(params, sort_keys=True)` for the row's request parameters
- `prompt_sha256`: SHA-256 of the UTF-8 prompt text rebuilt by `ajd.conditions.build_requests`
- `usage`: provider token counts (integer fields only, nested names joined with `.`)
- `has_provider_metadata`: `false` for rows logged before provider metadata was recorded (all in the
  first minutes of the run on 2026-10-24); their `served_model`, `finish_reason`, `api_route` and
  `usage` are `null`

Rows with provider metadata per judge (of 72,292): gpt-6.1-sol 72,292; gpt-6-luna 67,842; opus-5.5
72,030; haiku-5.5 69,555; gemini-4-argon 72,292; gemini-3.8-flash 72,292; deepseek-v4-pro 66,421;
deepseek-flash 72,030. Every row that has metadata reports the requested model as the served model.

`RUN_MANIFEST.json` summarises these per judge, together with the `openai`, `anthropic` and
`google-genai` versions from `uv.lock` and SHA-256 checksums of `experiment/ajd/config.py`,
`PROMPTS.md`, `data/sample.json`, `data/sample_items.parquet` and each provenance file. The export
script needs the raw logs, which are not released, so it documents the derivation but cannot be re-run
from this repository; the prompt hashes can be checked against `build_requests` directly.

## Differences from the code used for the paper

- Absolute paths are replaced by paths relative to the repository root.
- Comments in `ajd/config.py` that described provider capabilities are replaced by the exact settings
  used (no code change).
- `ajd/analyze.py` writes its tables and figures to `results/analysis/` instead of the manuscript folder.
- `ajd/runner.py` no longer contains a client for a third-party API reseller that was used only for
  earlier runs; those runs were archived and are not analysed (see `MODEL_SELECTION_LOG.md`).
- `scripts/opus_mode_check.py` reads provider metadata from `data/provenance/` instead of the raw logs
  and reports the two failed-attempt counts as `null`, because the failed-attempt log is not released.
  It also reports the request settings as their hash rather than in full.
- Script descriptions are reworded and the script for the pointwise figure is renamed
  `pointwise_figure.py`; no computation changed.
- The tests in `experiment/tests/` are updated to the final eight judges (they were written for an
  earlier line-up) and the coverage-waiver check sets the waiver it tests, as the real waiver set is
  now empty. All tests pass.
- `scripts/unpack_outputs.py` and `scripts/export_provenance.py` were added for the release.

## Citation

Anirudh Sengar. Do LLM Judges Grade Dialectal Arabic Reliably? A Controlled Audit on Parallel,
Objectively Labelled Items. In *Procedia Computer Science* (Proceedings of ACLing 2026, the 8th
International Conference on AI in Computational Linguistics), Elsevier, 2026.

## License

Code: MIT (see `LICENSE`). Content derived from DialectalArabicMMLU (the sample files, prompts filled
with item text, judge outputs that quote it, and the annotation sheets): CC BY 4.0, with attribution
to the DialectalArabicMMLU authors; please cite their paper.
