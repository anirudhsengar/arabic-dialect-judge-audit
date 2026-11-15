# Native-speaker check

Blinded check of the dialect renderings by native speakers. One rater each for Egyptian (EGY), Syrian
(SYR), Saudi (KSA) and Emirati (UAE) completed the check; no Moroccan (MAG) rater was available, so the
MAG sheet was not rated.

| Path | What it is |
|---|---|
| `INSTRUCTIONS.md` | Instructions given to raters |
| `sheets/annotation_<DIALECT>.csv` | Blank sheets as given to raters: 30 rows per dialect (24 primary + 6 diagnostic, in random order), without the gold label |
| `filled/annotation_<DIALECT>_A1.csv` | Completed sheets. Raters filled the same fields through a form; the only personal information recorded is the rater's native dialect |
| `key/answer_key.csv` | For each row: item key, domain, dialect, stratum, which candidate is the gold, English gold and distractor, item features (never shown to raters) |
| `key/composition.json` | Sample composition |

Built by `experiment/scripts/annotation_build.py` (seed 20261101). Analysed by
`experiment/scripts/annotation_analysis.py`, which writes `results/analysis/annotation_validation.json`:

```bash
cd experiment && uv run --project .. python scripts/annotation_analysis.py
```

## Sample

- Primary (120 cells): 24 (key, dialect) cells per dialect from the 1,024-key sample, one key from each
  of 24 different domains, 12 cells whose dialect gold string equals the MSA gold string and 12 whose
  string differs (UAE: 11 and 13, as one domain had no key of the first type). Drawn before any model
  output was loaded.
- Diagnostic (30 cells): 6 per dialect: 2 where the MSA gold was accepted but the dialect gold was
  rejected by at least 3 of the 8 judges (pointwise), 2 with the lowest dialect-MSA question word overlap
  (bottom 5%), 1 possible grammatical mismatch between question and candidate, and 1 candidate that
  refers to other options. Diagnostic cells are reported descriptively only.
