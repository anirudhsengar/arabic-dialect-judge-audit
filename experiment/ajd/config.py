"""Frozen experiment constants. Changing anything here after the full run invalidates results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MOCK_RESULTS_DIR = ROOT / "results_mock"
PAPER_DIR = ROOT / "paper"

HF_DATASET = "MBZUAI/Dialectal-Arabic-MMLU"
HF_REVISION = "d7ca5ac8dc2bed489ac00bbfaa3656c3a0bd8c85"
HF_FILE = "data/test-00000-of-00001.parquet"
HF_VERIFIED = "IFM/Dialectal-Arabic-MMLU_verified"

SEED = 2026
PER_DOMAIN = 8          # 32 domains x 8 = 256 base keys (all conditions)
EXT_PER_DOMAIN = 24     # +24 per domain = 1,024 keys for the core conditions (see conditions.py)
EXT_SEED = 2027
VERBOSE_PER_DOMAIN = 4  # first 4 sampled keys per domain = 128-key verbosity subset
VERBOSE_REPEATS = 3

VARIETIES = ["ENG", "MSA", "UAE", "KSA", "EGY", "SYR", "MAG"]
DIALECTS = ["UAE", "KSA", "EGY", "SYR", "MAG"]
VARIETY_NAMES = {
    "ENG": "English", "MSA": "MSA", "UAE": "Emirati", "KSA": "Saudi",
    "EGY": "Egyptian", "SYR": "Syrian", "MAG": "Moroccan",
}

# Judge models. Exact IDs are filled in after querying each provider's /models endpoint
# and must be frozen before the full run. base_url is explicit so an inherited proxy
# environment variable never silently reroutes calls.
ANTHROPIC = {"provider": "anthropic", "base_url": "https://api.anthropic.com",
             "key_env": "ANTHROPIC_API_KEY", "concurrency": 16}
# DeepSeek is run with JSON mode (response_format json_object; valid JSON, schema not enforced).
DEEPSEEK = {"provider": "openai_compat", "base_url": "https://api.deepseek.com",
            "key_env": "DEEPSEEK_API_KEY", "concurrency": 16, "structured": "json_object"}

OPENAI = {"provider": "openai_compat", "base_url": "https://api.openai.com/v1",
          "key_env": "OPENAI_API_KEY", "concurrency": 16, "batch": "openai"}
GOOGLE = {"provider": "google", "key_env": "GEMINI_API_KEY", "concurrency": 16, "batch": "google"}

# All judges are constrained to the output JSON schema (structured output), see conditions.output_schema.
# Judges (selection rule): per provider, the newest generally available model in the flagship/balanced
# price tier (~$2-5 per M input tokens) plus its newest small model; the $10/$50 tier (GPT-6 Astra,
# Claude Fable 5.1) is excluded for cost. All judges are called through their providers' own APIs
# (standard or Batch API; the route of every row is in data/provenance/ and RUN_MANIFEST.json).
# Only the DeepSeek judges are run with temperature 0; the others are run without a temperature
# parameter (provider default), so run-to-run stability is measured by the pair_en_rep condition.
MODELS = {
    # Settings used: gpt-6.1-sol reasoning_effort "low"; gpt-6-luna reasoning_effort "none".
    "gpt-6.1-sol": {**OPENAI, "model": "gpt-6.1-sol",
                    "params": {"reasoning_effort": "low", "max_completion_tokens": 2048},
                    "batch_chunk": 6800},  # enqueued-token limit 1.35M at our tier; ~188 tokens/request -> ~1.28M
    # Params identical to the 5,668 rows collected on 2026-10-24 so they remain valid.
    "gpt-6-luna": {**OPENAI, "model": "gpt-6-luna",
                   "params": {"reasoning_effort": "none", "max_completion_tokens": 64},
                   "batch_chunk": 30000},
    "opus-5.5": {**ANTHROPIC, "model": "claude-opus-5-5", "batch": "anthropic",
                 # Opus 5.5 run with adaptive thinking, effort "low", max_tokens 1024.
                 "params": {"max_tokens": 1024, "thinking": {"type": "adaptive"},
                            "output_config": {"effort": "low"}}},
    "haiku-5.5": {**ANTHROPIC, "model": "claude-haiku-5-5",
                  "params": {"max_tokens": 64, "thinking": {"type": "disabled"}}},
    # Gemini 4 Argon / 3.8 Flash run with thinking_level "low", max_output_tokens 4096.
    "gemini-4-argon": {**GOOGLE, "model": "gemini-4-argon",
                       "params": {"thinking_config": {"thinking_level": "low"}, "max_output_tokens": 4096},
                       # Tier-1 enqueued-token limit 5M; pilot: ~380 input tokens/request -> 70% margin.
                       "batch_chunk": 9000},
    "gemini-3.8-flash": {**GOOGLE, "model": "gemini-3.8-flash",
                         "params": {"thinking_config": {"thinking_level": "low"}, "max_output_tokens": 4096},
                         # Tier-1 enqueued-token limit 3M; pilot: ~150 input tokens/request -> 70% margin.
                         "batch_chunk": 14000},
    "deepseek-v4-pro": {**DEEPSEEK, "model": "deepseek-v4-pro",
                        "params": {"temperature": 0, "max_tokens": 1024,
                                   "extra_body": {"thinking": {"type": "disabled"}}}},
    "deepseek-flash": {**DEEPSEEK, "model": "deepseek-flash",
                       "params": {"temperature": 0, "max_tokens": 1024,
                                  "extra_body": {"thinking": {"type": "disabled"}}}},
}
