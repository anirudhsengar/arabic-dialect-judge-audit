"""Export per-row provenance and a run manifest from the raw judge outputs (read-only on results/).

For every judge in config.MODELS this reads results/<judge>.jsonl (the raw collection logs, which
carry provider metadata) and writes

  <out>/data/provenance/<judge>.provenance.jsonl.gz   one row per request (last row per rid)
  <out>/RUN_MANIFEST.json                             per-judge run summary

Provenance row fields:
  rid                    request ID (ajd.conditions.build_requests)
  requested_model        model ID sent in the request (row field `model`)
  served_model           model string returned by the provider (meta.served_model) or null
  created_at             UTC timestamp written by the client when the response was logged (row `ts`)
  finish_reason          provider stop/finish reason (meta.stop or meta.finish) or null if not recorded
  api_route              "batch" if the row carries a provider batch ID, "standard" if it carries
                         provider metadata without one, null if the row has no provider metadata
  batch_id_sha256        first 16 hex chars of sha256(batch ID), or null
  settings_sha256        sha256 of json.dumps(params, sort_keys=True) (request parameters of the row)
  prompt_sha256          sha256 of the UTF-8 prompt text rebuilt by ajd.conditions.build_requests
  usage                  token counts reported by the provider (integer fields only, flattened), or null
  has_provider_metadata  false for rows logged before provider metadata was recorded

The raw logs with provider metadata are not part of the anonymized release, so this script cannot be
re-run from the release package; it is included to document how the provenance files were produced.

Usage (from experiment/):
  uv run --project .. python scripts/export_provenance.py --out <package root>
"""
import argparse
import collections
import gzip
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from ajd import config as C  # noqa: E402
from ajd.conditions import build_requests  # noqa: E402
from ajd.data import load_sample  # noqa: E402

N_EXPECTED = 72292
PROVIDER = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "deepseek": "DeepSeek"}
ENDPOINT = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1",
            "google": "https://generativelanguage.googleapis.com/", "deepseek": "https://api.deepseek.com"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def provider_of(cfg) -> str:
    if cfg["provider"] == "anthropic":
        return "anthropic"
    if cfg["provider"] == "google":
        return "google"
    return "deepseek" if "deepseek" in cfg.get("base_url", "") else "openai"


def token_counts(u, prefix=""):
    """Integer token counts from a provider usage object, flattened with '.'; strings dropped."""
    out = {}
    if isinstance(u, dict):
        for k, v in u.items():
            name = f"{prefix}{k}"
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                out[name] = v
            elif isinstance(v, dict):
                out.update(token_counts(v, name + "."))
            elif isinstance(v, list):  # Gemini: [{"modality": "TEXT", "tokenCount": n}, ...]
                for it in v:
                    if isinstance(it, dict) and "modality" in it and isinstance(it.get("tokenCount"), int):
                        out[f"{name}.{it['modality']}"] = it["tokenCount"]
    return out


def lock_versions(lock: Path, names):
    data = tomllib.loads(lock.read_text())
    return {p["name"]: p["version"] for p in data["package"] if p["name"] in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, required=True, help="package root to write into")
    args = ap.parse_args()

    meta, items = load_sample()
    reqs = build_requests(meta, items)
    prompt_sha = {r["rid"]: sha(r["prompt"].encode("utf-8")) for r in reqs}
    assert len(prompt_sha) == N_EXPECTED, len(prompt_sha)

    out_dir = args.out / "data" / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = lock_versions(args.out / "uv.lock", {"openai", "anthropic", "google-genai"})
    manifest = {
        "description": "Per-judge run summary derived from the raw collection logs by "
                       "experiment/scripts/export_provenance.py. Timestamps are client-side UTC log times. "
                       "api_route counts come from the per-row provider metadata; rows logged before "
                       "metadata was recorded are counted under 'unknown (no provider metadata)'.",
        "n_expected_per_judge": N_EXPECTED,
        "client_package_versions": versions,
        "config_sha256": sha((args.out / "experiment/ajd/config.py").read_bytes()),
        "prompts_sha256": sha((args.out / "PROMPTS.md").read_bytes()),
        "sample_sha256": {"data/sample.json": sha((args.out / "data/sample.json").read_bytes()),
                          "data/sample_items.parquet": sha((args.out / "data/sample_items.parquet").read_bytes())},
        "judges": {},
    }

    for judge, cfg in C.MODELS.items():
        rows = {}
        with (args.results / f"{judge}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["rid"]] = r  # last row per rid
        unknown = set(rows) - set(prompt_sha)
        assert not unknown, f"{judge}: {len(unknown)} rids not in build_requests"

        served, routes, finish, settings = (collections.Counter() for _ in range(4))
        n_meta = 0
        ts = []
        prov = []
        for rid in sorted(rows, key=lambda k: (rows[k]["ts"], k)):
            r = rows[rid]
            m = r.get("meta") or None
            has = bool(m)
            n_meta += has
            bid = (m or {}).get("batch_id")
            route = None if not has else ("batch" if bid else "standard")
            fin = (m or {}).get("stop") or (m or {}).get("finish") if has else None
            usage = token_counts((m or {}).get("usage")) if has and (m or {}).get("usage") else None
            sset = json.dumps(r["params"], sort_keys=True)
            prov.append({
                "rid": rid, "requested_model": r["model"],
                "served_model": (m or {}).get("served_model") if has else None,
                "created_at": r["ts"], "finish_reason": fin, "api_route": route,
                "batch_id_sha256": sha(bid.encode())[:16] if bid else None,
                "settings_sha256": sha(sset.encode()), "prompt_sha256": prompt_sha[rid],
                "usage": usage or None, "has_provider_metadata": has,
            })
            served[(m or {}).get("served_model") if has else None] += 1
            routes[route or "unknown (no provider metadata)"] += 1
            finish[str(fin)] += 1
            settings[sset] += 1
            ts.append(r["ts"])

        path = out_dir / f"{judge}.provenance.jsonl.gz"
        with gzip.GzipFile(path, "wb", mtime=0) as g:
            for p in prov:
                g.write((json.dumps(p, ensure_ascii=False) + "\n").encode("utf-8"))

        pv = provider_of(cfg)
        used = {k for k in routes if k in ("standard", "batch")}
        endpoint = {frozenset({"standard"}): "official API (standard)",
                    frozenset({"batch"}): "official API (Batch API)",
                    frozenset({"standard", "batch"}): "official API (standard + Batch API)"}[frozenset(used)]
        assert len(settings) == 1, f"{judge}: settings vary across rows"
        manifest["judges"][judge] = {
            "provider": PROVIDER[pv],
            "endpoint": endpoint,
            "base_url": ENDPOINT[pv],
            "api_route_counts": dict(routes),
            "requested_model": cfg["model"],
            "served_model_counts": {str(k) if k is not None else "null (no provider metadata)": v
                                    for k, v in served.items()},
            "finish_reason_counts": {k if k != "None" else "null (not recorded)": v for k, v in finish.items()},
            "run_start": min(ts), "run_end": max(ts),
            "settings": cfg["params"],
            "settings_sha256": sha(next(iter(settings)).encode()),
            "settings_match_config": json.loads(next(iter(settings))) == cfg["params"],
            "response_format": cfg.get("structured", "json_schema"),
            "n_expected": N_EXPECTED, "n_received": len(rows),
            "n_with_provider_metadata": n_meta,
            "provenance_file": f"data/provenance/{judge}.provenance.jsonl.gz",
            "provenance_file_sha256": sha(path.read_bytes()),
        }
        print(f"{judge}: {len(rows)} rows, {n_meta} with provider metadata, routes {dict(routes)}")

    (args.out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
