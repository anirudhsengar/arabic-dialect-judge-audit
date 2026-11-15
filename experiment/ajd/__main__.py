"""CLI: python -m ajd {sample|count|pilot|run|analyze|models} ..."""
import argparse
import asyncio
import json
import os
import random

from dotenv import load_dotenv

from . import config as C


def main():
    load_dotenv(C.ROOT / ".env", override=True)
    ap = argparse.ArgumentParser(prog="ajd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample", help="download pinned data, check alignment, draw frozen sample")
    sub.add_parser("count", help="request counts per condition")
    sub.add_parser("status", help="progress, invalid outputs and logged errors per judge")
    sub.add_parser("models", help="list model IDs available to each configured key")
    for name in ("pilot", "run"):
        p = sub.add_parser(name)
        p.add_argument("judges", nargs="+", choices=list(C.MODELS))
        p.add_argument("--mock", action="store_true")
        p.add_argument("--conditions", nargs="*")
    b = sub.add_parser("batch", help="provider Batch API: submit or collect")
    b.add_argument("action", choices=["submit", "collect", "drive"])
    b.add_argument("judges", nargs="+", choices=list(C.MODELS))
    b.add_argument("--limit", type=int, help="submit at most N requests (for a test batch)")
    b.add_argument("--conditions", nargs="*")
    a = sub.add_parser("analyze")
    a.add_argument("--mock", action="store_true")
    a.add_argument("--allow-incomplete", action="store_true",
                   help="real run only: proceed despite incomplete coverage (results are marked non-final)")
    args = ap.parse_args()

    if args.cmd == "sample":
        from .data import build_sample
        out = build_sample()
        print(json.dumps({k: out[k] for k in ("alignment", "seed", "revision")}, indent=1))
        print(f"keys={len(out['keys'])} verbose_keys={len(out['verbose_keys'])} "
              f"flagged_in_sample={None if out['flagged_keys_in_sample'] is None else len(out['flagged_keys_in_sample'])}")
        return

    if args.cmd == "models":
        for name, cfg in C.MODELS.items():
            key = os.environ.get(cfg["key_env"]) or ("EMPTY" if os.environ.get(cfg.get("base_url_env", "")) else None)
            if not key:
                print(f"{name}: no {cfg['key_env']}")
                continue
            try:
                if cfg["provider"] == "anthropic":
                    from anthropic import Anthropic
                    ids = [m.id for m in Anthropic(api_key=key, base_url=cfg["base_url"]).models.list(limit=100)]
                else:
                    from openai import OpenAI
                    url = os.environ[cfg["base_url_env"]] if "base_url_env" in cfg else cfg["base_url"]
                    ids = sorted(m.id for m in OpenAI(api_key=key, base_url=url).models.list())
                print(f"{name}: {ids}")
            except Exception as e:
                print(f"{name}: ERROR {type(e).__name__}: {e}")
        return

    from .conditions import build_requests
    from .data import load_sample
    meta, items = load_sample()
    reqs = build_requests(meta, items)

    if args.cmd == "count":
        from collections import Counter
        c = Counter(r["condition"] for r in reqs)
        print(dict(c), "total per judge:", sum(c.values()))
        return

    if args.cmd == "status":
        want = {r["rid"] for r in reqs}
        for j in C.MODELS:
            p, e = C.RESULTS_DIR / f"{j}.jsonl", C.RESULTS_DIR / f"{j}.errors.jsonl"
            rows = [json.loads(line) for line in p.open(encoding="utf-8")] if p.exists() else []
            done = {r["rid"] for r in rows} & want
            invalid = sum(r["parsed"] is None for r in rows)
            errs = sum(1 for _ in e.open()) if e.exists() else 0
            print(f"{j:18s} {len(done):6d}/{len(want)} ({100*len(done)/len(want):5.1f}%)  invalid={invalid}  errors_logged={errs}")
        return

    if args.cmd == "analyze":
        from .analyze import analyze
        analyze(mock=args.mock, allow_incomplete=getattr(args, "allow_incomplete", False))
        return

    if args.cmd == "batch":
        from . import batch
        if args.conditions:
            reqs = [r for r in reqs if r["condition"] in args.conditions]
        for j in args.judges:
            if C.MODELS[j].get("batch") is None:
                raise SystemExit(f"{j} has no Batch API configured")
        if args.action == "drive":
            batch.drive(args.judges, reqs)
            return
        for j in args.judges:
            batch.submit(j, reqs, args.limit) if args.action == "submit" else batch.collect(j)
        return

    from .runner import run_judge
    if args.conditions:
        reqs = [r for r in reqs if r["condition"] in args.conditions]
    if args.cmd == "pilot":
        # Small fixed slice covering every condition and variety.
        pilot_keys = set(random.Random(C.SEED).sample(meta["keys"], 6))
        reqs = [r for r in reqs if r["key"] in pilot_keys]
    for j in args.judges:
        asyncio.run(run_judge(j, reqs, mock=args.mock))


if __name__ == "__main__":
    main()
