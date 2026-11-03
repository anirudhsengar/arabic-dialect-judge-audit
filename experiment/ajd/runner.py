"""Resumable async runner. Every response is appended to results/<judge>.jsonl with full provenance.

A request counts as done only when it returned a response (parsed or not). API errors are logged
to results/<judge>.errors.jsonl and retried on the next invocation.
"""
import asyncio
import datetime as dt
import json
import os
import random
import re

from . import config as C
from .conditions import output_schema

# Accepts e.g. {"winner": "A"} and the unambiguous {"winner": "Candidate A"} (seen from Gemini 3.1 Pro).
WINNER_RE = re.compile(r'"(?:winner|answer|verdict)"\s*:\s*"(?:candidate\s+)?(A|B|C|D|TIE|correct|incorrect)"', re.I)


def parse(text: str):
    if text is None:
        return None
    found = WINNER_RE.findall(text)
    if found:
        return found[-1].upper()  # last verdict, in case a model writes reasoning first
    t = text.strip().strip('`"\' .').upper()
    return t if t in {"A", "B", "C", "D", "TIE", "CORRECT", "INCORRECT"} else None


def _done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)["rid"] for line in path.open(encoding="utf-8")}


def _usage(u):
    """Provider-reported token usage as a plain dict (for cost tracking and provenance)."""
    if u is None:
        return None
    return u if isinstance(u, dict) else u.model_dump(exclude_none=True)


class MockClient:
    """Pipeline test only. Its outputs live in results_mock/ and must never reach the paper."""
    def __init__(self, cfg):
        self.rng = random.Random(0)

    async def complete(self, prompt, schema):
        await asyncio.sleep(0)
        if '"verdict"' in prompt:
            return '{"verdict": "%s"}' % self.rng.choice(["correct", "incorrect"]), {}
        if '"answer"' in prompt:
            return '{"answer": "%s"}' % self.rng.choice("ABCD"), {}
        choices = ["A", "B", "TIE"] if "TIE" in prompt else ["A", "B"]
        return '{"winner": "%s"}' % self.rng.choice(choices), {}


def _async_openai(cfg):
    from openai import AsyncOpenAI
    base_url = os.environ[cfg["base_url_env"]] if "base_url_env" in cfg else cfg["base_url"]
    return AsyncOpenAI(api_key=os.environ.get(cfg["key_env"], "EMPTY"), base_url=base_url,
                       max_retries=0, timeout=600)


class OpenAICompatClient:
    """Chat Completions API (DeepSeek)."""
    def __init__(self, cfg):
        self.cfg, self.client = cfg, _async_openai(cfg)

    async def complete(self, prompt, schema):
        fmt = ({"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema, "strict": True}}
               if self.cfg.get("structured", "json_schema") == "json_schema" else {"type": "json_object"})
        r = await self.client.chat.completions.create(
            model=self.cfg["model"], messages=[{"role": "user", "content": prompt}],
            response_format=fmt, **self.cfg["params"])
        if not r.choices:  # an error reported as HTTP 200 without choices
            raise RuntimeError(f"no choices: {r.model_extra}")
        extra = r.model_extra or {}
        return r.choices[0].message.content, {"served_model": r.model, "usage": _usage(r.usage),
                                              "credits": extra.get("credits_consumed")}


class AnthropicClient:
    def __init__(self, cfg):
        from anthropic import AsyncAnthropic
        self.cfg = cfg
        self.client = AsyncAnthropic(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"],
                                     max_retries=0, timeout=120)

    async def complete(self, prompt, schema):
        params = dict(self.cfg["params"])
        output_config = {**params.pop("output_config", {}), "format": {"type": "json_schema", "schema": schema}}
        r = await self.client.messages.create(
            model=self.cfg["model"], messages=[{"role": "user", "content": prompt}],
            output_config=output_config, **params)
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, {"served_model": r.model, "usage": _usage(r.usage), "stop": r.stop_reason}


# Explicit official endpoint: an inherited GOOGLE_GEMINI_BASE_URL (local proxy) must not reroute calls.
GEMINI_URL = "https://generativelanguage.googleapis.com/"


def google_config(params, schema):
    """generation_config for a Gemini request (shared by the live client and the Batch API)."""
    return {**params, "response_mime_type": "application/json", "response_json_schema": schema}


def google_text(resp: dict):
    """Answer text (non-thought parts) plus provenance from a GenerateContentResponse dict."""
    cands = resp.get("candidates") or []
    parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    meta = {"served_model": resp.get("modelVersion") or resp.get("model_version"),
            "usage": resp.get("usageMetadata") or resp.get("usage_metadata"),
            "finish": cands[0].get("finishReason") or cands[0].get("finish_reason") if cands else None}
    return text, meta


class GoogleClient:
    """Gemini API (google-genai). Used for pilots; bulk runs go through the Batch API (ajd.batch)."""
    def __init__(self, cfg):
        from google import genai
        self.cfg, self.client = cfg, genai.Client(api_key=os.environ[cfg["key_env"]], http_options={"base_url": GEMINI_URL})

    async def complete(self, prompt, schema):
        r = await self.client.aio.models.generate_content(
            model=self.cfg["model"], contents=prompt, config=google_config(self.cfg["params"], schema))
        return google_text(r.model_dump(mode="json", exclude_none=True))


CLIENTS = {"openai_compat": OpenAICompatClient, "google": GoogleClient,
           "anthropic": AnthropicClient, "mock": MockClient}


async def run_judge(judge: str, requests: list[dict], mock: bool = False, limit: int | None = None):
    cfg = {"provider": "mock", "model": "mock", "params": {}, "concurrency": 64} if mock else C.MODELS[judge]
    if not mock and cfg["model"] == "TBD":
        raise SystemExit(f"Set the exact model ID for {judge} in config.py first.")
    out_dir = C.MOCK_RESULTS_DIR if mock else C.RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    out_path, err_path = out_dir / f"{judge}.jsonl", out_dir / f"{judge}.errors.jsonl"
    done = _done_ids(out_path)
    todo = [r for r in requests if r["rid"] not in done][:limit]
    print(f"[{judge}] {len(done)} done, {len(todo)} to run")
    client = CLIENTS[cfg["provider"]](cfg)
    sem = asyncio.Semaphore(cfg["concurrency"])
    lock = asyncio.Lock()
    interval = 1.0 / cfg["rate"] if cfg.get("rate") else 0.0
    next_slot = [0.0]

    async def pace():
        """Space request starts by `interval` seconds (provider rate limits)."""
        if not interval:
            return
        async with lock:
            loop = asyncio.get_running_loop()
            wait = next_slot[0] - loop.time()
            next_slot[0] = max(next_slot[0], loop.time()) + interval
        if wait > 0:
            await asyncio.sleep(wait)
    stats = {"ok": 0, "err": 0}

    async def one(req):
        async with sem:
            last = None
            for attempt in range(6):
                try:
                    await pace()
                    text, meta = await client.complete(req["prompt"], output_schema(req["condition"]))
                    row = {"rid": req["rid"], "judge": judge, "model": cfg["model"],
                           "params": cfg["params"], "raw": text, "parsed": parse(text), "meta": meta,
                           "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
                    async with lock:
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["ok"] += 1
                    return
                except Exception as e:  # rate limits, 5xx, timeouts
                    last = f"{type(e).__name__}: {e}"
                    status = getattr(e, "status_code", None)
                    if status in (400, 401, 403, 404):
                        break  # not retryable; surface it
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
            async with lock:
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"rid": req["rid"], "error": last}) + "\n")
                stats["err"] += 1

    tasks = [asyncio.create_task(one(r)) for r in todo]
    for i, t in enumerate(asyncio.as_completed(tasks), 1):
        await t
        if i % 500 == 0 or i == len(tasks):
            print(f"[{judge}] {i}/{len(tasks)} ok={stats['ok']} err={stats['err']}")
    return stats
