"""Provider Batch APIs (OpenAI, Google) for bulk judge runs at 50% cost.

submit:  writes the not-yet-done requests of a judge as JSONL, uploads, creates batch jobs, and
         records them in results/batches/<judge>.json (job ids are never re-submitted blindly).
collect: polls jobs; for finished jobs downloads outputs and appends rows to results/<judge>.jsonl
         in the same format as the live runner. Failed requests are left missing and are picked
         up by the next submit.
"""
import datetime as dt
import json
import os

from . import config as C
from .conditions import output_schema
from .runner import GEMINI_URL, google_config, google_text, parse

OPENAI_MAX = 50_000  # requests per OpenAI batch


def _paths(judge):
    d = C.RESULTS_DIR / "batches"
    d.mkdir(parents=True, exist_ok=True)
    return C.RESULTS_DIR / f"{judge}.jsonl", d / f"{judge}.json", d


def _done(out_path):
    if not out_path.exists():
        return set()
    return {json.loads(line)["rid"] for line in out_path.open(encoding="utf-8")}


def _jobs(state_path):
    return json.loads(state_path.read_text()) if state_path.exists() else []


def _pending_rids(jobs):
    return {rid for j in jobs if j["state"] not in ("collected", "failed", "expired", "cancelled")
            for rid in j["rids"]}


def submit(judge, requests, limit=None):
    cfg = C.MODELS[judge]
    out_path, state_path, d = _paths(judge)
    jobs = _jobs(state_path)
    skip = _done(out_path) | _pending_rids(jobs)
    if cfg.get("batch_chunk") and _pending_rids(jobs):
        print(f"[{judge}] a chunk is still pending (enqueued-token limit); not submitting more")
        return
    todo = [r for r in requests if r["rid"] not in skip][:limit or cfg.get("batch_chunk")]
    if not todo:
        print(f"[{judge}] nothing to submit")
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    chunks = [todo[i:i + OPENAI_MAX] for i in range(0, len(todo), OPENAI_MAX)] if cfg["batch"] == "openai" else [todo]
    for n, chunk in enumerate(chunks):
        if cfg["batch"] == "anthropic":
            job_id = _create_anthropic(cfg, chunk)
            jobs.append({"id": job_id, "file": None, "state": "submitted", "n": len(chunk),
                         "rids": [r["rid"] for r in chunk], "submitted": stamp})
            state_path.write_text(json.dumps(jobs))
            print(f"[{judge}] submitted {job_id} with {len(chunk)} requests")
            continue
        f = d / f"{judge}.{stamp}.{n}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for r in chunk:
                schema = output_schema(r["condition"])
                if cfg["batch"] == "openai":
                    line = {"custom_id": r["rid"], "method": "POST", "url": "/v1/chat/completions",
                            "body": {"model": cfg["model"], "messages": [{"role": "user", "content": r["prompt"]}],
                                     "response_format": {"type": "json_schema", "json_schema":
                                                         {"name": "verdict", "schema": schema, "strict": True}},
                                     **cfg["params"]}}
                else:
                    line = {"key": r["rid"], "request": {
                        "contents": [{"role": "user", "parts": [{"text": r["prompt"]}]}],
                        "generation_config": google_config(cfg["params"], schema)}}
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        job_id = _create(cfg, f)
        jobs.append({"id": job_id, "file": f.name, "state": "submitted", "n": len(chunk),
                     "rids": [r["rid"] for r in chunk], "submitted": stamp})
        state_path.write_text(json.dumps(jobs))
        print(f"[{judge}] submitted {job_id} with {len(chunk)} requests")


def _anthropic(cfg):
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"])


def _create_anthropic(cfg, chunk):
    reqs = []
    for r in chunk:
        params = dict(cfg["params"])
        oc = {**params.pop("output_config", {}),
              "format": {"type": "json_schema", "schema": output_schema(r["condition"])}}
        reqs.append({"custom_id": r["rid"], "params": {
            "model": cfg["model"], "messages": [{"role": "user", "content": r["prompt"]}],
            "output_config": oc, **params}})
    return _anthropic(cfg).messages.batches.create(requests=reqs).id


def _create(cfg, path):
    if cfg["batch"] == "openai":
        from openai import OpenAI
        cl = OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"])
        up = cl.files.create(file=path.open("rb"), purpose="batch")
        return cl.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                 completion_window="24h").id
    from google import genai
    from google.genai import types
    cl = genai.Client(api_key=os.environ[cfg["key_env"]], http_options={"base_url": GEMINI_URL})
    up = cl.files.upload(file=str(path), config=types.UploadFileConfig(display_name=path.stem, mime_type="jsonl"))
    return cl.batches.create(model=cfg["model"], src=up.name, config={"display_name": path.stem}).name


def collect(judge):
    cfg = C.MODELS[judge]
    out_path, state_path, _ = _paths(judge)
    jobs = _jobs(state_path)
    for job in jobs:
        if job["state"] in ("collected", "failed", "expired", "cancelled"):
            continue
        state, lines = _fetch(cfg, job["id"])
        job["state"] = state
        if lines is not None:
            done = _done(out_path)
            ok = bad = 0
            with out_path.open("a", encoding="utf-8") as fh:
                for rid, text, meta, err in lines:
                    if err is not None or rid in done:
                        bad += err is not None
                        continue
                    row = {"rid": rid, "judge": judge, "model": cfg["model"], "params": cfg["params"],
                           "raw": text, "parsed": parse(text), "meta": {**meta, "batch_id": job["id"]},
                           "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ok += 1
            job["state"] = "collected"
            job["collected"] = {"ok": ok, "errors": bad}
            job["collected_at"] = __import__("time").time()
        print(f"[{judge}] {job['id']}: {job['state']} {job.get('collected', '')}")
        state_path.write_text(json.dumps(jobs))


def _fetch(cfg, job_id):
    """Returns (state, lines|None); lines are (rid, text, meta, error)."""
    if cfg["batch"] == "anthropic":
        cl = _anthropic(cfg)
        b = cl.messages.batches.retrieve(job_id)
        if b.processing_status != "ended":
            c = b.request_counts
            return f"{b.processing_status} ok={c.succeeded} errored={c.errored} processing={c.processing}", None
        lines = []
        for item in cl.messages.batches.results(job_id):
            res = item.result
            if res.type != "succeeded":
                lines.append((item.custom_id, None, None, getattr(res, "error", res.type)))
                continue
            msg = res.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            lines.append((item.custom_id, text, {"served_model": msg.model, "stop": msg.stop_reason,
                                                 "usage": msg.usage.model_dump(exclude_none=True)}, None))
        return "completed", lines
    if cfg["batch"] == "openai":
        from openai import OpenAI
        cl = OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"])
        b = cl.batches.retrieve(job_id)
        if b.status in ("failed", "expired", "cancelled"):
            print(f"  errors: {b.errors}")
            return b.status, None
        if b.status != "completed":
            rc = b.request_counts
            return f"{b.status} {rc.completed}/{rc.total} failed={rc.failed}" if rc else b.status, None
        lines = []
        for fid in (b.output_file_id, b.error_file_id):
            if not fid:
                continue
            for raw in cl.files.content(fid).text.splitlines():
                o = json.loads(raw)
                resp = o.get("response") or {}
                if o.get("error") or resp.get("status_code") != 200:
                    lines.append((o["custom_id"], None, None, o.get("error") or resp.get("body")))
                    continue
                body = resp["body"]
                text = body["choices"][0]["message"]["content"]
                lines.append((o["custom_id"], text, {"served_model": body.get("model"), "usage": body.get("usage")}, None))
        return "completed", lines
    from google import genai
    cl = genai.Client(api_key=os.environ[cfg["key_env"]], http_options={"base_url": GEMINI_URL})
    b = cl.batches.get(name=job_id)
    state = b.state.name if hasattr(b.state, "name") else str(b.state)
    if state in ("JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"):
        print(f"  error: {b.error}")
        return state.removeprefix("JOB_STATE_").lower(), None
    if state != "JOB_STATE_SUCCEEDED":
        return state, None
    lines = []
    for raw in cl.files.download(file=b.dest.file_name).decode("utf-8").splitlines():
        o = json.loads(raw)
        if "response" not in o:
            lines.append((o.get("key"), None, None, o.get("error") or o.get("status") or "no response"))
            continue
        text, meta = google_text(o["response"])
        lines.append((o["key"], text, meta, None))
    return "completed", lines


def drive(judges, requests, poll=180):
    """Collect finished jobs and submit next chunks until every judge has all requests."""
    import time
    while True:
        remaining = 0
        for j in judges:
            collect(j)
            out_path, state_path, _ = _paths(j)
            left = len({r["rid"] for r in requests} - _done(out_path))
            remaining += left
            jobs = _jobs(state_path)
            last = jobs[-1] if jobs else None
            # Back off 30 min after a chunk in which every request failed (provider-side rejection).
            if last and last.get("collected", {}).get("ok") == 0 and \
                    time.time() - last.get("collected_at", 0) < 1800:
                continue
            if left and not _pending_rids(jobs):
                submit(j, requests)
        print(f"[drive] {remaining} requests outstanding", flush=True)
        if not remaining:
            return
        time.sleep(poll)
