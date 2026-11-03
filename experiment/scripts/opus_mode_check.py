"""Opus pair_ar: standard (synchronous Messages API) vs Message Batches API rows (post hoc, descriptive).

After the Anthropic credit balance ran out during the Opus pair_ar run, the remaining pair_ar requests
(the last ones in request order; results/opus-5.5.errors.jsonl holds the failed synchronous attempts)
were resubmitted through the Message Batches API with identical model and params; those rows carry
meta.batch_id in results/opus-5.5.jsonl. Reads stored outputs only (no API calls).

(1) counts of pair_ar rows by mode, by variety and by key position in request order;
(2) mode check. Mode is confounded with key (whole keys fall in one mode, except one boundary key), so
    the same-key comparison is limited to that key; the main check compares, between batch-mode and
    standard-mode keys, (a) the rubric effect pair_ar - pair_en (same key, variety and order; pair_en is
    standard mode for all keys), (b) the pair_ar dialect gap (dialect mean - MSA) and its
    difference-in-differences with the pair_en dialect gap, with key-bootstrap CIs for the
    batch - standard difference (domain-stratified, B=2000, seed 2026, the same replicate weights for
    both groups); plus pair_ar / pair_en accuracy (MSA and mean dialect) and invalid rates by group.
Output: results/analysis/opus_mode_check.json and a printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))
from ajd import analyze as A  # noqa: E402
from ajd import config as C  # noqa: E402
from ajd import stats as S  # noqa: E402
from ajd.competence import paired_mean_over  # noqa: E402

J = "opus-5.5"
meta, items, req, df, cov = A.load(False)
assert cov["complete"], "coverage incomplete"
keys = sorted(meta["keys"])
W = S.stratified_weights(keys)
req = req.reset_index(drop=True)
req["req_idx"] = np.arange(len(req))  # build_requests order = submission order

raw = [json.loads(line) for line in (ROOT / f"results/{J}.jsonl").open(encoding="utf-8") if line.strip()]
FULL = all("params" in x for x in raw)
if FULL:  # full logs: mode from meta.batch_id, settings from params
    mode = pd.DataFrame([{"rid": x["rid"], "batch": bool((x.get("meta") or {}).get("batch_id"))} for x in raw])
    params = {(json.dumps(x["params"], sort_keys=True), x["model"]) for x in raw}
else:  # release package: outputs carry no provider metadata; use data/provenance (api_route, settings hash)
    import gzip
    with gzip.open(ROOT / f"data/provenance/{J}.provenance.jsonl.gz", "rt", encoding="utf-8") as f:
        prov = [json.loads(line) for line in f if line.strip()]
    mode = pd.DataFrame([{"rid": x["rid"], "batch": x["api_route"] == "batch"} for x in prov])
    params = {("settings_sha256:" + x["settings_sha256"], x["requested_model"]) for x in prov}
mode = mode.drop_duplicates("rid", keep="last")
ERR = ROOT / f"results/{J}.errors.jsonl"  # failed-attempt log; not included in the release package
errs = ({json.loads(line)["rid"] for line in ERR.open(encoding="utf-8") if line.strip()}
        if ERR.exists() else None)

dj = df[df.judge == J].merge(mode, on="rid", how="left").merge(req[["rid", "req_idx"]], on="rid")
ar, en = dj[dj.condition == "pair_ar"], dj[dj.condition == "pair_en"]
out = {"judge": J, "identical_model_params_across_modes": len(params) == 1, "params": sorted(params)[0][0],
       "rows_by_mode_and_condition": {c: {"standard": int((~g.batch).sum()), "batch": int(g.batch.sum())}
                                      for c, g in dj.groupby("condition")}}

# ---- (1) split
out["pair_ar_by_variety"] = {v: {"standard": int((~g.batch).sum()), "batch": int(g.batch.sum())}
                             for v, g in ar.groupby("variety")}
kpos = ar.groupby("key").req_idx.min().rank().astype(int)  # key position in request order (1..1024)
kfrac = ar.groupby("key").batch.mean()
split_keys = kfrac[(kfrac > 0) & (kfrac < 1)].index.tolist()
bkeys = kfrac[kfrac == 1].index
skeys = kfrac[kfrac == 0].index
out["pair_ar_by_key"] = {
    "n_keys_all_standard": int(len(skeys)), "n_keys_all_batch": int(len(bkeys)), "n_keys_split": len(split_keys),
    "batch_req_idx_range": [int(ar[ar.batch].req_idx.min()), int(ar[ar.batch].req_idx.max())],
    "standard_req_idx_range_pair_ar": [int(ar[~ar.batch].req_idx.min()), int(ar[~ar.batch].req_idx.max())],
    "batch_key_positions": [int(kpos[bkeys].min()), int(kpos[bkeys].max())],
    "standard_key_positions": [int(kpos[skeys].min()), int(kpos[skeys].max())],
    "split_keys": {k: {"position": int(kpos[k]),
                       "batch_rows": {f"{r.variety}/{r.order}": float(r.correct) for r in ar[(ar.key == k) & ar.batch].itertuples()},
                       "standard_rows": {f"{r.variety}/{r.order}": float(r.correct) for r in ar[(ar.key == k) & ~ar.batch].itertuples()}}
                   for k in split_keys},
    "n_batch_rows_with_failed_sync_attempt": int(ar[ar.batch].rid.isin(errs).sum()) if errs is not None else None,
    "n_failed_sync_attempts": len(errs) if errs is not None else None,
    "contiguous_in_request_order": bool(ar[ar.batch].req_idx.min() > ar[~ar.batch].req_idx.max()),
}
dom = lambda ks: pd.Series([k.split("::", 1)[0] for k in ks]).value_counts().sort_index().to_dict()  # noqa: E731
out["domains"] = {"batch_keys": dom(bkeys), "standard_keys": dom(skeys)}

# ---- (2) mode check: group contrasts, split key excluded
gb = np.isin(keys, bkeys)
gs = np.isin(keys, skeys)


def grp(x):
    """x: per-key vector -> batch-group, standard-group summaries and batch - standard difference."""
    x = np.asarray(x, float)
    xb, xs = np.where(gb, x, np.nan), np.where(gs, x, np.nan)
    bb, bs = S.boot_mean(W, xb), S.boot_mean(W, xs)
    return {"batch_keys": {"est": float(np.nanmean(xb)), "ci95": S._ci(bb), "n_keys": int((~np.isnan(xb)).sum())},
            "standard_keys": {"est": float(np.nanmean(xs)), "ci95": S._ci(bs), "n_keys": int((~np.isnan(xs)).sum())},
            "diff_batch_minus_standard": {"est": float(np.nanmean(xb) - np.nanmean(xs)), "ci95": S._ci(bb - bs)}}


Aar, Ben = A.mpair(ar, en, "correct", keys, "order")  # order-matched pair_ar vs pair_en
a, b = paired_mean_over(Aar, Ben, C.VARIETIES)
mc = {}
mc["rubric_effect_pair_ar_minus_pair_en_7var"] = grp(a - b)
a, b = paired_mean_over(Aar, Ben, C.DIALECTS)
mc["rubric_effect_dialects"] = grp(a - b)
mc["rubric_effect_MSA"] = grp(Aar["MSA"] - Ben["MSA"])
Gar = A.mpair(ar, ar, "correct", keys, "order", ref="MSA")
Gen = A.mpair(en, en, "correct", keys, "order", ref="MSA")
ga = np.subtract(*paired_mean_over(*Gar, C.DIALECTS))
ge = np.subtract(*paired_mean_over(*Gen, C.DIALECTS))
mc["pair_ar_dialect_gap"] = grp(ga)
mc["pair_en_dialect_gap_same_key_groups"] = grp(ge)
mc["did_dialect_gap_pair_ar_minus_pair_en"] = grp(ga - ge)
Kar, Ken = A.kmat(ar, "correct", keys), A.kmat(en, "correct", keys)
mc["accuracy"] = {"pair_ar_MSA": grp(Kar["MSA"]), "pair_ar_dialect_mean": grp(Kar[C.DIALECTS].mean(axis=1)),
                  "pair_ar_ENG": grp(Kar["ENG"]),
                  "pair_en_MSA": grp(Ken["MSA"]), "pair_en_dialect_mean": grp(Ken[C.DIALECTS].mean(axis=1))}
inv = ar.assign(inv=(~ar.valid).astype(float))
mc["pair_ar_invalid_rate"] = {"batch_rows": float(inv[inv.batch].inv.mean()), "standard_rows": float(inv[~inv.batch].inv.mean())}
# invalid rates by key group for every condition (all non-pair_ar conditions are standard mode for all keys)
mc["invalid_rate_by_key_group"] = {c: {"batch_keys": float(1 - g[g.key.isin(bkeys)].valid.mean()),
                                       "standard_keys": float(1 - g[g.key.isin(skeys)].valid.mean())}
                                   for c, g in dj.groupby("condition") if c in ("pair_en", "pair_ar", "point", "solve")}
rawd = {x["rid"]: x for x in raw}
bi = ar[ar.batch & ~ar.valid]
if FULL:
    stops = [(rawd[r].get("meta") or {}).get("stop") for r in bi.rid]
else:  # release package: finish_reason in data/provenance holds meta.stop
    provd = {x["rid"]: x for x in prov}
    stops = [provd[r]["finish_reason"] for r in bi.rid]
mc["batch_invalid_stop_reasons"] = pd.Series(stops).value_counts().to_dict()
mc["batch_invalid_by_domain"] = pd.Series([k.split("::", 1)[0] for k in bi.key]).value_counts().to_dict()
out["mode_check"] = mc
out["notes"] = ("Mode is confounded with key position (batch = last keys in request order); the split key has "
                "different varieties/orders in each mode, so no within-key comparison is possible. Group "
                "contrasts compare different keys, so they bound but cannot isolate a mode effect; pair_en "
                "(all standard mode) on the same key groups is the key-difficulty baseline.")

(ROOT / "results/analysis/opus_mode_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("pair_ar_by_variety", "domains")}))
print({k: v for k, v in out["pair_ar_by_key"].items() if k != "split_keys"}, out["pair_ar_by_key"]["split_keys"])
print("params identical:", out["identical_model_params_across_modes"], "invalid:", mc["pair_ar_invalid_rate"], mc["invalid_rate_by_key_group"], mc["batch_invalid_stop_reasons"], mc["batch_invalid_by_domain"])
p = lambda s: f"{100*s['est']:+6.2f} [{100*s['ci95'][0]:+6.2f},{100*s['ci95'][1]:+6.2f}]"  # noqa: E731
for name, r in list(mc.items()) + [(f"acc.{k}", v) for k, v in mc["accuracy"].items()]:
    if name in ("accuracy",) or "invalid" in name:
        continue
    print(f"{name:45s} B {p(r['batch_keys'])}  S {p(r['standard_keys'])}  B-S {p(r['diff_batch_minus_standard'])}")
