"""Load the pinned dataset, check alignment, and draw the frozen stratified sample."""
import hashlib
import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from . import config as C


def load_full() -> pd.DataFrame:
    path = hf_hub_download(C.HF_DATASET, C.HF_FILE, repo_type="dataset",
                           revision=C.HF_REVISION, local_dir=C.DATA_DIR / "raw")
    df = pd.read_parquet(path)
    df["choices"] = df["choices"].map(list)
    df["key"] = df["domain"] + "::" + df["qid"].astype(str)
    return df


def check_alignment(df: pd.DataFrame) -> dict:
    """Every key must exist in all 7 varieties with one shared gold answer."""
    counts = df.groupby("key")["dialect"].nunique()
    gold = df.groupby("key")["answer"].nunique()
    dup = df.duplicated(["key", "dialect"]).sum()
    report = {
        "rows": len(df), "keys": int(len(counts)),
        "keys_all_7": int((counts == 7).sum()),
        "keys_gold_mismatch": int((gold > 1).sum()),
        "duplicate_rows": int(dup),
        "bad_choice_len": int((df["choices"].map(len) != 4).sum()),
    }
    assert set(df["dialect"]) == set(C.VARIETIES), set(df["dialect"])
    assert report["keys_all_7"] == report["keys"] and report["keys_gold_mismatch"] == 0
    assert dup == 0 and report["bad_choice_len"] == 0
    return report


def flagged_keys() -> set:
    """Keys with any dialect row removed in the IFM 'verified' subset (robustness filter)."""
    files = [f for f in list_repo_files(C.HF_VERIFIED, repo_type="dataset") if f.endswith(".parquet")]
    ver = pd.concat(pd.read_parquet(hf_hub_download(C.HF_VERIFIED, f, repo_type="dataset",
                                                    local_dir=C.DATA_DIR / "raw_verified"))
                    for f in files)
    ver["key"] = ver["domain"] + "::" + ver["qid"].astype(str)
    # The verified subset only filters dialect rows, so a key is flagged when any dialect row is missing.
    have = ver.groupby("key")["dialect"].apply(set).to_dict()
    all_keys = set(load_full()["key"])
    return {k for k in all_keys if not set(C.DIALECTS) <= have.get(k, set())}


def distractor_index(key: str, gold: int) -> int:
    """Deterministic wrong option, identical across varieties for the same key."""
    wrong = [i for i in range(4) if i != gold]
    h = int(hashlib.sha256(f"{C.SEED}:{key}".encode()).hexdigest(), 16)
    return wrong[h % 3]


def build_sample() -> dict:
    df = load_full()
    report = check_alignment(df)
    keys = df[df["dialect"] == "ENG"][["key", "domain", "qid"]].sort_values(["domain", "qid"])
    rng = np.random.default_rng(C.SEED)
    sample, verbose = [], []
    ext_rng = np.random.default_rng(C.EXT_SEED)
    extension = []
    for domain, g in keys.groupby("domain", sort=True):
        picked = list(rng.choice(g["key"].to_numpy(), size=C.PER_DOMAIN, replace=False))
        sample += picked
        verbose += picked[: C.VERBOSE_PER_DOMAIN]
        # Extension drawn after, from the remaining keys, so the original 256 are unchanged.
        rest = g["key"][~g["key"].isin(picked)].to_numpy()
        extension += list(ext_rng.choice(rest, size=C.EXT_PER_DOMAIN, replace=False))
    try:
        flagged = flagged_keys()
    except Exception as e:  # robustness filter only; never block the main run
        print(f"warning: could not load verified subset: {e}")
        flagged = None
    out = {
        "seed": C.SEED, "revision": C.HF_REVISION, "alignment": report,
        "base_keys": sample, "keys": sample + extension, "verbose_keys": verbose,
        "flagged_keys_in_sample": sorted(set(sample + extension) & flagged) if flagged is not None else None,
    }
    C.DATA_DIR.mkdir(exist_ok=True)
    (C.DATA_DIR / "sample.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    sub = df[df["key"].isin(sample + extension)].copy()
    sub["distractor"] = [distractor_index(k, a) for k, a in zip(sub["key"], sub["answer"])]
    sub.to_parquet(C.DATA_DIR / "sample_items.parquet", index=False)
    return out


def load_sample():
    meta = json.loads((C.DATA_DIR / "sample.json").read_text())
    items = pd.read_parquet(C.DATA_DIR / "sample_items.parquet")
    items["choices"] = items["choices"].map(list)
    return meta, items
