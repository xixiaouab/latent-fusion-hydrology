"""Build the random-access normalized weather store for full ORBIT-2 in-loop FT.

wx_store.npy: (n_years*365, 19, 180, 360) fp16, normalized exactly as the
extractor does (nan_to_num then (x-mean)/std). modes: create | fill.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
D = Path(os.environ.get("ERA5_DAYMET_DIR", "era5_daymet"))
sys.path.insert(0, str(F / "scripts"))
import extract_latents as EX  # noqa: E402  (top-level stubs + IN_VARS + load_shard)

years = json.load(open(F / "dataset/meta.json"))["years"]
OUT = F / "wx_store.npy"

nm = np.load(D / "normalize_mean.npz")
ns = np.load(D / "normalize_std.npz")


def stat(store, v):
    for c in EX.KEY_ALIAS.get(v, [v]):
        if c in store.files:
            return float(np.asarray(store[c]).ravel()[0])
    raise KeyError(v)


mean = np.array([stat(nm, v) for v in EX.IN_VARS], np.float32)[:, None, None]
std = np.array([stat(ns, v) for v in EX.IN_VARS], np.float32)[:, None, None]

if sys.argv[1] == "create":
    np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16,
                              shape=(len(years) * 365, 19, 180, 360))
    print("WX_CREATED", flush=True)
else:
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    mm = np.lib.format.open_memmap(OUT, mode="r+")
    for yi, y in enumerate(years):
        if yi % world != rank:
            continue
        fp = next(D / s / f"{y}_0.npz" for s in ("train", "val", "test")
                  if (D / s / f"{y}_0.npz").exists())
        arr = EX.load_shard(fp, EX.IN_VARS)[:365]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = (arr - mean[None]) / std[None]
        mm[yi * 365: yi * 365 + arr.shape[0]] = arr.astype(np.float16)
        print(f"[{rank}] wx {y} done", flush=True)
    print(f"[{rank}] WX_FILL_DONE", flush=True)
