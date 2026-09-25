"""Transpose h7 cache: per-year (days, P, D) day-major -> single (P, days_all, D).

Patch-major layout gives contiguous 30-day reads per patch at train time
(same lesson as transpose_latents.py)."""
import os
import json
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
C = F / "h7_cache"
meta = json.load(open(F / "dataset/meta.json"))
years = meta["years"]
first = np.load(C / f"{years[0]}_0_h7.npy", mmap_mode="r")
P, D = first.shape[1], first.shape[2]
nd = len(years) * 365
out = np.lib.format.open_memmap(C / "h7_pm.npy", mode="w+",
                                dtype=np.float16, shape=(P, nd, D))
for yi, y in enumerate(years):
    a = np.load(C / f"{y}_0_h7.npy")[:365]           # (365, P, D)
    out[:, yi * 365:(yi + 1) * 365, :] = a.transpose(1, 0, 2)
    print(f"transposed {y}", flush=True)
out.flush()
print(f"H7_TRANSPOSE_OK {out.shape}", flush=True)
