"""Rewrite latents (day, basin, dim) -> (basin, day, dim) for per-basin reads."""
import os
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
D = F / "dataset"
src_order = ([(f"latents/train/{y}_0_latents.npz", y) for y in range(1980, 2019)]
             + [(f"latents/val/{y}_0_latents.npz", y) for y in (2019, 2020)]
             + [("latents/test/2021_0_latents.npz", 2021)])
YEARS = list(range(1980, 2022))
NB, ND, DD = 671, len(YEARS) * 365, 1024

dst = np.lib.format.open_memmap(D / "latents_bm.npy", mode="w+",
                                dtype=np.float16, shape=(NB, ND, DD))
for rel, y in src_order:
    z = np.load(F / rel, allow_pickle=True)
    a = z["latents"]                       # (365, nb, 1024)
    yi = YEARS.index(y)
    dst[:, yi * 365:(yi + 1) * 365, :] = np.ascontiguousarray(a.transpose(1, 0, 2))
    print("transposed", y, flush=True)
dst.flush()
print("TRANSPOSE_OK")
