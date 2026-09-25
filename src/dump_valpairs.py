"""Dump the exact 64-basin x 6-origin validation panel (basin ids, origin
dates, areas) to val_pairs.json so the NWM comparison evaluates on
identical windows. Replicates fusion_train.build_val_pairs offline."""
import os
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
sys.path.insert(0, str(F / "scripts"))
import fusion_train as FT  # noqa: E402

T = np.load(F / "dataset/streamflow.npy")
meta = json.load(open(F / "dataset/meta.json"))
bids = meta["basin_ids"]
areas = meta["areas_m2"]
CTX, LAT_PAST, HOR = FT.CTX, FT.LAT_PAST, FT.HOR

def yr(y):
    return (y - 1980) * 365

i0, i1 = yr(2010), yr(2013)
cov = np.isfinite(T[i0:i1]).mean(0)
lo = max(i0, CTX + LAT_PAST)
pairs = []
for bi in np.argsort(-cov):
    if cov[bi] < 0.7 or len(pairs) >= 64 * 6:
        break
    got = 0
    for og in np.linspace(lo, i1 - HOR - 1, 18).astype(int):
        fut = T[og:og + HOR, bi]
        ctx = T[og - CTX:og, bi]
        if np.isfinite(fut).sum() >= 8 and np.isfinite(ctx).mean() >= 0.5:
            pairs.append((int(bi), int(og)))
            got += 1
            if got >= 6:
                break

def date_of(i):
    # 365-day calendar: leap years drop Dec 31, so Jan1+doy is exact.
    return str(dt.date(1980 + i // 365, 1, 1) + dt.timedelta(days=int(i % 365)))

out = {
    "HOR": HOR,
    "pairs": [{"basin": bids[bi],
               "area_m2": areas[bids[bi]] if isinstance(areas, dict) else areas[bi],
               "bi": bi, "og": og, "date0": date_of(og)} for bi, og in pairs],
    "obs": {f"{bids[bi]}_{og}": [None if not np.isfinite(v) else round(float(v), 5)
            for v in T[og:og + HOR, bi]] for bi, og in pairs},
}
json.dump(out, open(F / "dataset/val_pairs.json", "w"))
nb = len({p['basin'] for p in out['pairs']})
print(f"VALPAIRS_OK {len(pairs)} pairs, {nb} basins", flush=True)
