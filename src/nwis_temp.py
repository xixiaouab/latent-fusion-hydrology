"""Download NWIS daily-mean stream temperature (00010) for the 671 CAMELS gauges
and build dataset/temperature.npy aligned to the existing daymet-365 tables.

Login-node friendly: stdlib urllib + numpy, single thread, polite pacing.
Calendar convention matches build_fusion_dataset.py: leap Dec-31 dropped, Feb-29 kept
(slot = tm_yday - 1, skip tm_yday == 366).
"""
import os
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
meta = json.load(open(F / "dataset" / "meta.json"))
ids = [str(b).zfill(8) for b in meta["basin_ids"]]
years = meta["years"]
ndays = len(years) * 365
col = {b: j for j, b in enumerate(ids)}
T = np.full((ndays, len(ids)), np.nan, np.float32)

y0set = set(years)


def didx(d):
    if d.year not in y0set:
        return None
    yd = d.timetuple().tm_yday
    if yd == 366:
        return None
    return years.index(d.year) * 365 + yd - 1


BASE = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={s}"
        "&startDT=1980-01-01&endDT=2014-12-31&parameterCd=00010&statCd=00003")

nrows = 0
for c0 in range(0, len(ids), 100):
    chunk = ids[c0:c0 + 100]
    url = BASE.format(s=",".join(chunk))
    body = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                body = r.read().decode("utf-8", "replace")
            break
        except Exception as e:
            print(f"chunk {c0}: retry {attempt} after {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    if body is None:
        print(f"chunk {c0}: FAILED", flush=True)
        continue
    vcols = []
    for ln in body.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split("\t")
        if p[0] == "agency_cd":
            vcols = [i for i, h in enumerate(p) if h.endswith("_00010_00003")]
            continue
        if p[0] != "USGS" or not vcols:
            continue
        site, date = p[1], p[2]
        j = col.get(site)
        if j is None:
            continue
        try:
            i = didx(dt.date.fromisoformat(date))
        except ValueError:
            continue
        if i is None:
            continue
        for vi in vcols:
            if vi < len(p) and p[vi] not in ("", "Ice", "Eqp", "***"):
                try:
                    T[i, j] = float(p[vi])
                    nrows += 1
                    break
                except ValueError:
                    pass
    print(f"chunk {c0}: total rows {nrows}", flush=True)
    time.sleep(2)

np.save(F / "dataset" / "temperature.npy", T)
obs_per_basin = np.isfinite(T).sum(0)
iv0, iv1 = years.index(2010) * 365, (years.index(2012) + 1) * 365
it0, it1 = years.index(2013) * 365, (years.index(2014) + 1) * 365
ir0, ir1 = years.index(1980) * 365, (years.index(2009) + 1) * 365
cov_val = np.isfinite(T[iv0:iv1]).mean(0)
cov_tst = np.isfinite(T[it0:it1]).mean(0)
cov_trn = np.isfinite(T[ir0:ir1]).mean(0)
print(f"basins with >=1000 obs days: {(obs_per_basin >= 1000).sum()}")
print(f"val 2010-12 cov>=0.7: {(cov_val >= 0.7).sum()}  >=0.5: {(cov_val >= 0.5).sum()}")
print(f"test 2013-14 cov>=0.7: {(cov_tst >= 0.7).sum()}")
print(f"train 1980-2009 cov>=0.1: {(cov_trn >= 0.1).sum()}  >=0.3: {(cov_trn >= 0.3).sum()}")
print(f"total obs rows {nrows}; table {T.shape} -> temperature.npy")
print("TEMP_OK")
