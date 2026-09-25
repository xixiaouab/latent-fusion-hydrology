"""NWM v2.1 retrospective vs our validation panel, on identical windows.

Runs on a DTN (outbound internet + same Lustre). Steps:
1. gauge -> NWM feature_id (COMID) via NLDI, cached
2. selective read of NOAA's public zarr (streamflow, 2010-2013, 64 reaches)
3. hourly -> daily mean, m3/s -> mm/day by basin area
4. NSE per basin on the exact 6x10-day windows; median across basins
"""
import os
import datetime as dt
import json
from pathlib import Path

import numpy as np
import requests
import s3fs
import xarray as xr

FUS = Path(os.environ.get("FUSION_ROOT", ".")) / "dataset"
OUT = Path(os.environ.get("NWM_OUT", "nwm_compare"))
OUT.mkdir(exist_ok=True)
vp = json.load(open(FUS / "val_pairs.json"))
pairs, obs, HOR = vp["pairs"], vp["obs"], vp["HOR"]
basins = sorted({p["basin"] for p in pairs})
area = {p["basin"]: p["area_m2"] for p in pairs}
print(f"panel: {len(pairs)} pairs / {len(basins)} basins", flush=True)

# ---- 1. gauge -> comid (cached) ----
cpath = OUT / "comids.json"
comid = json.load(open(cpath)) if cpath.exists() else {}
for g in basins:
    if g in comid:
        continue
    try:
        r = requests.get(
            f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{g}",
            timeout=30).json()
        comid[g] = int(r["features"][0]["properties"]["comid"])
    except Exception as e:
        comid[g] = None
        print(f"NLDI_FAIL {g}: {e}", flush=True)
json.dump(comid, open(cpath, "w"))
ok = [g for g in basins if comid.get(g)]
print(f"comids: {len(ok)}/{len(basins)} resolved", flush=True)

# ---- 2. selective zarr read ----
fs = s3fs.S3FileSystem(anon=True)
ds = xr.open_zarr(
    s3fs.S3Map("s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr", s3=fs),
    consolidated=True)
fids = [comid[g] for g in ok]
sub = ds["streamflow"].sel(time=slice("2010-01-01", "2013-01-15"),
                           feature_id=fids)
print(f"pulling {sub.shape} hourly values...", flush=True)
q = sub.load()  # (time, feature) small: ~26k x 64
daily = q.resample(time="1D").mean()
ddates = [str(t)[:10] for t in np.asarray(daily["time"].values).astype("datetime64[D]")]
didx = {d: i for i, d in enumerate(ddates)}
dvals = np.asarray(daily.values)  # (days, features)
fcol = {g: j for j, g in enumerate(ok)}
np.save(OUT / "nwm_daily_cms.npy", dvals)

# ---- 3+4. windows, mm/day, NSE ----
def window_dates(d0):
    # 365-day calendar: skip Dec 31 in leap years
    d = dt.date.fromisoformat(d0)
    out = []
    while len(out) < HOR:
        if not (d.month == 12 and d.day == 31 and
                (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0))):
            out.append(str(d))
        d += dt.timedelta(days=1)
    return out

def nse(sim, ob):
    m = np.isfinite(sim) & np.isfinite(ob)
    if m.sum() < 5:
        return None
    s, o = sim[m], ob[m]
    den = ((o - o.mean()) ** 2).sum()
    return float(1 - ((s - o) ** 2).sum() / den) if den > 0 else None

per_basin = {}
for g in ok:
    sims, obss = [], []
    for p in [p for p in pairs if p["basin"] == g]:
        wd = window_dates(p["date0"])
        sim = np.array([dvals[didx[d], fcol[g]] if d in didx else np.nan
                        for d in wd], float)
        sim = sim * 8.64e7 / area[g]          # m3/s -> mm/day
        ob = np.array([np.nan if v is None else v
                       for v in obs[f"{g}_{p['og']}"]], float)
        sims.append(sim)
        obss.append(ob)
    v = nse(np.concatenate(sims), np.concatenate(obss))
    if v is not None:
        per_basin[g] = v
med = float(np.median(list(per_basin.values())))
json.dump({"per_basin": per_basin, "median": med},
          open(OUT / "nwm_result.json", "w"))
print(f"NWM_RESULT median NSE {med:.4f} over {len(per_basin)} basins", flush=True)
