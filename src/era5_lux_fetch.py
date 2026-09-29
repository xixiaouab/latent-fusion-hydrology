"""ERA5 daily aggregates for the Luxembourg window -- v2: parallel + resumable.

Why v2: ARCO pressure-level chunks hold all 37 levels (1,37,721,1440), so v1's
one-level-at-a-time reads paid the full chunk 12 times (~42 min each, ~8.4 h).
v2 reads each pressure-level variable ONCE (all three levels from the same
chunks), runs the four pressure-level variables and two single-level groups as
separate processes, and writes one file per output channel (ch_<name>.npy), so
a crash resumes at variable granularity.

usage: EXP3_DIR=... python era5_lux_fetch.py <worker>
  worker in: temperature | u_component_of_wind | v_component_of_wind |
             specific_humidity | sl_a | sl_b | merge
  merge -> eu_store.npy (n_days, 19, 128, 128) float32 + eu_dates.json +
           eu_grid.json, channel order == global checkpoint dict_in_variables
           minus the 4 statics.  PRECIP_SCALE env (default 1) multiplies the
           precipitation channel at merge time (unit fix after UNIT CHECK).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

OUT = Path(os.environ.get("EXP3_DIR", "exp3"))
OUT.mkdir(parents=True, exist_ok=True)

D0, D1 = "2020-03-01", "2021-10-31"
LAT0, LAT1 = 68.0, 36.25                     # descending (ERA5 native): 128 cells
LON0, LON1 = -12.0, 19.75                    # 128 cells, crosses 0 deg
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LONS = (np.arange(LON0, LON1 + 1e-6, 0.25) % 360).round(2)
LEVELS = [200, 500, 850]
PL_STRIDE = int(os.environ.get("PL_STRIDE", "1"))    # hours between samples (pressure levels)
BLOCK_PL = int(os.environ.get("BLOCK_PL", "10"))     # days per read, pressure-level vars
BLOCK_SL = int(os.environ.get("BLOCK_SL", "30"))     # days per read, single-level vars

# (channel name, ARCO variable, level, daily aggregate) -- checkpoint order
SPEC = [
    ("2m_temperature",              "2m_temperature",            None, "mean"),
    ("2m_temperature_max",          "2m_temperature",            None, "max"),
    ("2m_temperature_min",          "2m_temperature",            None, "min"),
    ("temperature_200",             "temperature",               200,  "mean"),
    ("temperature_500",             "temperature",               500,  "mean"),
    ("temperature_850",             "temperature",               850,  "mean"),
    ("10m_u_component_of_wind",     "10m_u_component_of_wind",   None, "mean"),
    ("u_component_of_wind_200",     "u_component_of_wind",       200,  "mean"),
    ("u_component_of_wind_500",     "u_component_of_wind",       500,  "mean"),
    ("u_component_of_wind_850",     "u_component_of_wind",       850,  "mean"),
    ("10m_v_component_of_wind",     "10m_v_component_of_wind",   None, "mean"),
    ("v_component_of_wind_200",     "v_component_of_wind",       200,  "mean"),
    ("v_component_of_wind_500",     "v_component_of_wind",       500,  "mean"),
    ("v_component_of_wind_850",     "v_component_of_wind",       850,  "mean"),
    ("specific_humidity_200",       "specific_humidity",         200,  "mean"),
    ("specific_humidity_500",       "specific_humidity",         500,  "mean"),
    ("specific_humidity_850",       "specific_humidity",         850,  "mean"),
    ("total_precipitation_24hr",    "total_precipitation",       None, "sum"),
    ("volumetric_soil_water_layer_1", "volumetric_soil_water_layer_1", None, "mean"),
]
WORKERS = {
    "temperature": ["temperature"],
    "u_component_of_wind": ["u_component_of_wind"],
    "v_component_of_wind": ["v_component_of_wind"],
    "specific_humidity": ["specific_humidity"],
    "sl_a": ["2m_temperature", "total_precipitation"],
    "sl_b": ["10m_u_component_of_wind", "10m_v_component_of_wind",
             "volumetric_soil_water_layer_1"],
}
dates = [str(d)[:10] for d in np.arange(np.datetime64(D0), np.datetime64(D1) + 1)]
n = len(dates)


def ch_path(name):
    return OUT / f"ch_{name}.npy"


def fetch_var(ds, var):
    entries = [(name, lev, agg) for name, v, lev, agg in SPEC if v == var]
    if all(ch_path(name).exists() for name, _, _ in entries):
        print(f"{var}: all channels present, skip", flush=True)
        return
    is_pl = entries[0][1] is not None
    da = ds[var].sel(latitude=slice(LAT0, LAT1), longitude=LONS)
    if is_pl:
        da = da.sel(level=LEVELS)
    block = BLOCK_PL if is_pl else BLOCK_SL
    stride = PL_STRIDE if is_pl else 1
    per_day = 24 // stride
    outs = {name: np.full((n, 128, 128), np.nan, np.float32) for name, _, _ in entries}
    tv = time.time()
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        t0 = time.time()
        blk = da.sel(time=slice(dates[b0], dates[b1 - 1] + "T23:59"))
        if stride > 1:
            blk = blk.isel(time=slice(0, None, stride))
        arr = np.asarray(blk.values, np.float32)          # (h, [L], 128, 128)
        assert arr.shape[0] == per_day * (b1 - b0), (var, b0, arr.shape)
        arr = arr.reshape(b1 - b0, per_day, *arr.shape[1:])
        for name, lev, agg in entries:
            a = arr if lev is None else arr[:, :, LEVELS.index(lev)]
            outs[name][b0:b1] = getattr(np, agg)(a, axis=1)
        print(f"  {var}: days {b0}-{b1} in {time.time()-t0:.1f}s "
              f"(elapsed {time.time()-tv:.0f}s)", flush=True)
    for name, _, _ in entries:
        assert np.isfinite(outs[name]).all(), name
        np.save(ch_path(name), outs[name])
        print(f"  saved {ch_path(name).name} mean={outs[name].mean():.4g}", flush=True)
    print(f"{var}: VAR_DONE in {time.time()-tv:.0f}s", flush=True)


def merge(ds):
    store = np.zeros((n, len(SPEC), 128, 128), np.float32)
    pscale = os.environ.get("PRECIP_SCALE", "auto")
    nm = OUT / "normalize_mean.npz"
    for ci, (name, *_rest) in enumerate(SPEC):
        a = np.load(ch_path(name))
        assert a.shape == (n, 128, 128), (name, a.shape)
        if name == "total_precipitation_24hr":
            if pscale == "auto" and nm.exists():
                # ERA5 precipitation is in metres; the checkpoint may use mm.
                ratio = float(np.asarray(np.load(nm)[name]).ravel()[0]) / max(a.mean(), 1e-12)
                s = 1000.0 if 300 < ratio < 3000 else (0.001 if 1 / 3000 < ratio < 1 / 300 else 1.0)
                print(f"  precip unit auto-check: ckpt/ours mean ratio {ratio:.3g} -> scale x{s}", flush=True)
            else:
                s = float(pscale) if pscale != "auto" else 1.0
            a = a * s
        store[:, ci] = a
    np.save(OUT / "eu_store.npy", store)
    json.dump(dates, open(OUT / "eu_dates.json", "w"))
    json.dump({"lat": list(map(float, ds.latitude.sel(latitude=slice(LAT0, LAT1)).values)),
               "lon": list(map(float, LONS))}, open(OUT / "eu_grid.json", "w"))
    nm = OUT / "normalize_mean.npz"
    if nm.exists():
        mu, sd = np.load(nm), np.load(OUT / "normalize_std.npz")
        print("UNIT CHECK (ours vs ckpt mean / ckpt std):", flush=True)
        for ci, (name, *_rest) in enumerate(SPEC):
            if name in mu:
                print(f"  {name}: ours {store[:, ci].mean():.4g}  "
                      f"ckpt {float(np.asarray(mu[name]).ravel()[0]):.4g} / "
                      f"{float(np.asarray(sd[name]).ravel()[0]):.4g}", flush=True)
    print("EU_FETCH_DONE", store.shape, flush=True)


if __name__ == "__main__":
    w = sys.argv[1]
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    if w == "merge":
        merge(ds)
    else:
        for var in WORKERS[w]:
            fetch_var(ds, var)
        print(f"WORKER_DONE {w}", flush=True)
