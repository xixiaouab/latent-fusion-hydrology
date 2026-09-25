"""Zero-shot fusion transfer at a dam-regulated basin (TS-model Experiment 4).

Basin 10312150 (Carson River below Lahontan Reservoir, NV). The CAMELS-trained
fusion adapter (adapter_best.pt) is evaluated as-is — no retraining — under
four configurations: {without, with} DamOutflow covariate x {K=0 full future
covariates, K=10 all future covariates masked}.

Units: the CSV's Streamflow is already mm/day (median 0.31); DamOutflow is
m^3/s and is converted to mm/day over the 4664.6 km^2 drainage area so both
covariate and target magnitudes match the adapter's training distribution. 40 forecast origins spread across 2018 (the paper's eval year);
metric = pooled NSE over all origin-days, matching the notebook's pooling.
"""
import os
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

F = Path(os.environ.get("FUSION_ROOT", "."))
sys.path.insert(0, str(F / "scripts"))
from fusion_train import FusionAdapter, frozen_forward, nse, CTX, HOR, LAT_PAST, LAT_FUT  # noqa: E402
from fixload import load_pipeline  # noqa: E402

AREA_KM2 = 1801 * 2.58999          # sq mi -> km^2
CMS2MM = 86.4 / AREA_KM2           # m^3/s -> mm/day over the basin
FORCING_COLS = ["Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown",
                "CRainf_frac", "CAPE", "PotEvap", "Rainf", "SWdown"]

# ---- CSV (daily, 2016-01-01 .. 2018-12-31) ----
rows = [ln.rstrip("\n").split(",") for ln in open(F / "exp4/10312150_dam514.csv")]
hdr = rows[0]
ci = {c: hdr.index(c) for c in hdr}
dates, rec = [], []
for r in rows[1:]:
    dates.append(dt.date.fromisoformat(r[ci["datetime"]][:10]))
    rec.append([float(r[ci[c]]) if r[ci[c]] not in ("", "nan") else np.nan
                for c in FORCING_COLS + ["Streamflow", "DamOutflow"]])
rec = np.asarray(rec, np.float32)
flow = rec[:, len(FORCING_COLS)]
outf = rec[:, len(FORCING_COLS) + 1] * CMS2MM
forc = rec[:, :len(FORCING_COLS)]
print(f"csv: {len(dates)} days {dates[0]} -> {dates[-1]}; flow mm/day "
      f"median {np.nanmedian(flow):.3f} mean {np.nanmean(flow):.3f} "
      f"max {np.nanmax(flow):.2f}", flush=True)

# ---- latents: (day,1,1024) year files -> per-date lookup ----
YRS = [2016, 2017, 2018]
lat_by_year = {y: np.load(F / f"exp4_latents/{y}_0_latents.npz")["latents"][:, 0, :]
               for y in YRS}
def lat_of(date):
    yd = date.timetuple().tm_yday
    if yd == 366:                       # daymet drops leap Dec-31: reuse Dec-30
        yd = 365
    return lat_by_year[date.year][yd - 1]
L = np.stack([lat_of(d) for d in dates]).astype(np.float32)   # (days, 1024)

# ---- eval panel: 40 origins across 2018 ----
i2018 = [i for i, d in enumerate(dates) if d.year == 2018]
lo, hi = max(i2018[0], CTX), i2018[-1] - HOR + 1
origins = np.linspace(lo, hi, 40).astype(int)

dev = torch.device("cuda")
pipe = load_pipeline(os.environ.get("TS_MODEL_DIR", F / "ts_model"), device="cuda")
pipe.model.eval()
for p in pipe.model.parameters():
    p.requires_grad_(False)
adapter = FusionAdapter(per_day=True).to(dev)
adapter.load_state_dict(torch.load(F / "runs/fusion_v2p_frt/adapter_best.pt",
                                   map_location=dev))
adapter.eval()

def run(with_outflow, mask_days, bs=8):
    cov = np.concatenate([forc, outf[:, None]], 1) if with_outflow else forc
    m0s, m1s, obss = [], [], []
    for c0 in range(0, len(origins), bs):
        tg, fx, ff, y, lw = [], [], [], [], []
        for og in origins[c0:c0 + bs]:
            tg.append(flow[og - CTX:og])
            fx.append(cov[og - CTX:og])
            f = cov[og:og + HOR].copy()
            if mask_days > 0:
                f[HOR - mask_days:, :] = np.nan
            ff.append(f)
            y.append(flow[og:og + HOR])
            lw.append(L[og - LAT_PAST:og + LAT_FUT])
        t = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).to(dev).float()
        tg, fx, ff, yv, lw = t(tg), t(fx), t(ff), t(y), t(lw)
        q0, ftok = frozen_forward(pipe, tg, fx, ff, dev)
        with torch.no_grad():
            dq = adapter(ftok, lw)
        m0s.append(torch.clamp(q0, min=0.0)[:, 21 // 2].cpu().numpy())
        m1s.append(torch.clamp(q0 + dq, min=0.0)[:, 21 // 2].cpu().numpy())
        obss.append(yv.cpu().numpy())
        del q0, ftok, dq, tg, fx, ff, yv, lw
        torch.cuda.empty_cache()
    m0 = np.concatenate(m0s); m1 = np.concatenate(m1s); obs = np.concatenate(obss)
    nb = nse(m0.ravel(), obs.ravel())
    nf = nse(m1.ravel(), obs.ravel())
    tag = f"outflow={'Y' if with_outflow else 'N'} K={mask_days}"
    print(f"EXP4 {tag}: base {nb:.4f} fused {nf:.4f} delta {nf-nb:+.4f}",
          flush=True)

for wo in (False, True):
    for k in (0, 10):
        run(wo, k)
print("EXP4_DONE", flush=True)
