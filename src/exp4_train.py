"""exp4_train: regulated-basin (Carson River 10312150) TRAINED fusion cells.

Strategy A: frozen TS backbone + frozen ORBIT-2 latents (cached 2016-2018)
+ FusionAdapter trained from scratch on 2017 origins; evaluated on the same
40 origins across 2018 as exp4_eval (so zero-shot numbers stay comparable).
Protocol: strict pre-issue latents (past 30 d, no future); future covariates
masked for the last K days of the horizon (K=10 = fully blind).

usage: python exp4_train.py NAME K WITH_OUTFLOW(0|1) [STEPS]
"""
import os
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

F = Path(os.environ.get("FUSION_ROOT", "."))
sys.path.insert(0, str(F / "scripts"))
import fusion_train as FT  # noqa: E402
from fixload import load_pipeline  # noqa: E402

FT.LAT_PAST, FT.LAT_FUT = 30, 0          # strict no-future latent window
CTX, HOR = FT.CTX, FT.HOR

NAME = sys.argv[1]
K = int(sys.argv[2])
WITH_OUT = bool(int(sys.argv[3]))
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 600

AREA_KM2 = 1801 * 2.58999
CMS2MM = 86.4 / AREA_KM2
FORCING_COLS = ["Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown",
                "CRainf_frac", "CAPE", "PotEvap", "Rainf", "SWdown"]

rows = [l.rstrip("\n").split(",") for l in open(F / "exp4/10312150_dam514.csv")]
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

lat_by_year = {y: np.load(F / f"exp4_latents/{y}_0_latents.npz")["latents"][:, 0, :]
               for y in (2016, 2017, 2018)}


def lat_of(date):
    yd = date.timetuple().tm_yday
    arr = lat_by_year[date.year]
    return arr[min(yd - 1, len(arr) - 1)]


L = np.stack([lat_of(d) for d in dates]).astype(np.float32)
cov = np.concatenate([forc, outf[:, None]], 1) if WITH_OUT else forc

# Carson has only 3 years and the backbone needs 2 years of context, so all
# usable origins live in 2018. Temporal split inside 2018: first 55% train,
# 20-day buffer (no target-window overlap), last stretch = validation.
first2018 = next(i for i, d in enumerate(dates) if d.year == 2018)
lo = max(first2018, CTX, FT.LAT_PAST)
hi = len(dates) - HOR
split = lo + int((hi - lo) * 0.55)
train_origins = np.arange(lo, split)
val_origins = np.linspace(split + 20, hi, 30).astype(int)
print(f"EXP4TRAIN {NAME}: K={K} outflow={'Y' if WITH_OUT else 'N'} "
      f"train_origins={len(train_origins)} val={len(val_origins)}"
      f"@[{split + 20},{hi}] steps={STEPS}", flush=True)

dev = torch.device("cuda")
pipe = load_pipeline(os.environ.get("TS_MODEL_DIR", F / "ts_model"), device="cuda")
pipe.model.eval()
for p in pipe.model.parameters():
    p.requires_grad_(False)
adapter = FT.FusionAdapter(per_day=True).to(dev)
opt = torch.optim.Adam(adapter.parameters(), lr=2e-4)
LEVELS = pipe.model.quantile_levels.to(dev).float()
NQ = LEVELS.numel()
wq = torch.ones(NQ, device=dev)
wq[NQ // 2] = 3.0


def make_batch(ogs):
    tg = np.stack([flow[og - CTX:og] for og in ogs])
    fx = np.stack([cov[og - CTX:og] for og in ogs])
    ffl = []
    for og in ogs:
        f = cov[og:og + HOR].copy()
        if K > 0:
            f[HOR - K:, :] = np.nan
        ffl.append(f)
    ff = np.stack(ffl)
    y = np.stack([flow[og:og + HOR] for og in ogs])
    lw = np.stack([L[og - FT.LAT_PAST:og + FT.LAT_FUT] for og in ogs])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float()
    return t(tg), t(fx), t(ff), t(y), t(lw)


def evaluate():
    adapter.eval()
    m0s, m1s, obs = [], [], []
    with torch.no_grad():
        for c0 in range(0, len(val_origins), 8):
            tg, fx, ff, y, lw = make_batch(val_origins[c0:c0 + 8])
            q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
            dq = adapter(ftok, lw)
            m0s.append(torch.clamp(q0, min=0.0)[:, NQ // 2].cpu().numpy())
            m1s.append(torch.clamp(q0 + dq, min=0.0)[:, NQ // 2].cpu().numpy())
            obs.append(y.cpu().numpy())
    adapter.train()
    m0 = np.concatenate(m0s).ravel()
    m1 = np.concatenate(m1s).ravel()
    ob = np.concatenate(obs).ravel()
    return FT.nse(m0, ob), FT.nse(m1, ob)


rng = np.random.default_rng(0)
nb0, nf0 = evaluate()
print(f"BASELINE {nb0:.4f} fused_init {nf0:.4f}", flush=True)
best = nf0
for step in range(1, STEPS + 1):
    ogs = rng.choice(train_origins, 16, replace=True)
    tg, fx, ff, y, lw = make_batch(ogs)
    with torch.no_grad():
        q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
    dq = adapter(ftok, lw)
    loss = FT.pinball(q0 + dq, y, LEVELS, wq)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        nb, nf = evaluate()
        tag = ""
        if nf > best:
            best = nf
            torch.save(adapter.state_dict(), F / f"runs/{NAME}_best.pt")
            tag = "BEST"
        print(f"EVAL step {step}: base {nb:.4f} fused {nf:.4f} "
              f"delta {nf - nb:+.4f} {tag}", flush=True)
print(f"EXP4TRAIN_DONE {NAME} best_fused {best:.4f} baseline {nb0:.4f} "
      f"best_delta {best - nb0:+.4f}", flush=True)
