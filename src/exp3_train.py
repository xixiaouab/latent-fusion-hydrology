"""exp3_train: CAMELS-LUX basin 40, 15-min streamflow, 24 h horizon (96 steps).

Frozen TS backbone + frozen ORBIT-2 (global 126M) daily latents + FusionAdapter.
Latent protocol: strict pre-issue -- the LAT_PAST calendar days BEFORE the
origin's day (the origin day itself is excluded: its daily aggregate would
contain hours after the issue time).
Covariate protocol: K = number of trailing horizon steps whose future
covariates are hidden (K=96 fully blind, K=0 open-book as in the repo notebook).
Splits: train origins 2020-04-01..2020-10-31, val origins 30 spread over
2020-11-01..2020-12-31 (checkpoint selection), test = the notebook's own 30
origins across 2021 (identical selection rule) -- reported at best-val and last.

usage: python exp3_train.py NAME K BIG(0|1) [STEPS=800] [LAT_PAST=30] [TOPFT=off|frozen|ft]
  TOPFT != off: ORBIT-2's last block + final norm run in-loop on the cached
  pre-top-block features of the 3x3 basin box (attention over those 9 tokens);
  frozen = localization control, ft = fine-tune its weights at lr x 0.1.
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

F = Path(os.environ.get("FUSION_ROOT", "."))
sys.path.insert(0, str(F / "scripts"))
import fusion_train as FT  # noqa: E402
from fixload import load_pipeline  # noqa: E402

LUX_CSV = Path(os.environ.get(        # the time-series model's sub-hourly example basin
    "LUX_CSV", Path(os.environ.get("TS_MODEL_DIR", F / "ts_model"))
    / "Example/data/exp3_subhourly_lux/LUX_40.csv"))

NAME = sys.argv[1]
K = int(sys.argv[2])
BIG = bool(int(sys.argv[3]))
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 800
LAT_PAST = int(sys.argv[5]) if len(sys.argv) > 5 else 30
TOPFT = sys.argv[6] if len(sys.argv) > 6 else "off"
DEVICE = os.environ.get("EXP3_DEVICE", "cuda")
IN = Path(os.environ.get("EXP3_IN", str(F / "exp3")))     # extraction outputs
SMOKE = os.environ.get("EXP3_SMOKE") == "1"                 # CPU smoke test: 2 steps, 2 origins
assert TOPFT in ("off", "frozen", "ft"), TOPFT

CTX, HOR = 8760, 96                      # notebook: ~91 days of 15-min context, 24 h ahead
FT.CTX, FT.HOR = CTX, HOR                # frozen_forward / adapter read these globals
FT.LAT_PAST, FT.LAT_FUT = LAT_PAST, 0
COV_COLS = ["Precip", "AirTemp"]
EVAL_YEAR, N_ORIGINS = 2021, 30
OUT = F / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)

# ---- data ----------------------------------------------------------------
rows = [l.rstrip("\n").split(",") for l in
        open(LUX_CSV)]
hdr = rows[0]
ci = {c: hdr.index(c) for c in hdr}
ts = [dt.datetime.fromisoformat(r[ci["datetime"]]) for r in rows[1:]]
rec = np.asarray([[float(r[ci[c]]) if r[ci[c]] not in ("", "nan") else np.nan
                   for c in ["Q"] + COV_COLS] for r in rows[1:]], np.float32)
flow, cov = rec[:, 0], rec[:, 1:]
n = len(ts)
assert all((ts[i + 1] - ts[i]) == dt.timedelta(minutes=15) for i in range(n - 1)), "gaps"

NOLAT = os.environ.get("EXP3_NOLAT") == "1"          # baseline-only dry run (no latents yet)
if NOLAT:
    L = np.zeros((610, 1024), np.float32)
    lat_day0, STEPS = dt.date(2020, 3, 1), 0
    print("EXP3_NOLAT=1: zero latents, baseline anchors only", flush=True)
else:
    lz = np.load(IN / "lux_latents.npz")
    L = lz["latents"].astype(np.float32)                # (days, 1024)
    lat_day0 = dt.date.fromisoformat(str(lz["dates"][0]))
    if TOPFT != "off":                                  # (days, 9, 1024) pre-top-block box tokens
        L = np.load(IN / "lux_h7_box.npy").astype(np.float32)
        assert L.shape[0] == lz["latents"].shape[0], L.shape
lat_days = L.shape[0]


def lat_window(og):
    """rows of L for the LAT_PAST days strictly before the origin's calendar day"""
    d = (ts[og].date() - lat_day0).days
    assert LAT_PAST <= d <= lat_days, (ts[og], d)
    return L[d - LAT_PAST:d]


years = np.array([t.year for t in ts])
# notebook rule: positions in EVAL_YEAR with full context and horizon
pos = np.where(years == EVAL_YEAR)[0]
pos = pos[(pos >= CTX) & (pos + HOR <= n)]
test_origins = pos[np.linspace(0, len(pos) - 1, N_ORIGINS).astype(int)]
lo = max(CTX, next(i for i, t in enumerate(ts) if (t.date() - lat_day0).days >= LAT_PAST))
tr_hi = next(i for i, t in enumerate(ts) if t >= dt.datetime(2020, 11, 1))
train_origins = np.arange(lo, tr_hi - HOR)
vl_hi = next(i for i, t in enumerate(ts) if t >= dt.datetime(2021, 1, 1))
val_origins = np.linspace(tr_hi, vl_hi - HOR, N_ORIGINS).astype(int)
BATCH, EVAL_EVERY = 16, 50
if SMOKE:
    val_origins, test_origins, STEPS, BATCH, EVAL_EVERY = val_origins[:2], test_origins[:2], 2, 2, 1
print(f"EXP3TRAIN {NAME}: K={K}/{HOR} big={int(BIG)} lat_past={LAT_PAST} steps={STEPS} | "
      f"train {len(train_origins)} origins {ts[train_origins[0]]}..{ts[train_origins[-1]]} | "
      f"val {len(val_origins)} {ts[val_origins[0]]}..{ts[val_origins[-1]]} | "
      f"test {len(test_origins)} {ts[test_origins[0]]}..{ts[test_origins[-1]]}", flush=True)

# ---- models --------------------------------------------------------------
dev = torch.device(DEVICE)
pipe = load_pipeline(os.environ.get("TS_MODEL_DIR", F / "ts_model"), device=DEVICE)
pipe.model.eval()
for p_ in pipe.model.parameters():
    p_.requires_grad_(False)
print(f"backbone context_length {pipe.fc.context_length} (notebook passes {CTX}; the "
      f"pipeline pads/truncates to its own length, frozen_forward does the same)", flush=True)
if BIG:
    adapter = FT.FusionAdapter(d_h=1024, n_heads=16, n_blocks=4, per_day=True,
                               latent_layers=4).to(dev)
else:
    adapter = FT.FusionAdapter(per_day=True).to(dev)
if TOPFT != "off":
    adapter = FT.TopFTAdapter(adapter, FT.TopBlockORBIT(IN / "lux_topft_meta.pt"),
                              trainable=TOPFT == "ft").to(dev)
    print(f"TOPFT={TOPFT}: ORBIT-2 top block "
          f"{sum(p_.numel() for p_ in adapter.top.parameters())/1e6:.1f}M params in-loop", flush=True)
print(f"adapter params: {sum(p_.numel() for p_ in adapter.parameters())/1e6:.2f}M", flush=True)
if TOPFT == "ft":
    top_ids = {id(p_) for p_ in adapter.top.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p_ for p_ in adapter.parameters() if id(p_) not in top_ids], "lr": 2e-4},
         {"params": list(adapter.top.parameters()), "lr": 2e-5}], weight_decay=1e-4)
else:
    opt = torch.optim.AdamW([p_ for p_ in adapter.parameters() if p_.requires_grad],
                            lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, STEPS))
LEVELS = pipe.model.quantile_levels.to(dev).float()
NQ = LEVELS.numel()
wq = torch.ones(NQ, device=dev)
wq[NQ // 2] = 3.0
wq = wq / wq.mean()


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
    lw = np.stack([lat_window(og) for og in ogs])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float()
    lat = t(lw)
    if TOPFT != "off":                                  # (B, T, 9, D) + mask + uniform box weights
        k = lw.shape[2]
        lat = (lat, torch.ones(len(ogs), k, dtype=torch.bool, device=dev),
               torch.full((len(ogs), k), 1.0 / k, device=dev))
    return t(tg), t(fx), t(ff), t(y), lat


def nse_lead(obs, sim, min_samples=24):
    """notebook's NSE: pooled over the given pairs, NaN-guarded"""
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < min_samples:
        return np.nan
    den = np.sum((o - o.mean()) ** 2)
    scale = max(abs(o.mean()), 1.0)
    if den <= (1e-10 * scale) ** 2 * len(o):
        return np.nan
    return 1.0 - np.sum((s - o) ** 2) / den


def run(origins, bs=8):
    adapter.eval()
    m0s, m1s, obs = [], [], []
    with torch.no_grad():
        for c0 in range(0, len(origins), bs):
            tg, fx, ff, y, lw = make_batch(origins[c0:c0 + bs])
            q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
            dq = adapter(ftok, lw)
            m0s.append(torch.clamp(q0, min=0.0)[:, NQ // 2].cpu().numpy())
            m1s.append(torch.clamp(q0 + dq, min=0.0)[:, NQ // 2].cpu().numpy())
            obs.append(y.cpu().numpy())
    adapter.train()
    return np.concatenate(obs), np.concatenate(m0s), np.concatenate(m1s)   # (n, HOR)


def summarize(ob, m0, m1):
    out = {}
    for tag, sim in (("base", m0), ("fused", m1)):
        by_lead = np.array([nse_lead(ob[:, h], sim[:, h]) for h in range(HOR)])
        out[tag] = {"pooled": float(nse_lead(ob.ravel(), sim.ravel())),
                    "median_lead": float(np.nanmedian(by_lead)),
                    "lead1": float(by_lead[0]), "lead96": float(by_lead[-1]),
                    "by_lead": [float(x) for x in by_lead]}
    return out


def fmt(tag, s):
    return (f"{tag} base pooled {s['base']['pooled']:.4f} medlead {s['base']['median_lead']:.4f} "
            f"(15min {s['base']['lead1']:.3f} 24h {s['base']['lead96']:.3f}) | "
            f"fused pooled {s['fused']['pooled']:.4f} medlead {s['fused']['median_lead']:.4f} "
            f"(15min {s['fused']['lead1']:.3f} 24h {s['fused']['lead96']:.3f}) | "
            f"delta pooled {s['fused']['pooled']-s['base']['pooled']:+.4f} "
            f"medlead {s['fused']['median_lead']-s['base']['median_lead']:+.4f}")


log = open(OUT / "train_log.jsonl", "a")
s_val = summarize(*run(val_origins))
s_test = summarize(*run(test_origins))
print("VAL0 " + fmt("", s_val), flush=True)
print("TEST0 " + fmt("", s_test), flush=True)
log.write(json.dumps({"step": 0, "val": s_val, "test": s_test}) + "\n")
best, best_step = s_val["fused"]["pooled"], 0
torch.save(adapter.state_dict(), OUT / "adapter_best.pt")   # step 0 (gate 0) == baseline
rng = np.random.default_rng(0)
t0 = time.time()
for step in range(1, STEPS + 1):
    ogs = rng.choice(train_origins, BATCH, replace=True)
    tg, fx, ff, y, lw = make_batch(ogs)
    with torch.no_grad():
        q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
    dq = adapter(ftok, lw)
    loss = FT.pinball(torch.clamp(q0 + dq, min=0.0) + 1e-6, y, LEVELS, wq)
    if not torch.isfinite(loss):
        print(f"step {step}: non-finite loss skipped", flush=True)
        opt.zero_grad()
        continue
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p_ for p_ in adapter.parameters() if p_.requires_grad], 1.0)
    opt.step()
    sched.step()
    if step % EVAL_EVERY == 0:
        s_val = summarize(*run(val_origins))
        tag = ""
        if s_val["fused"]["pooled"] > best:
            best, best_step = s_val["fused"]["pooled"], step
            torch.save(adapter.state_dict(), OUT / "adapter_best.pt")
            tag = "BEST"
        print(f"EVAL step {step} loss {loss.item():.4f} gate {adapter.gate.item():.3f} "
              f"({(time.time()-t0)/step:.2f}s/step) | val pooled base {s_val['base']['pooled']:.4f} "
              f"fused {s_val['fused']['pooled']:.4f} delta "
              f"{s_val['fused']['pooled']-s_val['base']['pooled']:+.4f} {tag}", flush=True)
        log.write(json.dumps({"step": step, "loss": float(loss.item()),
                              "gate": float(adapter.gate.item()), "val": s_val}) + "\n")
        log.flush()
torch.save(adapter.state_dict(), OUT / "adapter_last.pt")
s_last = summarize(*run(test_origins))
print("TEST_LAST " + fmt("", s_last), flush=True)
adapter.load_state_dict(torch.load(OUT / "adapter_best.pt", map_location=dev))
s_best = summarize(*run(test_origins))
print(f"TEST_BESTVAL(step {best_step}) " + fmt("", s_best), flush=True)
log.write(json.dumps({"final": True, "best_step": best_step, "test_last": s_last,
                      "test_bestval": s_best}) + "\n")
log.close()
print(f"EXP3TRAIN_DONE {NAME} K={K} big={int(BIG)} best_val_step={best_step} "
      f"test_base_pooled={s_best['base']['pooled']:.4f} "
      f"test_fused_pooled_bestval={s_best['fused']['pooled']:.4f} "
      f"test_fused_pooled_last={s_last['fused']['pooled']:.4f} "
      f"test_base_medlead={s_best['base']['median_lead']:.4f} "
      f"test_fused_medlead_bestval={s_best['fused']['median_lead']:.4f}", flush=True)
