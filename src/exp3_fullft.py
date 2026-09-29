"""exp3_fullft: Luxembourg sub-hourly (CAMELS-LUX 40, 15-min flow, 24 h ahead)
with ORBIT-2 (global 126M) IN THE TRAINING LOOP.

MODE=ft     : every ORBIT-2 encoder parameter is fine-tuned (lr x ORBIT_LR_SCALE)
MODE=frozen : the same script with ORBIT-2 frozen -> apples-to-apples reference
The time-series backbone stays frozen; the fusion adapter is the 100M 'big'
configuration used by the frozen cells.

Inputs : exp3/lux_wx.npy          exact normalized ORBIT-2 inputs saved by extract_lux.py
         exp3/lux_model_meta.json checkpoint grid, pos-embed crop, basin token box
Sampling: each rank draws one training DAY per step and B_LOCAL origins on that
day, so all its samples share one latent window (one LAT_PAST-day ORBIT-2 pass
per rank); effective batch = world * B_LOCAL.
Protocol, splits and metrics are identical to exp3_train.py.

usage (under srun, one task per GCD):
  python exp3_fullft.py NAME K MODE [STEPS=400] [LAT_PAST=30] [B_LOCAL=1] [BUDGET_MIN=95]
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

F = Path(os.environ.get("FUSION_ROOT", "."))
E = F / "exp3"
LUX_CSV = Path(os.environ.get(        # the time-series model's sub-hourly example basin
    "LUX_CSV", Path(os.environ.get("TS_MODEL_DIR", F / "ts_model"))
    / "Example/data/exp3_subhourly_lux/LUX_40.csv"))
sys.path.insert(0, str(F / "scripts"))
import extract_latents as EX  # noqa: E402  (stubs + Res_Slim_ViT; disables MIOpen convs)
import fusion_train as FT     # noqa: E402
from fixload import load_pipeline  # noqa: E402

RSV = sys.modules["climate_learn.models.hub.res_slimvit"]

NAME, K, MODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert MODE in ("ft", "frozen"), MODE
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
LAT_PAST = int(sys.argv[5]) if len(sys.argv) > 5 else 30
B_LOCAL = int(sys.argv[6]) if len(sys.argv) > 6 else 1
BUDGET_MIN = float(sys.argv[7]) if len(sys.argv) > 7 else 95.0
LR, ORBIT_LR_SCALE = 2e-4, 0.05
CHUNK = 5                                   # days per checkpointed ORBIT-2 forward
DEVICE = os.environ.get("EXP3_DEVICE", "cuda")
BACKEND = os.environ.get("EXP3_BACKEND", "nccl")
IN = Path(os.environ.get("EXP3_IN", str(E)))                # extraction outputs
SMOKE = os.environ.get("EXP3_SMOKE") == "1"                 # CPU smoke test (gloo, world 1)
T_START = time.time()

CTX, HOR = 8760, 96
FT.CTX, FT.HOR = CTX, HOR
FT.LAT_PAST, FT.LAT_FUT = LAT_PAST, 0
COV_COLS = ["Precip", "AirTemp"]
EVAL_YEAR, N_ORIGINS = 2021, 30

rank = int(os.environ.get("SLURM_PROCID", 0))
world = int(os.environ.get("SLURM_NTASKS", 1))
local = int(os.environ.get("SLURM_LOCALID", 0)) % max(torch.cuda.device_count(), 1)
if DEVICE == "cuda":
    torch.cuda.set_device(local)
dist.init_process_group(BACKEND, init_method="env://", rank=rank, world_size=world,
                        timeout=dt.timedelta(minutes=30))
r0 = rank == 0
dev = torch.device(DEVICE)


def log0(*a):
    if r0:
        print(*a, flush=True)


OUT = F / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)

# ---- data (identical to exp3_train.py) -------------------------------------
rows = [l.rstrip("\n").split(",") for l in
        open(LUX_CSV)]
hdr = rows[0]
ci = {c: hdr.index(c) for c in hdr}
ts = [dt.datetime.fromisoformat(r[ci["datetime"]]) for r in rows[1:]]
rec = np.asarray([[float(r[ci[c]]) if r[ci[c]] not in ("", "nan") else np.nan
                   for c in ["Q"] + COV_COLS] for r in rows[1:]], np.float32)
flow, cov = rec[:, 0], rec[:, 1:]
n = len(ts)

meta = json.load(open(IN / "lux_model_meta.json"))
lz = np.load(IN / "lux_latents.npz")
lat_day0 = dt.date.fromisoformat(str(lz["dates"][0]))
WX = np.load(IN / "lux_wx.npy", mmap_mode="r")            # (days, 23, 128, 128) fp16
lat_days = WX.shape[0]
assert lat_days == lz["latents"].shape[0], (WX.shape, lz["latents"].shape)


def day_of(og):
    d = (ts[og].date() - lat_day0).days
    assert LAT_PAST <= d <= lat_days, (ts[og], d)
    return d


years = np.array([t.year for t in ts])
pos = np.where(years == EVAL_YEAR)[0]
pos = pos[(pos >= CTX) & (pos + HOR <= n)]
test_origins = pos[np.linspace(0, len(pos) - 1, N_ORIGINS).astype(int)]
lo = max(CTX, next(i for i, t in enumerate(ts) if (t.date() - lat_day0).days >= LAT_PAST))
tr_hi = next(i for i, t in enumerate(ts) if t >= dt.datetime(2020, 11, 1))
train_origins = np.arange(lo, tr_hi - HOR)
vl_hi = next(i for i, t in enumerate(ts) if t >= dt.datetime(2021, 1, 1))
val_origins = np.linspace(tr_hi, vl_hi - HOR, N_ORIGINS).astype(int)
EVAL_EVERY, N_EQUIV = 50, 16
if SMOKE:
    val_origins, test_origins, EVAL_EVERY, N_EQUIV = val_origins[:2], test_origins[:2], 1, 2
tdays = {}
for og in train_origins:
    tdays.setdefault(day_of(og), []).append(int(og))
train_day_list = sorted(tdays)
log0(f"EXP3FT {NAME}: mode={MODE} K={K}/{HOR} lat_past={LAT_PAST} steps={STEPS} "
     f"world={world} b_local={B_LOCAL} (eff batch {world * B_LOCAL}) budget={BUDGET_MIN}min | "
     f"train {len(train_origins)} origins on {len(train_day_list)} days | val {len(val_origins)} | "
     f"test {len(test_origins)}")

# ---- ORBIT-2, built exactly as extract_lux.py --------------------------------
p = EX.MODEL_KW["patch_size"]
H, W = meta["img"]
GVARS = meta["gvars"]
oh, ow = meta["ckpt_grid"]
pe_idx = torch.tensor((np.asarray(meta["pos_rows"])[:, None] * ow
                       + np.asarray(meta["pos_cols"])[None, :]).ravel(), device=dev)
box_t = torch.tensor(meta["box_idx"], device=dev)
assert pe_idx.numel() == (H // p) * (W // p)

ck = torch.load(E / "global_126m_precipitation.ckpt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
assert sd["pos_embed"].shape[1] == oh * ow, sd["pos_embed"].shape
m = EX.Res_Slim_ViT(default_vars=GVARS, img_size=(oh * p, ow * p), in_channels=len(GVARS),
                    out_channels=1, history=1, **EX.MODEL_KW)
msd = m.state_dict()
dropped = [k for k in sd if k in msd and sd[k].shape != msd[k].shape]
assert not dropped, dropped
for k in [k for k in sd if k not in msd]:
    del sd[k]
msg = m.load_state_dict(sd, strict=False)
assert not [k for k in msg.missing_keys if not k.startswith(("head", "path2"))], msg.missing_keys
del ck, sd
RSV.interpolate_pos_embed_on_the_fly = lambda pos_embed, patch_size, new_size: \
    pos_embed.index_select(1, pe_idx)
m.data_config(meta["res_km"], (H, W), len(GVARS), 1)
m = m.to(dev)
m.eval()                                           # no dropout: FT and frozen differ only in weights
for q in m.parameters():
    q.requires_grad_(False)
if MODE == "ft":
    # trainable = exactly the tensors forward_encoder touches (decoder stays frozen,
    # pos_embed is the checkpoint's fixed sin-cos table)
    for q in m.parameters():
        q.requires_grad_(True)
    m.pos_embed.requires_grad_(False)
    xs = torch.from_numpy(np.ascontiguousarray(WX[:1])).to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        m.forward_encoder(xs.float(), GVARS).float().sum().backward()
    unused = [n_ for n_, q in m.named_parameters() if q.requires_grad and q.grad is None]
    for q in m.parameters():
        if q.grad is None:
            q.requires_grad_(False)
        q.grad = None
    log0(f"ORBIT-2: froze {len(unused)} tensors unused by the encoder, e.g. {unused[:4]}")
orb_params = [q for q in m.parameters() if q.requires_grad]
log0(f"ORBIT-2 trainable params: {sum(q.numel() for q in orb_params)/1e6:.1f}M "
     f"(total {sum(q.numel() for q in m.parameters())/1e6:.1f}M)")


class OrbitPool(nn.Module):
    """ORBIT-2 encoder -> mean of the basin box tokens: one 1024-vector per day."""

    def __init__(self, model, trainable):
        super().__init__()
        self.m = model
        self.trainable = trainable

    def chunk(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tok = self.m.forward_encoder(x.float(), GVARS)
        return tok.index_select(1, box_t).float().mean(1)

    def forward(self, x):                          # x (n_days, 23, H, W)
        outs = []
        for s in range(0, x.shape[0], CHUNK):
            c = x[s:s + CHUNK]
            if self.trainable and torch.is_grad_enabled():
                outs.append(torch.utils.checkpoint.checkpoint(self.chunk, c, use_reentrant=False))
            else:
                outs.append(self.chunk(c))
        return torch.cat(outs, 0)


class Joint(nn.Module):
    def __init__(self, orbit, adapter):
        super().__init__()
        self.orbit = orbit
        self.adapter = adapter

    def forward(self, ftok, x):
        lat = self.orbit(x)                                        # (LAT_PAST, D)
        return self.adapter(ftok, lat.unsqueeze(0).expand(ftok.shape[0], -1, -1))


orbit = OrbitPool(m, trainable=MODE == "ft")
pipe = load_pipeline(os.environ.get("TS_MODEL_DIR", F / "ts_model"), device=DEVICE)
pipe.model.eval()
for q in pipe.model.parameters():
    q.requires_grad_(False)
adapter = FT.FusionAdapter(d_h=1024, n_heads=16, n_blocks=4, per_day=True,
                           latent_layers=4).to(dev)
joint = Joint(orbit, adapter).to(dev)
net = nn.parallel.DistributedDataParallel(joint, device_ids=[local]) if world > 1 else joint
log0(f"adapter params: {sum(q.numel() for q in adapter.parameters())/1e6:.2f}M")
groups = [{"params": list(adapter.parameters()), "lr": LR}]
if orb_params:
    groups.append({"params": orb_params, "lr": LR * ORBIT_LR_SCALE})
opt = torch.optim.AdamW(groups, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, STEPS))
trainables = list(adapter.parameters()) + orb_params
LEVELS = pipe.model.quantile_levels.to(dev).float()
NQ = LEVELS.numel()
wq = torch.ones(NQ, device=dev)
wq[NQ // 2] = 3.0
wq = wq / wq.mean()


def ts_batch(ogs):
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
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float()
    return t(tg), t(fx), t(ff), t(y)


@torch.no_grad()
def latents_for(days_needed):
    out = {}
    ds = sorted(days_needed)
    for s in range(0, len(ds), 8):
        dd = ds[s:s + 8]
        lat = orbit.chunk(torch.from_numpy(np.ascontiguousarray(WX[dd])).to(dev))
        for j, d in enumerate(dd):
            out[d] = lat[j]
    return out


def nse_lead(obs, sim, min_samples=24):
    msk = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[msk], sim[msk]
    if len(o) < min_samples:
        return np.nan
    den = np.sum((o - o.mean()) ** 2)
    scale = max(abs(o.mean()), 1.0)
    if den <= (1e-10 * scale) ** 2 * len(o):
        return np.nan
    return 1.0 - np.sum((s - o) ** 2) / den


@torch.no_grad()
def run(origins, bs=8):
    need = set()
    for og in origins:
        d = day_of(og)
        need.update(range(d - LAT_PAST, d))
    LD = latents_for(need)
    m0s, m1s, obs = [], [], []
    for c0 in range(0, len(origins), bs):
        chunk = origins[c0:c0 + bs]
        tg, fx, ff, y = ts_batch(chunk)
        lw = torch.stack([torch.stack([LD[d] for d in range(day_of(og) - LAT_PAST, day_of(og))])
                          for og in chunk])
        q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
        dq = adapter(ftok, lw)
        m0s.append(torch.clamp(q0, min=0.0)[:, NQ // 2].cpu().numpy())
        m1s.append(torch.clamp(q0 + dq, min=0.0)[:, NQ // 2].cpu().numpy())
        obs.append(y.cpu().numpy())
    return np.concatenate(obs), np.concatenate(m0s), np.concatenate(m1s)


def summarize(ob, m0, m1):
    out = {}
    for tag, sim in (("base", m0), ("fused", m1)):
        by_lead = np.array([nse_lead(ob[:, h], sim[:, h]) for h in range(HOR)])
        out[tag] = {"pooled": float(nse_lead(ob.ravel(), sim.ravel())),
                    "median_lead": float(np.nanmedian(by_lead)),
                    "lead1": float(by_lead[0]), "lead96": float(by_lead[-1]),
                    "by_lead": [float(v) for v in by_lead]}
    return out


def fmt(s):
    return (f"base pooled {s['base']['pooled']:.4f} medlead {s['base']['median_lead']:.4f} "
            f"(15min {s['base']['lead1']:.3f} 24h {s['base']['lead96']:.3f}) | "
            f"fused pooled {s['fused']['pooled']:.4f} medlead {s['fused']['median_lead']:.4f} "
            f"(15min {s['fused']['lead1']:.3f} 24h {s['fused']['lead96']:.3f}) | "
            f"delta pooled {s['fused']['pooled']-s['base']['pooled']:+.4f} "
            f"medlead {s['fused']['median_lead']-s['base']['median_lead']:+.4f}")


def cpu_state():
    return {k: v.detach().cpu().clone() for k, v in joint.state_dict().items()}


log = None
best, best_step, best_state = -1e9, 0, None
if r0:
    log = open(OUT / "train_log.jsonl", "a")
    chk = latents_for(range(N_EQUIV))
    a = torch.stack([chk[d] for d in range(N_EQUIV)]).cpu().numpy()
    b = lz["latents"][:N_EQUIV].astype(np.float32)
    print(f"EQUIV in-loop (bf16) vs extracted latents, {N_EQUIV} days: rel err "
          f"{np.linalg.norm(a - b) / np.linalg.norm(b):.4f}", flush=True)
    s_val = summarize(*run(val_origins))
    s_test = summarize(*run(test_origins))
    print("VAL0 " + fmt(s_val), flush=True)
    print("TEST0 " + fmt(s_test), flush=True)
    log.write(json.dumps({"step": 0, "val": s_val, "test": s_test}) + "\n")
    log.flush()
    best, best_step, best_state = s_val["fused"]["pooled"], 0, cpu_state()
dist.barrier()

rng = np.random.default_rng(1000 + rank)
t0 = time.time()
flag = torch.zeros(1, device=dev)
last_step = 0
for step in range(1, STEPS + 1):
    d = int(train_day_list[rng.integers(len(train_day_list))])
    ogs = rng.choice(tdays[d], B_LOCAL, replace=len(tdays[d]) < B_LOCAL)
    x = torch.from_numpy(np.ascontiguousarray(WX[d - LAT_PAST:d])).to(dev)
    tg, fx, ff, y = ts_batch(ogs)
    with torch.no_grad():
        q0, ftok = FT.frozen_forward(pipe, tg, fx, ff, dev)
    dq = net(ftok, x)
    loss = FT.pinball(torch.clamp(q0 + dq, min=0.0) + 1e-6, y, LEVELS, wq)
    ok = torch.isfinite(loss.detach()).float().reshape(1)
    if world > 1:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    opt.zero_grad(set_to_none=True)
    if ok.item() < 1:
        log0(f"step {step}: non-finite loss on some rank, skipped")
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(trainables, 1.0)
        opt.step()
    sched.step()
    last_step = step
    if r0 and (step in (5, 10, 25) or step % 50 == 0):
        print(f"step {step} loss {loss.item():.4f} gate {adapter.gate.item():.3f} "
              f"({(time.time()-t0)/step:.2f}s/step, elapsed {(time.time()-T_START)/60:.1f}min)",
              flush=True)
    if step % EVAL_EVERY == 0 or step == STEPS:
        if r0:
            s_val = summarize(*run(val_origins))
            tag = ""
            if s_val["fused"]["pooled"] > best:
                best, best_step, best_state = s_val["fused"]["pooled"], step, cpu_state()
                tag = "BEST"
            print(f"EVAL step {step}: val pooled base {s_val['base']['pooled']:.4f} fused "
                  f"{s_val['fused']['pooled']:.4f} delta "
                  f"{s_val['fused']['pooled']-s_val['base']['pooled']:+.4f} {tag}", flush=True)
            log.write(json.dumps({"step": step, "loss": float(loss.item()),
                                  "gate": float(adapter.gate.item()), "val": s_val}) + "\n")
            log.flush()
        dist.barrier()
    if step % 10 == 0:
        flag.fill_(1.0 if (r0 and (time.time() - T_START) / 60 > BUDGET_MIN) else 0.0)
        if world > 1:
            dist.broadcast(flag, 0)
        if flag.item() > 0:
            log0(f"BUDGET reached at step {step} ({(time.time()-T_START)/60:.1f} min): stopping")
            break

if r0:
    if last_step % EVAL_EVERY != 0 and last_step != STEPS:        # evaluate the final weights once more
        s_val = summarize(*run(val_origins))
        if s_val["fused"]["pooled"] > best:
            best, best_step, best_state = s_val["fused"]["pooled"], last_step, cpu_state()
        print(f"EVAL step {last_step}: val pooled fused {s_val['fused']['pooled']:.4f}", flush=True)
    s_last = summarize(*run(test_origins))
    print("TEST_LAST " + fmt(s_last), flush=True)
    joint.load_state_dict(best_state)
    s_best = summarize(*run(test_origins))
    print(f"TEST_BESTVAL(step {best_step}) " + fmt(s_best), flush=True)
    torch.save({k[len("adapter."):]: v for k, v in best_state.items() if k.startswith("adapter.")},
               OUT / "adapter_best.pt")
    if MODE == "ft":
        torch.save({k[len("orbit.m."):]: v for k, v in best_state.items() if k.startswith("orbit.m.")},
                   OUT / "orbit_best.pt")
    log.write(json.dumps({"final": True, "best_step": best_step, "last_step": last_step,
                          "test_last": s_last, "test_bestval": s_best}) + "\n")
    log.close()
    print(f"EXP3FT_DONE {NAME} mode={MODE} K={K} steps_run={last_step} best_val_step={best_step} "
          f"test_base_pooled={s_best['base']['pooled']:.4f} "
          f"test_fused_pooled_bestval={s_best['fused']['pooled']:.4f} "
          f"test_fused_pooled_last={s_last['fused']['pooled']:.4f} "
          f"test_base_medlead={s_best['base']['median_lead']:.4f} "
          f"test_fused_medlead_bestval={s_best['fused']['median_lead']:.4f}", flush=True)
dist.barrier()
dist.destroy_process_group()
