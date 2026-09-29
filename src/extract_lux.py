"""ORBIT-2 (global-finetune 126M) latent extraction for the Luxembourg window.

Input : exp3/eu_store.npy  (days, 19, 128, 128) raw daily aggregates (ARCO order: lat DESC)
        exp3/eu_grid.json  window lat/lon
        exp3/*_0.25deg.npy global statics, (720, 1440), SOUTH-UP (row 0 = -90), lon 0..359.75
        exp3/normalize_{mean,std}.npz  (ERA5 0.25 deg stats from the HF repo)
Output: exp3/lux_latents.npz  pooled (days, 1024) fp16  [3x3 token box at basin]
        exp3/lux_tokens.npy   full-grid (days, 4096, 1024) fp16  [for later variants]

Conventions matched to the checkpoint:
  * orientation: the window is flipped to south-up so dynamic fields, statics
    and the lattitude channel share the checkpoint's row order;
  * the model is BUILT at the checkpoint's own grid (pos_embed loads exactly),
    then data_config() switches to the 128x128 window;
  * the model's on-the-fly pos-embed resize assumes a 2:1 grid and rescales;
    our input is already at the model's native 0.25 deg, so it is replaced by a
    CROP (geographic if the checkpoint grid is the full globe, else centred),
    which keeps the native token spacing in both directions.
Single GPU; a 1-rank gloo group satisfies the model's torch.distributed calls.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

F = Path(os.environ.get("FUSION_ROOT", "."))
E = F / "exp3"
sys.path.insert(0, str(F / "scripts"))
import extract_latents as EX  # noqa: E402  (stubs, Res_Slim_ViT, MODEL_KW)

RSV = sys.modules["climate_learn.models.hub.res_slimvit"]

STATICS = ["land_sea_mask", "orography", "lattitude", "landcover"]  # global yaml order
DYNAMIC = [
    "2m_temperature", "2m_temperature_max", "2m_temperature_min",
    "temperature_200", "temperature_500", "temperature_850",
    "10m_u_component_of_wind", "u_component_of_wind_200",
    "u_component_of_wind_500", "u_component_of_wind_850",
    "10m_v_component_of_wind", "v_component_of_wind_200",
    "v_component_of_wind_500", "v_component_of_wind_850",
    "specific_humidity_200", "specific_humidity_500", "specific_humidity_850",
    "total_precipitation_24hr", "volumetric_soil_water_layer_1",
]
GVARS = STATICS + DYNAMIC                          # 23 channels
STEP = 0.25
H = W = 128
BASIN_LATLON = (49.77, 6.09)                       # LUX_40 (Luxembourg) approx centroid
DEVICE = os.environ.get("LUX_DEVICE", "cuda")
SMOKE_DAYS = int(os.environ.get("LUX_SMOKE_DAYS", "0"))   # >0: CPU smoke test, encode only these days
OUTD = Path(os.environ.get("LUX_OUT", str(E)))
OUTD.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29633")
dist.init_process_group("gloo", rank=0, world_size=1)

# ---- dynamic window -> south-up ------------------------------------------
dyn = np.load(E / "eu_store.npy")                  # (days, 19, H, W)
days = dyn.shape[0]
assert dyn.shape[1:] == (19, H, W), dyn.shape
grid = json.load(open(E / "eu_grid.json"))
lat = np.asarray(grid["lat"], np.float64)
lon = np.asarray(grid["lon"], np.float64)
assert len(lat) == H and len(lon) == W, (len(lat), len(lon))
if lat[0] > lat[-1]:
    dyn = np.ascontiguousarray(dyn[:, :, ::-1, :])
    lat = lat[::-1].copy()
print(f"window lat {lat[0]}..{lat[-1]} (south-up) lon {lon[0]}..{lon[-1]}", flush=True)

# ---- statics: crop the global south-up grids -----------------------------
rows = np.round((lat + 90.0) / STEP).astype(int)
cols = np.round((lon % 360) / STEP).astype(int)
latg = np.load(E / "lattitude_0.25deg.npy").astype(np.float64)
GH, GW = latg.shape[-2:]
assert (GH, GW) == (720, 1440), latg.shape
assert np.allclose(latg.reshape(GH, GW)[rows, 0], lat), "static latitude rows do not match window"
stat = []
for v in STATICS:
    g = np.load(E / f"{v}_0.25deg.npy").astype(np.float32).reshape(GH, GW)
    stat.append(g[rows][:, cols])
stat = np.stack(stat)                              # (4, H, W)


def cell(la, lo):
    r = int(round((la - lat[0]) / STEP))
    c = int(round(((lo % 360) - (lon[0] % 360)) % 360 / STEP))
    return r, c


lux, atl, alp = cell(*BASIN_LATLON), cell(50.0, -12.0), cell(46.0, 8.0)
lsm, oro = stat[0], stat[1]
print(f"STATIC CHECK land_sea_mask lux={lsm[lux]:.2f} atlantic={lsm[atl]:.2f} | "
      f"orography lux={oro[lux]:.0f} alps={oro[alp]:.0f} | lat-channel row0={stat[2][0, 0]:.2f} "
      f"rowN={stat[2][-1, 0]:.2f}", flush=True)
assert lsm[lux] > 0.5 and lsm[atl] < 0.5, "static crop is wrong"
assert oro[alp] > oro[lux], "orography crop looks wrong"
t2m = dyn[:, 0].mean(0)
swv = dyn[:, 18].mean(0)                           # soil moisture: ~0 over sea, >0 over land
sea, land = lsm < 0.1, lsm > 0.9
print(f"DYNAMIC CHECK t2m south-row {t2m[0].mean():.1f}K north-row {t2m[-1].mean():.1f}K | "
      f"swvl1 sea {swv[sea].mean():.3f} land {swv[land].mean():.3f}", flush=True)
assert t2m[0].mean() > t2m[-1].mean(), "dynamic window is not south-up"
assert swv[land].mean() > 0.1 and swv[sea].mean() < 0.5 * swv[land].mean(), \
    "dynamic fields and static land-sea mask are misaligned"

# ---- normalization (checkpoint's own ERA5 0.25 deg stats) ----
nm, ns = np.load(E / "normalize_mean.npz"), np.load(E / "normalize_std.npz")


def stat_of(store, v):
    for cand in EX.KEY_ALIAS.get(v, [v]):
        if cand in store.files:
            return float(np.asarray(store[cand]).ravel()[0])
    raise KeyError(v)


mean = np.array([stat_of(nm, v) for v in GVARS], np.float32)
std = np.array([stat_of(ns, v) for v in GVARS], np.float32)

# ---- model at the checkpoint's own grid ----------------------------------
p = EX.MODEL_KW["patch_size"]
Hp, Wp = H // p, W // p                            # 64 x 64 tokens
ck = torch.load(E / "global_126m_precipitation.ckpt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
Lck = sd["pos_embed"].shape[1]
oh = int(round((Lck // 2) ** 0.5))
ow = 2 * oh
assert oh * ow == Lck, f"checkpoint pos_embed {Lck} tokens is not a 2:1 grid"
print(f"checkpoint pos grid {oh}x{ow} tokens (= {oh*p}x{ow*p} px)", flush=True)
dev = torch.device(DEVICE)
model = EX.Res_Slim_ViT(default_vars=GVARS, img_size=(oh * p, ow * p),
                        in_channels=len(GVARS), out_channels=1, history=1,
                        **EX.MODEL_KW)
msd = model.state_dict()
dropped = [k for k in sd if k in msd and sd[k].shape != msd[k].shape]
unused = [k for k in sd if k not in msd]
for k in dropped + unused:
    del sd[k]
msg = model.load_state_dict(sd, strict=False)
print("load_state_dict:", msg, "| shape-dropped:", dropped, "| unused:", unused[:8], flush=True)
assert not dropped, dropped
assert not [k for k in msg.missing_keys if not k.startswith(("head", "path2"))], msg.missing_keys
model = model.to(dev)

if (oh, ow) == (GH // p, GW // p):                 # full-globe grid: geographic crop
    rr = rows[::p] // p
    cc = cols[::p] // p
    how = "geographic"
else:                                              # tile-relative grid: centred crop
    assert oh >= Hp and ow >= Wp, (oh, ow)
    rr = np.arange((oh - Hp) // 2, (oh - Hp) // 2 + Hp)
    cc = np.arange((ow - Wp) // 2, (ow - Wp) // 2 + Wp)
    how = "centred"
pe_idx = torch.from_numpy((rr[:, None] * ow + cc[None, :]).ravel()).to(dev)
print(f"pos_embed: {how} crop rows {rr[0]}..{rr[-1]} cols {cc[0]}..{cc[-1]}", flush=True)


def crop_pos_embed(pos_embed, patch_size, new_size):
    assert (new_size[0] // patch_size, new_size[1] // patch_size) == (Hp, Wp), new_size
    return pos_embed.index_select(1, pe_idx)


RSV.interpolate_pos_embed_on_the_fly = crop_pos_embed
model.data_config(28, (H, W), len(GVARS), 1)       # 0.25 deg ~ 28 km (yaml ERA5-IMERG-FUSED)
model.eval()

br, bc = lux[0] // p, lux[1] // p
box = [(r, c) for r in range(br - 1, br + 2) for c in range(bc - 1, bc + 2)]
box_idx = [r * Wp + c for r, c in box]
print(f"basin token box rows {br-1}..{br+1} cols {bc-1}..{bc+1} idx {box_idx}", flush=True)

tokens = np.zeros((days, Hp * Wp, EX.MODEL_KW["embed_dim"]), np.float16)
# optional side output for top-block fine-tuning cells: the input of the LAST
# transformer block at the basin box tokens (fail-safe: never blocks the main output)
h7, cap = None, {}
try:
    hook = model.blocks[-2].register_forward_hook(lambda m, i, o: cap.__setitem__("x", o))
    box_t = torch.tensor(box_idx, device=dev)
    h7 = np.zeros((days, len(box_idx), EX.MODEL_KW["embed_dim"]), np.float16)
except Exception as ex:  # noqa: BLE001
    print("H7 capture disabled:", repr(ex), flush=True)
wx = np.zeros((days, len(GVARS), H, W), np.float16)   # exact normalized model inputs (in-loop FT)
mean_t = torch.from_numpy(mean).to(dev)[None, :, None, None]
std_t = torch.from_numpy(std).to(dev)[None, :, None, None]
stat_t = torch.from_numpy(stat).to(dev)
t0 = time.time()
with torch.no_grad():
    n_enc = SMOKE_DAYS if SMOKE_DAYS > 0 else days
    for s in range(0, n_enc, 8):
        d = torch.from_numpy(dyn[s:min(s + 8, n_enc)]).to(dev)
        x = torch.cat([stat_t[None].expand(d.shape[0], -1, -1, -1), d], 1)
        x = torch.nan_to_num(x, nan=0.0)
        x = (x - mean_t) / std_t
        wx[s:s + x.shape[0]] = x.cpu().numpy().astype(np.float16)
        if s == 0:
            zm = x.mean(dim=(0, 2, 3)).cpu().numpy()
            zs = x.std(dim=(0, 2, 3)).cpu().numpy()
            print("UNIT CHECK (normalized mean/std per channel, want |mean|<~3, std O(1)):", flush=True)
            for i, v in enumerate(GVARS):
                flag = "  <-- SUSPECT" if (abs(zm[i]) > 3 or zs[i] < 0.05) else ""
                print(f"  {v}: {zm[i]:+.2f} / {zs[i]:.2f}{flag}", flush=True)
        tok = model.forward_encoder(x.float(), GVARS)      # (B, L, D)
        assert tok.shape[1] == Hp * Wp, tok.shape
        tokens[s:s + x.shape[0]] = tok.float().cpu().numpy().astype(np.float16)
        if h7 is not None:
            try:
                h7[s:s + x.shape[0]] = cap.pop("x").index_select(1, box_t).float().cpu().numpy().astype(np.float16)
            except Exception as ex:  # noqa: BLE001
                print("H7 capture failed, disabled:", repr(ex), flush=True)
                h7 = None
print(f"encoded {days} days in {time.time()-t0:.0f}s", flush=True)

if h7 is not None:
    try:
        top = f"blocks.{len(model.blocks) - 1}."
        top_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
                  if k.startswith(top) or k.startswith("norm.")}
        np.save(OUTD / "lux_h7_box.npy", h7)
        torch.save({"top_sd": top_sd, "box_idx": box_idx}, OUTD / "lux_topft_meta.pt")
        print(f"H7 saved {h7.shape}, top_sd keys {len(top_sd)} ({top}*, norm.*)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print("H7 save failed:", repr(ex), flush=True)

np.save(OUTD / "lux_wx.npy", wx)
json.dump({"ckpt_grid": [oh, ow], "pos_rows": [int(v) for v in rr], "pos_cols": [int(v) for v in cc],
           "box_idx": [int(v) for v in box_idx], "gvars": GVARS, "res_km": 28, "img": [H, W],
           "pos_crop": how}, open(OUTD / "lux_model_meta.json", "w"))
print("saved lux_wx.npy", wx.shape, "+ lux_model_meta.json", flush=True)
if SMOKE_DAYS == 0:
    np.save(OUTD / "lux_tokens.npy", tokens)
pooled = tokens[:, box_idx].astype(np.float32).mean(1)     # (days, 1024)
dates = json.load(open(E / "eu_dates.json"))
assert len(dates) == days
np.savez(OUTD / "lux_latents.npz", latents=pooled.astype(np.float16),
         dates=np.array(dates), box=np.array(box_idx))
print(f"latent stats: mean {pooled.mean():+.3f} std {pooled.std():.3f} "
      f"day-to-day corr {np.corrcoef(pooled[:-1].ravel(), pooled[1:].ravel())[0, 1]:.3f}", flush=True)
print("LUX_EXTRACT_DONE", pooled.shape, flush=True)
