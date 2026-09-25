"""ORBIT-2 latent extraction for TS-model fusion.

Per SLURM rank: process year-shards [rank::world_size] from the fine-tune
dataset (ERA5-Daymet fused, 15 arcmin input), run the frozen Res_Slim_ViT
encoder, pool patch tokens per CAMELS basin, save (days, basins, D) float16.

Run (single-rank smoke):
  SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 python extract_latents.py \
      --data-dir .../15.0_arcmin/v4 --split train --limit-files 1
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# MIOpen conv kernels are flaky in this env (miopenStatusInternalError);
# the only convs here are tiny PatchEmbeds, native fallback costs nothing.
torch.backends.cudnn.enabled = False

HERE = Path(__file__).resolve().parent
FUSION = HERE.parent

# ---- xformers shim (attention.py imports it at module level; the DEFAULT
# ---- branch we use only calls torch F.scaled_dot_product_attention) --------
if "xformers" not in sys.modules:
    xf = types.ModuleType("xformers")
    xf.ops = types.SimpleNamespace()
    comp = types.ModuleType("xformers.components")
    att = types.ModuleType("xformers.components.attention")
    core = types.ModuleType("xformers.components.attention.core")
    def _no_xformers(*a, **k):
        raise RuntimeError("xformers shim: CK path not available")
    core.scaled_dot_product_attention = _no_xformers
    xf.components = comp
    comp.attention = att
    att.core = core
    for name, mod in [("xformers", xf), ("xformers.components", comp),
                      ("xformers.components.attention", att),
                      ("xformers.components.attention.core", core)]:
        sys.modules[name] = mod

# ---- bypass heavy package __init__s (loaders/visualize pull matplotlib etc.)
SRC = FUSION / "ORBIT-2" / "src"
CL = SRC / "climate_learn"
for name, path in [("climate_learn", CL),
                   ("climate_learn.models", CL / "models"),
                   ("climate_learn.models.hub", CL / "models" / "hub"),
                   ("climate_learn.utils", CL / "utils")]:
    stub = types.ModuleType(name)
    stub.__path__ = [str(path)]
    sys.modules[name] = stub

from climate_learn.models.hub.res_slimvit import Res_Slim_ViT          # noqa: E402
from climate_learn.models.hub.components.pos_embed import (            # noqa: E402
    interpolate_pos_embed,
)
from climate_learn.utils.fused_attn import FusedAttn                   # noqa: E402
import torch.distributed as dist                                       # noqa: E402

DEFAULT_VARS = [
    "land_sea_mask", "orography", "lattitude", "landcover",
    "2m_temperature", "2m_temperature_max", "2m_temperature_min",
    "temperature_200", "temperature_500", "temperature_850",
    "10m_u_component_of_wind", "u_component_of_wind_200",
    "u_component_of_wind_500", "u_component_of_wind_850",
    "10m_v_component_of_wind", "v_component_of_wind_200",
    "v_component_of_wind_500", "v_component_of_wind_850",
    "specific_humidity_200", "specific_humidity_500",
    "specific_humidity_850", "total_precipitation_24hr",
    "volumetric_soil_water_layer_1",
]
IN_VARS = [
    "land_sea_mask", "landcover", "orography", "lattitude",
    "2m_temperature", "temperature_200", "temperature_500",
    "temperature_850", "u_component_of_wind_200", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_200",
    "v_component_of_wind_500", "v_component_of_wind_850",
    "specific_humidity_200", "specific_humidity_500",
    "specific_humidity_850", "total_precipitation_24hr",
    "volumetric_soil_water_layer_1",
]
# shard key aliases (daymet files use 'latitude' + 'lattitude' both)
KEY_ALIAS = {"lattitude": ["lattitude", "latitude"]}

MODEL_KW = dict(superres_mag=4, cnn_ratio=4, patch_size=2, embed_dim=1024,
                depth=8, decoder_depth=4, num_heads=16, mlp_ratio=4,
                drop_path=0.1, drop_rate=0.1, tensor_par_size=1,
                tensor_par_group=None, FusedAttn_option=FusedAttn.DEFAULT)
SPATIAL_RES_KM = 28


def load_shard(path, in_vars):
    z = np.load(path)
    chans = []
    for v in in_vars:
        key = None
        for cand in KEY_ALIAS.get(v, [v]):
            if cand in z.files:
                key = cand
                break
        if key is None:
            raise KeyError(f"{v} not in {path} (has {z.files[:8]}...)")
        chans.append(z[key][:, 0].astype(np.float32))   # (days, H, W)
    return np.stack(chans, axis=1)                       # (days, C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help=".../15.0_arcmin/vX")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt", default=str(FUSION / "ckpt/us_126m_precipitation.ckpt"))
    ap.add_argument("--masks", default=str(FUSION / "masks/basin_patch_masks.npz"))
    ap.add_argument("--norm-dir", default=None, help="dir with normalize_mean/std.npz (default: data-dir)")
    ap.add_argument("--out-dir", default=str(FUSION / "latents"))
    ap.add_argument("--batch-days", type=int, default=16)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--res-km", type=int, default=18)
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29612")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
    dev = torch.device("cuda")

    ddir = Path(args.data_dir)
    norm_dir = Path(args.norm_dir) if args.norm_dir else ddir
    nm = np.load(norm_dir / "normalize_mean.npz")
    ns = np.load(norm_dir / "normalize_std.npz")

    def norm_stat(store, v):
        for cand in KEY_ALIAS.get(v, [v]):
            if cand in store.files:
                return float(np.asarray(store[cand]).ravel()[0])
        raise KeyError(v)

    mean = np.array([norm_stat(nm, v) for v in IN_VARS], dtype=np.float32)
    std = np.array([norm_stat(ns, v) for v in IN_VARS], dtype=np.float32)

    la = np.load(ddir / "lat.npy"); lo = np.load(ddir / "lon.npy")
    H, W = len(la), len(lo)
    p = MODEL_KW["patch_size"]
    Hp, Wp = H // p, W // p

    # basin masks: npz with per-basin sparse patch indices + weights
    mz = np.load(args.masks, allow_pickle=True)
    if "grid" in mz.files:
        assert tuple(mz["grid"][:2]) == (Hp, Wp), \
            f"mask grid {mz['grid'][:2]} != data patch grid {(Hp, Wp)}"
    basin_ids = list(mz["basin_ids"])
    nb = len(basin_ids)
    Wmat = np.zeros((nb, Hp * Wp), dtype=np.float32)
    for bi, (idx, w) in enumerate(zip(mz["patch_idx"], mz["patch_w"])):
        Wmat[bi, np.asarray(idx, dtype=np.int64)] = np.asarray(w, np.float32)
    Wmat = torch.from_numpy(Wmat / Wmat.sum(1, keepdims=True)).to(dev)

    model = Res_Slim_ViT(default_vars=DEFAULT_VARS, img_size=(H, W),
                         in_channels=len(IN_VARS), out_channels=1, history=1,
                         **MODEL_KW).to(dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    msd = model.state_dict()
    for k in list(sd.keys()):
        if k not in msd:
            del sd[k]
        elif sd[k].shape != msd[k].shape:
            if k == "pos_embed":
                interpolate_pos_embed(model, sd, new_size=(H, W))
            else:
                del sd[k]
    msg = model.load_state_dict(sd, strict=False)
    if rank == 0:
        print("load_state_dict:", msg, flush=True)
    model.data_config(args.res_km, (H, W), len(IN_VARS), 1)
    model.eval()

    files = sorted(f for f in (ddir / args.split).glob("*.npz")
                   if not f.name.startswith("climatology"))
    if args.limit_files:
        files = files[: args.limit_files]
    mine = files[rank::world]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rank {rank}/{world}] {len(mine)} shards on {dev}", flush=True)

    for fp in mine:
        t0 = time.time()
        arr = load_shard(fp, IN_VARS)                     # (days, C, H, W)
        days = arr.shape[0]
        out = np.zeros((days, nb, MODEL_KW["embed_dim"]), dtype=np.float16)
        with torch.no_grad():
            for s in range(0, days, args.batch_days):
                x = torch.from_numpy(arr[s:s + args.batch_days]).to(dev)
                x = torch.nan_to_num(x, nan=0.0)
                x = (x - torch.from_numpy(mean).to(dev)[None, :, None, None]) \
                    / torch.from_numpy(std).to(dev)[None, :, None, None]
                tok = model.forward_encoder(x, IN_VARS)   # (B, L, D)
                pooled = torch.einsum("nl,bld->bnd", Wmat, tok)
                out[s:s + x.shape[0]] = pooled.float().cpu().numpy().astype(np.float16)
        np.savez_compressed(out_dir / f"{fp.stem}_latents.npz",
                            latents=out, basin_ids=np.array(basin_ids),
                            source=str(fp))
        print(f"[rank {rank}] {fp.name}: {days}d x {nb}b in {time.time()-t0:.1f}s",
              flush=True)
    print(f"[rank {rank}] EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()

