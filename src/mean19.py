"""Basin-mean table of the 19 ORBIT-2 input variables (fairness baseline).

Phase "extract" (rank-sharded): per year shard, 2x2 avg-pool to the 90x180
patch grid, then weighted-average with the same normalized basin-patch weights
the latent extractor uses -> mean19/{stem}_m19.npz (days, nb, 19) fp32.
Phase "merge": assemble dataset/forcing19.npy (15330, 671, 19).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

F = Path(os.environ.get("FUSION_ROOT", "."))
D = Path(os.environ.get("ERA5_DAYMET_DIR", "era5_daymet"))

src = (F / "scripts/extract_latents.py").read_text()
ns = {}
exec(re.search(r"IN_VARS = \[.*?\]", src, re.S).group(0), ns)
m = re.search(r"KEY_ALIAS = \{.*?\}", src, re.S)
exec(m.group(0) if m else "KEY_ALIAS = {}", ns)
IN_VARS, KEY_ALIAS = ns["IN_VARS"], ns["KEY_ALIAS"]

mz = np.load(F / "masks/basin_patch_masks.npz", allow_pickle=True)
ids = list(mz["basin_ids"])
nb = len(ids)
Wm = np.zeros((nb, 90 * 180), np.float32)
for bi, (idx, w) in enumerate(zip(mz["patch_idx"], mz["patch_w"])):
    Wm[bi, np.asarray(idx, np.int64)] = np.asarray(w, np.float32)
Wm = Wm / Wm.sum(1, keepdims=True)

out_dir = F / "mean19"
out_dir.mkdir(exist_ok=True)


def load_shard(path):
    z = np.load(path)
    chans = []
    for v in IN_VARS:
        key = next((c for c in KEY_ALIAS.get(v, [v]) if c in z.files), None)
        if key is None:
            raise KeyError(f"{v} not in {path}")
        chans.append(z[key][:, 0].astype(np.float32))
    return np.stack(chans, axis=1)                      # (days, 19, 180, 360)


def extract():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    files = sorted(
        fp for split in ("train", "val", "test") for fp in (D / split).glob("*_0.npz")
        if not fp.stem.startswith("climatology")
    )
    mine = files[rank::world]
    for fp in mine:
        arr = load_shard(fp)
        days = arr.shape[0]
        pooled = arr.reshape(days, 19, 90, 2, 180, 2).mean(axis=(3, 5))
        flat = pooled.reshape(days, 19, 90 * 180)
        bm = np.einsum("dcp,bp->dbc", flat, Wm).astype(np.float32)  # (days, nb, 19)
        np.savez_compressed(out_dir / f"{fp.stem}_m19.npz", m19=bm)
        print(f"[rank {rank}] {fp.stem}: {bm.shape}", flush=True)
    print(f"[rank {rank}] M19_EXTRACT_DONE", flush=True)


def merge():
    import json
    meta = json.load(open(F / "dataset/meta.json"))
    years = meta["years"]
    blocks = []
    for y in years:
        fp = out_dir / f"{y}_0_m19.npz"
        if not fp.exists():
            raise FileNotFoundError(fp)
        blocks.append(np.load(fp)["m19"][:365])
    big = np.concatenate(blocks, axis=0)
    assert big.shape == (len(years) * 365, nb, 19), big.shape
    np.save(F / "dataset/forcing19.npy", big)
    print(f"M19_MERGE_OK {big.shape} -> dataset/forcing19.npy", flush=True)
    for c, v in enumerate(IN_VARS):
        col = big[:, :, c]
        print(f"  {v:32s} mean={np.nanmean(col):.4g} std={np.nanstd(col):.4g}",
              flush=True)


if __name__ == "__main__":
    {"extract": extract, "merge": merge}[sys.argv[1]]()
