"""Minimal meta.json (build's latent-merge phase skipped as redundant)."""
import os
import json
from pathlib import Path
import numpy as np
import shapefile

F = Path(os.environ.get("FUSION_ROOT", "."))
mz = np.load(F / "masks/basin_patch_masks.npz", allow_pickle=True)
basin_ids = [str(b) for b in mz["basin_ids"]]
sf = shapefile.Reader(str(F / "camels/HCDN_nhru_final_671.shp"))
fields = [f[0] for f in sf.fields[1:]]
fi_id, fi_area = fields.index("hru_id"), fields.index("AREA")
area = {str(r[fi_id]).zfill(8): float(r[fi_area]) for r in sf.iterRecords()}
YEARS = list(range(1980, 2022))
meta = {"basin_ids": basin_ids, "years": YEARS, "n_days": len(YEARS) * 365,
        "forcing_channels": ["prcp_mmday", "srad_wm2", "tmax_c", "tmin_c", "vp_pa"],
        "areas_m2": {g: area.get(g) for g in basin_ids},
        "splits": {"train_years": [1980, 2009], "val_years": [2010, 2012],
                   "test_years": [2013, 2014]}}
json.dump(meta, open(F / "dataset/meta.json", "w"))
print("META_OK", len(basin_ids))
