#!/bin/bash
# Stage the Luxembourg (exp3) inputs. Needs outbound internet (a data-transfer node).
#   - ORBIT-2 global fine-tuned checkpoint + its ERA5 0.25 deg normalization stats
#     and 0.25 deg static layers (Hugging Face, jychoi-hpc/ORBIT-2)
#   - Python deps for the public ARCO-ERA5 zarr (xarray, zarr, gcsfs)
# The 15-min basin file (LUX_40.csv) ships with the time-series model's examples;
# the training scripts read it from $TS_MODEL_DIR/Example/data/exp3_subhourly_lux/
# unless LUX_CSV points elsewhere.
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
E=$F/exp3
PY=${PYTHON:-python3}
mkdir -p $E
HF=https://huggingface.co/jychoi-hpc/ORBIT-2/resolve/main
declare -A U=(
  [global_126m_precipitation.ckpt]=global-finetune/global_126m_precipitation.ckpt
  [global_126m.yaml]=global-finetune/global_126m_precipitation.yaml
  [normalize_mean.npz]=mean_std/era5/0.25_deg/normalize_mean.npz
  [normalize_std.npz]=mean_std/era5/0.25_deg/normalize_std.npz
  [land_sea_mask_0.25deg.npy]=static_variables/land_sea_mask_0.25deg.npy
  [landcover_0.25deg.npy]=static_variables/landcover_0.25deg.npy
  [lattitude_0.25deg.npy]=static_variables/lattitude_0.25deg.npy
  [orography_0.25deg.npy]=static_variables/orography_0.25deg.npy
)
# The large checkpoint is served through a CDN whose DNS was intermittent on our
# transfer nodes; retry each file until it has a plausible size.
for f in "${!U[@]}"; do
  for t in $(seq 1 40); do
    [ -s $E/$f ] && sz=$(stat -c%s $E/$f) || sz=0
    { [ "$f" = global_126m_precipitation.ckpt ] && [ $sz -gt 100000000 ]; } && break
    { [ "$f" != global_126m_precipitation.ckpt ] && [ $sz -gt 300 ]; } && break
    curl -fsSL --connect-timeout 15 $HF/${U[$f]} -o $E/$f 2>/dev/null && continue
    sleep 5
  done
  echo "$f -> $([ -s $E/$f ] && stat -c%s $E/$f || echo MISSING)"
done
$PY -m pip install xarray zarr gcsfs > $E/pip_stage.log 2>&1 && echo DEPS_OK
echo STAGE_DONE
