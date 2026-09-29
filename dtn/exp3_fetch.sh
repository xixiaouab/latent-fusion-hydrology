#!/bin/bash
# Luxembourg (exp3) ERA5 window: 6 parallel readers, then merge -> $E/eu_store.npy.
# Needs outbound internet (a data-transfer node). ~40 min; resumable per variable.
# usage: nohup bash dtn/exp3_fetch.sh > $FUSION_ROOT/exp3/fetch.log 2>&1 &
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
E=$F/exp3
PY=${PYTHON:-python3}          # needs numpy, xarray, zarr, gcsfs
mkdir -p $E
export EXP3_DIR=$E
date
pids=()
for g in temperature u_component_of_wind v_component_of_wind specific_humidity sl_a sl_b; do
  nice -n 10 $PY $F/scripts/era5_lux_fetch.py $g > $E/fetch_$g.log 2>&1 &
  pids+=($!)
  echo "launched $g pid $!"
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
date
echo "WORKERS_DONE fail=$fail"
nice -n 10 $PY $F/scripts/era5_lux_fetch.py merge
date
echo FETCH_V2_DONE
