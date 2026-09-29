#!/bin/bash
# CPU smoke test of every exp3 code path before spending GPU time
# (2 days / 2 steps / 2 origins, niced; a few minutes on 8 threads).
# Needs the staged inputs and $E/eu_store.npy; writes to $E/smoke and runs/smoke_*.
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
E=$F/exp3
S=$E/smoke
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MASTER_ADDR=localhost MASTER_PORT=29655
mkdir -p $S
flt(){ grep -v -i "warn\|deprecat\|^ *$\|autocast" ; }
echo "== extract"; date
LUX_DEVICE=cpu LUX_SMOKE_DAYS=2 LUX_OUT=$S nice -n 19 $PY -u $F/scripts/extract_lux.py 2>&1 | flt | tail -45
echo "== train default (small adapter)"; date
EXP3_DEVICE=cpu EXP3_IN=$S EXP3_SMOKE=1 nice -n 19 $PY -u $F/scripts/exp3_train.py smoke_train 96 0 2>&1 | flt | tail -14
echo "== train topft (big adapter + ORBIT-2 top block)"; date
EXP3_DEVICE=cpu EXP3_IN=$S EXP3_SMOKE=1 nice -n 19 $PY -u $F/scripts/exp3_train.py smoke_topft 96 1 2 30 ft 2>&1 | flt | tail -14
echo "== fullft (ORBIT-2 in loop, 1 rank gloo)"; date
EXP3_DEVICE=cpu EXP3_BACKEND=gloo EXP3_IN=$S EXP3_SMOKE=1 SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 \
  nice -n 19 $PY -u $F/scripts/exp3_fullft.py smoke_fullft 96 ft 2 2 1 30 2>&1 | flt | tail -24
date; echo SMOKE_ALL_DONE
