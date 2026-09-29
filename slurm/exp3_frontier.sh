#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J exp3lux
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/exp3_%j.out
# Luxembourg sub-hourly (exp3) on one GCD.
# usage: sbatch slurm/exp3_frontier.sh prep    # baseline anchors (no latents) + wait for the
#                                              # ERA5 store + ORBIT-2 latent extraction
#        sbatch slurm/exp3_frontier.sh train NAME K BIG [STEPS] [LAT_PAST] [TOPFT=off|frozen|ft]
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True
MODE=${1:?prep|train}
shift
run1(){ srun -N1 -n1 --gpus=1 $PY -u "$@"; }
if [ "$MODE" = prep ]; then
  # baseline anchors: the time-series model alone (zero latents, no training)
  EXP3_NOLAT=1 srun -N1 -n1 --gpus=1 $PY -u $F/scripts/exp3_train.py base_k0 0 0 || { echo PREP_FAIL_base_k0; exit 1; }
  EXP3_NOLAT=1 srun -N1 -n1 --gpus=1 $PY -u $F/scripts/exp3_train.py base_k96 96 0 || { echo PREP_FAIL_base_k96; exit 1; }
  echo BASE_ANCHORS_DONE
  for i in $(seq 1 100); do
    grep -q EU_FETCH_DONE $F/exp3/fetch.log 2>/dev/null && [ -s $F/exp3/eu_store.npy ] && break
    sleep 60
  done
  grep -q EU_FETCH_DONE $F/exp3/fetch.log || { echo STORE_TIMEOUT; exit 1; }
  run1 $F/scripts/extract_lux.py || { echo PREP_FAIL_extract; exit 1; }
  [ -s $F/exp3/lux_latents.npz ] || { echo PREP_FAIL_nolatents; exit 1; }
  echo PREP_DONE
else
  run1 $F/scripts/exp3_train.py "$@"
  echo SBATCH_EXIT_$?
fi
