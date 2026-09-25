#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J temp
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/temp_%j.out

# Stream-temperature task cell (exp2 convention: streamflow joins covariates).
# usage: sbatch temp_frontier.sh <mask_days 0-10>
K=${1:?usage: sbatch temp_frontier.sh <mask_days 0-10>}
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29700 + K))
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST

srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps 2000 --batch 16 --eval-every 200 --ddp \
  --target-npy temperature.npy --flow-as-covariate \
  --mask-days $K --per-day-queries --median-weight 3 \
  --out $F/runs/temp_k$K > $F/runs/temp_k${K}_console.log 2>&1
echo SBATCH_EXIT_$?
