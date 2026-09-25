#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J scar
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/scar_%j.out

# Scarcity-curve cell: mask the LAST $1 of 10 future-forcing days.
# Submit the whole matrix (6 nodes in parallel, one K per node):
#   for K in 0 2 4 6 8 10; do sbatch scripts/scarcity_frontier.sh $K; done
K=${1:?usage: sbatch scarcity_frontier.sh <mask_days 0-10>}
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + K))
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST

srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps 2000 --batch 16 --eval-every 200 --ddp \
  --mask-days $K --per-day-queries --median-weight 3 \
  --out $F/runs/scar_k$K > $F/runs/scar_k${K}_console.log 2>&1
echo SBATCH_EXIT_$?
