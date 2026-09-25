#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/tbig_%j.out
# Water-temperature cells, champion recipe + strict no-future protocol.
# usage: sbatch temp_big.sh <NAME> <K> <BIG 0|1> <UF 0|1> [TOPFT off|frozen|ft]
NAME=${1:?} ; K=${2:?} ; BIG=${3:-1} ; UF=${4:-0} ; TOPFT=${5:-off}
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29800 + SLURM_JOB_ID % 100))
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
EXTRA=""
[ "$BIG" = 1 ] && EXTRA="$EXTRA --d-hidden 1024 --n-heads 16 --n-blocks 4 --latent-layers 4"
[ "$TOPFT" != off ] && EXTRA="$EXTRA --orbit-topft $TOPFT"
BATCH=16; GA=1; STEPS=2000
if [ "$UF" = 1 ]; then
  EXTRA="$EXTRA --unfreeze-ts --ts-lr-scale 0.05 --forcing-npy forcing15.npy"
  BATCH=2; GA=8; STEPS=1400
fi
srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps $STEPS --batch $BATCH --grad-accum $GA --eval-every 200 --ddp \
  --target-npy temperature.npy --flow-as-covariate \
  --mask-days $K --lat-fut 0 --per-day-queries --median-weight 3 \
  $EXTRA \
  --out $F/runs/$NAME > $F/runs/${NAME}_console.log 2>&1
echo SBATCH_EXIT_$?
