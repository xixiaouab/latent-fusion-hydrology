#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J big19
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/big_%j.out

# Advisor's cells: 19-var protocol, 100M adapter, optional backbone FT.
# args: NAME MASK_DAYS LAT_FUT BIG(0/1) UNFREEZE(0/1) BATCH GA STEPS
NAME=${1:?}; K=${2:?}; LF=${3:?}; BIG=${4:?}; UF=${5:?}; B=${6:?}; GA=${7:?}; ST=${8:?}; FORC=${9:-forcing19.npy}; TOPFT=${10:-off}
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((30000 + K * 7 + LF * 3 + BIG * 11 + UF * 13))
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True

X=""
[ "$BIG" = "1" ] && X="$X --d-hidden 1024 --n-heads 16 --n-blocks 4 --latent-layers 4"
[ "$UF" = "1" ] && X="$X --unfreeze-ts --ts-lr-scale 0.05"
[ "$TOPFT" != "off" ] && X="$X --orbit-topft $TOPFT"

srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps $ST --batch $B --grad-accum $GA \
  --eval-every 200 --ddp --forcing-npy $FORC \
  --mask-days $K --lat-fut $LF $X \
  --per-day-queries --median-weight 3 \
  --out $F/runs/$NAME > $F/runs/${NAME}_console.log 2>&1
echo SBATCH_EXIT_$?
