#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J refw10
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/refw10_%j.out
# Window-10 frozen cached reference: apples-to-apples for the fullft cell.
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29733
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps 2000 --batch 8 --grad-accum 2 \
  --eval-every 200 --ddp --forcing-npy forcing19.npy \
  --mask-days 10 --lat-past 10 --lat-fut 0 \
  --d-hidden 1024 --n-heads 16 --n-blocks 4 --latent-layers 4 \
  --per-day-queries --median-weight 3 \
  --out $F/runs/ref_w10 > $F/runs/ref_w10_console.log 2>&1
echo SBATCH_EXIT_$?
