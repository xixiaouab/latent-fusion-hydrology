#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J fullft
#SBATCH -N 16
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/fullft_%j.out

# Full ORBIT-2 in-loop fine-tuning: frozen TS-model, 126M ORBIT-2 + 100M
# adapter trainable. 16 nodes x 8 GCD, batch 1/GCD = effective 128, 320 steps.
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}

# multi-node RCCL recipe (proven by S0 round 6)
module reset
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load rccl-net-plugin
module load craype-accel-amd-gfx90a

export PYTHONPATH=$F/pylibs:$F/scripts
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
export MASTER_ADDR
export MASTER_PORT=29717
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REALVR_HANG_DUMP=180

srun -N16 -n128 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
  $PY -u $F/scripts/fusion_train.py --steps 400 --batch 1 --grad-accum 1 \
  --eval-every 80 --ddp --forcing-npy forcing19.npy \
  --mask-days 10 --lat-past 10 --lat-fut 0 --orbit-fullft \
  --d-hidden 1024 --n-heads 16 --n-blocks 4 --latent-layers 4 \
  --per-day-queries --median-weight 3 \
  --out $F/runs/c5_fullft > $F/runs/c5_fullft_console.log 2>&1
echo SBATCH_EXIT_$?
