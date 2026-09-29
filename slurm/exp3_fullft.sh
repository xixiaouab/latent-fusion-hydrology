#!/bin/bash -l
#SBATCH -A YOUR_PROJECT
#SBATCH -J exp3ft
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --export=NONE
#SBATCH -o logs/exp3ft_%j.out
# Luxembourg exp3 with ORBIT-2 in the loop: multi-node DDP, one task per GCD.
# usage: sbatch [-N nodes] exp3_fullft.sh NAME K MODE(ft|frozen) [STEPS] [LAT_PAST] [B_LOCAL] [BUDGET_MIN]
# Clean Frontier environment (works when submitted from a data-transfer node too):
# login shell + --export=NONE,
# then the proven multi-node RCCL recipe (S0 round 6). No Slurm GPU binding: with
# --gpu-bind=closest ranks 6 and 7 of each node shared one GCD (RCCL "Duplicate GPU");
# every task sees 8 GCDs and exp3_fullft.py selects cuda:SLURM_LOCALID.
unset SLURM_EXPORT_ENV
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
module reset > /dev/null 2>&1
module load PrgEnv-gnu/8.7.0 cpe/26.03 rocm/7.1.1 rccl-net-plugin craype-accel-amd-gfx90a
module -t list 2>&1 | tr '\n' ' '; echo
export PYTHONPATH=$F/pylibs
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
export MASTER_ADDR
export MASTER_PORT=$((29900 + SLURM_JOB_ID % 90))
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
NAME=${1:?NAME}
NN=$SLURM_JOB_NUM_NODES
echo "nodes=$NN master=$MASTER_ADDR:$MASTER_PORT"
srun -N$NN -n$((NN * 8)) --ntasks-per-node=8 -c7 --gpus-per-node=8 \
  $PY -u $F/scripts/exp3_fullft.py "$@" > $F/runs/${NAME}_console.log 2>&1
echo SBATCH_EXIT_$?
