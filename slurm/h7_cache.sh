#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J h7cache
#SBATCH -N 6
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -o logs/h7_%j.out
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
export MASTER_ADDR
export MASTER_PORT=29631
D=${ERA5_DAYMET_DIR:?export ERA5_DAYMET_DIR=/path/to/era5-daymet}
mkdir -p $F/h7_cache
for SPLIT in train val test; do
  srun -N6 -n48 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
    $PY -u $F/scripts/extract_h7.py --data-dir $D --split $SPLIT \
    --masks $F/masks/basin_patch_masks.npz --out-dir $F/h7_cache --batch-days 4
done
echo CACHE_EXIT_$?
srun -N1 -n1 $PY -u $F/scripts/transpose_h7.py
echo SBATCH_EXIT_$?
