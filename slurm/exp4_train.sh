#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -o logs/exp4t_%j.out
# Regulated-basin trained cells: 4 configs sequentially on one GCD.
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
run1(){ srun -N1 -n1 --gpus=1 $PY -u $F/scripts/exp4_train.py "$@" ; }
run1 reg_noout_k10 10 0 600
run1 reg_out_k10   10 1 600
run1 reg_noout_k0   0 0 600
run1 reg_out_k0     0 1 600
echo EXP4T_ALL_DONE
