#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -J exp3bdl
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -o logs/exp3bdl_%j.out
# Six single-GCD exp3 cells run CONCURRENTLY on one node (one GCD each).
# 3000 steps each = the configuration behind the reported numbers
# (at 800 steps the fusion gain is still ~0).
F=${FUSION_ROOT:?export FUSION_ROOT=/path/to/workdir}
PY=${PYTHON:-python3}
export PYTHONPATH=$F/pylibs
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=FAST
export PYTORCH_ALLOC_CONF=expandable_segments:True
CELLS=(
  "lux_k96_big 96 1 3000"
  "lux_k0_big 0 1 3000"
  "lux_k96_small 96 0 3000"
  "lux_k0_small 0 0 3000"
  "lux_k96_big_topfrozen 96 1 3000 30 frozen"
  "lux_k96_big_topft 96 1 3000 30 ft"
)
for c in "${CELLS[@]}"; do
  name=${c%% *}
  srun -N1 -n1 -c7 --gpus=1 --gpu-bind=closest --mem=64G --exact \
    $PY -u $F/scripts/exp3_train.py $c > $F/runs/${name}_console.log 2>&1 &
  echo "launched $name"
  sleep 2
done
wait
grep -h EXP3TRAIN_DONE $F/runs/lux_*_console.log
echo BUNDLE_DONE
