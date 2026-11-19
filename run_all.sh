#!/usr/bin/env bash
# Run the full experiment matrix:
#   5 seeds x {single, vec} x 2,000,000 steps
#
# Usage:
#   ./run_all.sh
#   SEEDS="0 1 2 3 4" MODES="single vec" TOTAL_STEPS=2000000 JOBS=8 ./run_all.sh
#
# If the conda env is not active:
#   PYTHON=/home/lhc/miniconda3/envs/DSC6007/bin/python ./run_all.sh
#
# Outputs:
#   results/train_seed{seed}.npz
#   results/train_seed{seed}_vec.npz
#   results/logs/{mode}_seed{seed}.log
#   checkpoints/ppo_{mode}_env/seed{seed}/ppo_clip_<step>.pth

set -euo pipefail

cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2 3 4}"
MODES="${MODES:-single vec}"
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
JOBS="${JOBS:-8}"

if [[ -n "${PYTHON:-}" ]]; then
    :
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="python"
fi
export PYTHON
export TOTAL_STEPS

# One process per core: avoid oversubscription across the parallel runs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/logs checkpoints

echo "python=${PYTHON}"
echo "seeds=${SEEDS}"
echo "modes=${MODES}"
echo "total_steps=${TOTAL_STEPS}"
echo "parallel_jobs=${JOBS}"

for seed in $SEEDS; do
    for mode in $MODES; do
        printf '%s %s\n' "$seed" "$mode"
    done
done | xargs -P "$JOBS" -n 2 bash -c '
    seed="$1"
    mode="$2"
    log="results/logs/${mode}_seed${seed}.log"
    echo "[start] seed=${seed} mode=${mode}"
    "$PYTHON" train.py \
        --mode "$mode" \
        --seed "$seed" \
        --total-steps "$TOTAL_STEPS" \
        > "$log" 2>&1
    echo "[done]  seed=${seed} mode=${mode} log=${log}"
' _

echo "all runs finished"
