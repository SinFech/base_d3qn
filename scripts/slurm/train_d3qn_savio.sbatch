#!/bin/bash
# Submit from the repository root so relative log paths resolve into ./logs/.
#
# Example GPU smoke run:
# RUN_NAME=d3qn_sig_gpu_smoke \
# NUM_EPISODES=2 \
# TOTAL_STEPS=500 \
# EVAL_INTERVAL=0 \
# sbatch scripts/slurm/train_d3qn_savio.sbatch

#SBATCH -A ic_fintech296mb
#SBATCH -p savio2_1080ti
#SBATCH -q savio_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J d3qn_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/d3qn_signature_capital_6act_per06_n3_worstfold_gpu.yaml}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
SIGNATURE_DEVICE="${SIGNATURE_DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-d3qn_savio_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
NUM_EPISODES="${NUM_EPISODES:-}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

module purge
module load python
module load gcc
module load cuda
hash -r

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Expected Python executable at $PYTHON_BIN" >&2
    echo "Create the environment first or override PYTHON_BIN." >&2
    exit 1
fi

GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH:-}"

echo "date=$(date)"
echo "node=$(hostname)"
echo "repo_root=$REPO_ROOT"
echo "python_bin=$PYTHON_BIN"
echo "config_path=$CONFIG_PATH"
echo "train_device=$TRAIN_DEVICE"
echo "signature_device=$SIGNATURE_DEVICE"
echo "run_name=$RUN_NAME"
echo "output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" -u -c "import torch, pysiglib; print('torch_cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'); print('pysiglib_cuda', getattr(pysiglib, 'BUILT_WITH_CUDA', None))"

cmd=(
    "$PYTHON_BIN"
    -u
    scripts/train.py
    --config "$CONFIG_PATH"
    --device "$TRAIN_DEVICE"
    --run-name "$RUN_NAME"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$NUM_EPISODES" ]]; then
    cmd+=(--num-episodes "$NUM_EPISODES")
fi
if [[ -n "$TOTAL_STEPS" ]]; then
    cmd+=(--total-steps "$TOTAL_STEPS")
fi
if [[ -n "$LOG_INTERVAL" ]]; then
    cmd+=(--log-interval "$LOG_INTERVAL")
fi
if [[ -n "$EVAL_INTERVAL" ]]; then
    cmd+=(--override "train.eval_interval=$EVAL_INTERVAL")
fi
if [[ -n "$SIGNATURE_DEVICE" ]]; then
    cmd+=(--override "env.obs.signature.torch.device=$SIGNATURE_DEVICE")
fi

IFS=';' read -r -a extra_override_items <<< "$EXTRA_OVERRIDES"
for override in "${extra_override_items[@]}"; do
    if [[ -n "$override" ]]; then
        cmd+=(--override "$override")
    fi
done

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
