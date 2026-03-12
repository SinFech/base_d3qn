#!/bin/bash
# Walk-forward D3QN on Savio (3 folds x multi-seed)
#
# Example:
# RUN_TAG=wf_sellfrac_$(date +%Y%m%d_%H%M%S) \
# D3QN_CONFIG=configs/d3qn_signature_capital_6act_per06_n3_sellfrac_gpu.yaml \
# SEEDS=42,43,44,45,46 \
# TRAIN_EPISODES=260 \
# MAX_TOTAL_STEPS=100000 \
# sbatch --time=12:00:00 scripts/slurm/wf_d3qn_savio.sbatch

#SBATCH -A ic_fintech296mb
#SBATCH -p savio2_1080ti
#SBATCH -q savio_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J d3qn_wf
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
D3QN_CONFIG="${D3QN_CONFIG:-configs/d3qn_signature_capital_6act_per06_n3_sellfrac_gpu.yaml}"
FOLDS_JSON="${FOLDS_JSON:-configs/folds_rolling_long_oos.json}"
SEEDS="${SEEDS:-42,43,44,45,46}"
RUN_TAG="${RUN_TAG:-wf_d3qn_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/wf_d3qn_sellfrac}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TRAIN_SPLIT="${TRAIN_SPLIT:-1.0}"
TRAIN_EPISODES="${TRAIN_EPISODES:-260}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"

module purge
module load python
module load gcc
module load cuda
hash -r

cd "$REPO_ROOT"
mkdir -p logs "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Expected Python executable at $PYTHON_BIN" >&2
    exit 1
fi

GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR

echo "date=$(date)"
echo "node=$(hostname)"
echo "repo_root=$REPO_ROOT"
echo "python_bin=$PYTHON_BIN"
echo "d3qn_config=$D3QN_CONFIG"
echo "folds_json=$FOLDS_JSON"
echo "seeds=$SEEDS"
echo "run_tag=$RUN_TAG"
echo "output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" -u -c "import torch, pysiglib; print('torch_cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'); print('pysiglib_cuda', getattr(pysiglib, 'BUILT_WITH_CUDA', None))"

cmd=(
    "$PYTHON_BIN"
    -u
    scripts/walk_forward_protocol.py
    --algos d3qn
    --d3qn-config "$D3QN_CONFIG"
    --folds-json "$FOLDS_JSON"
    --seeds "$SEEDS"
    --device "$TRAIN_DEVICE"
    --train-split "$TRAIN_SPLIT"
    --train-episodes "$TRAIN_EPISODES"
    --max-total-steps "$MAX_TOTAL_STEPS"
    --eval-episodes "$EVAL_EPISODES"
    --output-dir "$OUTPUT_DIR"
    --run-tag "$RUN_TAG"
)

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
