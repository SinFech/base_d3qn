#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run.sh [config] [run_name] [output_dir]

Runs training and then evaluation.

Arguments:
  config     Path to config YAML (default: configs/baseline.yaml)
  run_name   Base run name (default: run_YYYYmmdd_HHMMSS)
  output_dir Output directory (default: runs)

For full training options:
  scripts/train.py -h
EOF
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON. Create it with: uv venv .venv && uv sync" >&2
  exit 1
fi

CONFIG="${1:-configs/baseline.yaml}"
RUN_NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${3:-runs}"

"$PYTHON" scripts/train.py --config "$CONFIG" --run-name "$RUN_NAME" --output-dir "$OUTPUT_DIR"

CHECKPOINT="$OUTPUT_DIR/$RUN_NAME/checkpoints/checkpoint_latest.pt"
"$PYTHON" scripts/eval.py --checkpoint "$CHECKPOINT" --config "$CONFIG"
