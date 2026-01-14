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
  run_name   Base run name (default: config filename without extension)
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
CONFIG_BASENAME="$(basename "$CONFIG")"
CONFIG_STEM="${CONFIG_BASENAME%.*}"
RUN_NAME="${2:-$CONFIG_STEM}"
OUTPUT_DIR="${3:-runs}"

"$PYTHON" scripts/train.py --config "$CONFIG" --run-name "$RUN_NAME" --output-dir "$OUTPUT_DIR"

RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
if [[ ! -d "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -dt "$OUTPUT_DIR"/"$RUN_NAME"_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "Run directory not found for $RUN_NAME in $OUTPUT_DIR." >&2
  exit 1
fi

CHECKPOINT="$RUN_DIR/checkpoints/checkpoint_latest.pt"
"$PYTHON" scripts/eval.py --checkpoint "$CHECKPOINT" --config "$CONFIG"
