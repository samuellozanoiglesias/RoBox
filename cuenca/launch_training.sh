#!/usr/bin/env bash
set -euo pipefail

# Cluster launcher for full RoBox integration.
# Defaults target /data/samuel_lozano/RoBox but can be overridden.

usage() {
  cat <<EOF
Usage: $(basename "$0") [--root PATH] [--config PATH] [--respect-config] [--python CMD] [--log-dir PATH]

Options:
  --root PATH            RoBox repository root (default: /data/samuel_lozano/RoBox)
  --config PATH          Config path relative to root or absolute (default: config.yaml)
  --respect-config       Do not force training.run_training=true
  --python CMD           Python executable (default: python3)
  --log-dir PATH         Directory for launcher logs (default: <root>/logs)
  -h, --help             Show this help

Environment overrides:
  ROBOX_ROOT, ROBOX_CONFIG, ROBOX_PYTHON, ROBOX_LOG_DIR
EOF
}

ROOT="${ROBOX_ROOT:-/data/samuel_lozano/RoBox}"
CONFIG="${ROBOX_CONFIG:-config.yaml}"
PYTHON_CMD="${ROBOX_PYTHON:-python3}"
LOG_DIR="${ROBOX_LOG_DIR:-}"
RESPECT_CONFIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --respect-config)
      RESPECT_CONFIG=1
      shift
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "Error: root directory does not exist: $ROOT" >&2
  exit 1
fi

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Error: python command not found: $PYTHON_CMD" >&2
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$ROOT/logs"
fi
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/launch_training_${STAMP}.log"

cd "$ROOT"

echo "[launch_training] root=$ROOT" | tee -a "$LOG_FILE"
echo "[launch_training] config=$CONFIG" | tee -a "$LOG_FILE"
echo "[launch_training] python=$PYTHON_CMD" | tee -a "$LOG_FILE"
echo "[launch_training] log=$LOG_FILE" | tee -a "$LOG_FILE"

CMD=("$PYTHON_CMD" "run_training.py" "--config" "$CONFIG")
if [[ "$RESPECT_CONFIG" -eq 1 ]]; then
  CMD+=("--respect-config")
fi

echo "[launch_training] command=${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[launch_training] completed successfully" | tee -a "$LOG_FILE"
