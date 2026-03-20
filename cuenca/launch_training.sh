#!/usr/bin/env bash
# Author: Samuel Lozano
set -euo pipefail

# Cluster launcher for full RoBox integration.
# Defaults target /home/samuel_lozano/RoBox but can be overridden.

usage() {
  cat <<EOF
Usage: $(basename "$0") [--root PATH] [--config PATH] [--respect-config] [--python CMD] [--log-dir PATH]

Options:
  --root PATH            RoBox repository root (default: /home/samuel_lozano/RoBox)
  --config PATH          Config path relative to root or absolute (default: config.yaml)
  --respect-config       Do not force training.run_training=true
  --python CMD           Python executable (default: python3)
  --log-dir PATH         Directory for launcher logs (default: <root>/logs)
  -h, --help             Show this help

Environment overrides:
  ROBOX_ROOT, ROBOX_CONFIG, ROBOX_PYTHON, ROBOX_LOG_DIR
EOF
}

ROOT="${ROBOX_ROOT:-/home/samuel_lozano/RoBox}"
CONFIG="${ROBOX_CONFIG:-config.yaml}"
PYTHON_CMD="${ROBOX_PYTHON:-python3}"
LOG_DIR="${ROBOX_LOG_DIR:-}"
RESPECT_CONFIG=0

# [DEBUG - launch_training.sh] Initial variable values
echo "[DEBUG - launch_training.sh] ROOT initial: $ROOT"
echo "[DEBUG - launch_training.sh] CONFIG initial: $CONFIG"
echo "[DEBUG - launch_training.sh] PYTHON_CMD initial: $PYTHON_CMD"
echo "[DEBUG - launch_training.sh] LOG_DIR initial: $LOG_DIR"
echo "[DEBUG - launch_training.sh] RESPECT_CONFIG initial: $RESPECT_CONFIG"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      echo "[DEBUG - launch_training.sh] --root set: $ROOT"
      ;;
    --config)
      CONFIG="$2"
      shift 2
      echo "[DEBUG - launch_training.sh] --config set: $CONFIG"
      ;;
    --respect-config)
      RESPECT_CONFIG=1
      shift
      echo "[DEBUG - launch_training.sh] --respect-config set: $RESPECT_CONFIG"
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      echo "[DEBUG - launch_training.sh] --python set: $PYTHON_CMD"
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      echo "[DEBUG - launch_training.sh] --log-dir set: $LOG_DIR"
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

echo "[DEBUG - launch_training.sh] ROOT directory exists: $ROOT"

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi

echo "[DEBUG - launch_training.sh] CONFIG absolute path: $CONFIG"

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

echo "[DEBUG - launch_training.sh] CONFIG file exists: $CONFIG"

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Error: python command not found: $PYTHON_CMD" >&2
  exit 1
fi

echo "[DEBUG - launch_training.sh] PYTHON_CMD found: $PYTHON_CMD"


# Set LOG_DIR for launcher logs
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$ROOT/logs"
fi
mkdir -p "$LOG_DIR"

# Set DATA_DIR for experiment outputs
DATA_DIR="/data/samuel_lozano/RoBox"
echo "[DEBUG - launch_training.sh] DATA_DIR set: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo "[DEBUG - launch_training.sh] LOG_DIR resolved: $LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/launch_training_${STAMP}.log"

# Export environment variable for downstream scripts
export ROBOX_DATA_DIR="$DATA_DIR"

echo "[DEBUG - launch_training.sh] STAMP: $STAMP"
echo "[DEBUG - launch_training.sh] LOG_FILE: $LOG_FILE"

cd "$ROOT"

echo "[DEBUG - launch_training.sh] Changed directory to ROOT: $ROOT"

echo "[launch_training] root=$ROOT" | tee -a "$LOG_FILE"
echo "[launch_training] config=$CONFIG" | tee -a "$LOG_FILE"
echo "[launch_training] python=$PYTHON_CMD" | tee -a "$LOG_FILE"
echo "[launch_training] log=$LOG_FILE" | tee -a "$LOG_FILE"

CMD=("$PYTHON_CMD" "run_training.py" "--config" "$CONFIG")
if [[ "$RESPECT_CONFIG" -eq 1 ]]; then
  CMD+=("--respect-config")
fi

# Inform downstream code to use DATA_DIR for pair_... outputs
echo "[DEBUG - launch_training.sh] Will use DATA_DIR for experiment outputs: $DATA_DIR" | tee -a "$LOG_FILE"

echo "[DEBUG - launch_training.sh] CMD array: ${CMD[*]}"

echo "[launch_training] command=${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[DEBUG - launch_training.sh] Python command executed. Exit code: $?"

echo "[launch_training] completed successfully" | tee -a "$LOG_FILE"
echo "[DEBUG - launch_training.sh] Script completed successfully"
