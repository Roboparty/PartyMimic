#!/bin/bash
# Train PartyMimic (the defaults of partymimic.train are the recipe).
# Usage: bash scripts/train.sh <log_dir> --robot g1mj --motion_dir <npz clips> [more train.py flags, e.g. --seed 1]
# About 31 min on one RTX 4090 for the default 12000 iterations. Python: $PYTHON, default "uv run python".
set -eu
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: bash scripts/train.sh <log_dir> [train.py flags]}; shift
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # the reference runs used 4 CPU threads
PYTHONPATH=src ${PYTHON:-uv run python} -u -m partymimic.train --log_dir "$LOG_DIR" "$@" 2>&1 | tee "$LOG_DIR/stdout.log"
