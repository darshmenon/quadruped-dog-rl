#!/bin/bash
# Train Go2 stair/ledge climbing policy (MuJoCo).
#
# Usage:
#   ./scripts/train_stairs.sh
#   ./scripts/train_stairs.sh --blind --init-from-flat --timesteps 300000 --n_envs 4 --device cpu
#   ./scripts/train_stairs.sh --blind

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN=${PYTHON:-python3}
echo "Backend: MuJoCo stairs"
exec "$PYTHON_BIN" "$ROOT/training/train_stairs.py" "$@"
