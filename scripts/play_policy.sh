#!/bin/bash
# Play/visualize a trained quadruped walking policy
#
# Usage:
#   ./scripts/play_policy.sh go2
#   ./scripts/play_policy.sh go1

set -e

TASK=${1:-go2}
PYTHON_BIN=${PYTHON:-python3}

echo "Playing task: $TASK"

cd training
"$PYTHON_BIN" legged_gym/scripts/play.py --task="$TASK"
