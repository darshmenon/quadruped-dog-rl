#!/bin/bash
# Play/visualize a trained quadruped walking policy
#
# Usage:
#   ./scripts/play_policy.sh go2
#   ./scripts/play_policy.sh go1

set -e

TASK=${1:-go2}

echo "Playing task: $TASK"

cd training
python legged_gym/scripts/play.py --task="$TASK"
