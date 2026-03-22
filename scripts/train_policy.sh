#!/bin/bash
# Train a quadruped walking policy using Unitree RL Gym (Isaac Gym backend)
#
# Usage:
#   ./scripts/train_policy.sh go2          # train Unitree Go2
#   ./scripts/train_policy.sh go1          # train Unitree Go1
#   ./scripts/train_policy.sh --headless   # no GUI (faster)
#
# Requirements:
#   - Isaac Gym installed (https://developer.nvidia.com/isaac-gym)
#   - pip install -e training/

set -e

TASK=${1:-go2}
HEADLESS=${2:-""}

echo "Training task: $TASK"

cd training
python legged_gym/scripts/train.py --task="$TASK" $HEADLESS
