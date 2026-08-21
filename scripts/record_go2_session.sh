#!/usr/bin/env bash
set -euo pipefail

stamp="$(date +%Y%m%d_%H%M%S)"
out="${1:-bags/go2_${stamp}}"

mkdir -p "$(dirname "$out")"

exec ros2 bag record -o "$out" \
  /tf \
  /tf_static \
  /clock \
  /odom \
  /joint_states \
  /imu/data \
  /points \
  /map \
  /cmd_vel \
  /obstacle_tracker/state \
  /obstacle_tracker/markers
