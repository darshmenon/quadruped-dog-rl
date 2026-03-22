#!/bin/bash
# Launch Gazebo simulation for a quadruped robot
#
# Usage:
#   ./scripts/launch_sim.sh go1
#   ./scripts/launch_sim.sh spot
#   ./scripts/launch_sim.sh mini_cheetah

set -e

source /opt/ros/humble/setup.bash

ROBOT=${1:-go1}

echo "Launching Gazebo sim for: $ROBOT"

ros2 launch launch/gazebo_sim.launch.py robot:="$ROBOT"
