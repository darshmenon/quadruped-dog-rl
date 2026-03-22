#!/bin/bash
# Spawn Unitree Go2 in Gazebo Garden and connect ROS2 bridge
#
# Usage: ./scripts/spawn_go2_gazebo.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
URDF="$REPO_DIR/urdf/go2_unitree/urdf/go2.urdf"
SDF="/tmp/go2.sdf"

source /opt/ros/humble/setup.bash

echo "Converting URDF to SDF..."
gz sdf -p "$URDF" > "$SDF"

echo "Starting Gazebo Garden..."
gz sim -r empty.sdf &
GZ_PID=$!
sleep 4

echo "Spawning Go2..."
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 5000 \
  --req "sdf_filename: '$SDF', name: 'go2', pose: {position: {z: 0.5}}"

echo "Starting ROS2 bridge and robot_state_publisher..."
ROBOT_DESC=$(cat "$URDF")
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$ROBOT_DESC" -p use_sim_time:=true &
ros2 run ros_gz_bridge parameter_bridge \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  /cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist &

echo "Starting RViz2..."
ros2 run rviz2 rviz2 &

echo ""
echo "Go2 running in Gazebo Garden."
echo "Send velocity commands:"
echo "  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}' --once"
echo ""
echo "Press Ctrl+C to stop."
wait $GZ_PID
