# Troubleshooting

## No `/map`

- Wait at least 20 seconds; `slam3d_go2.launch.py` starts RTAB-Map after Gazebo and TF are ready.
- Check `/points`:

```bash
ros2 topic hz /points
```

- Check TF:

```bash
python3 scripts/check_stack_go2.py --skip-obstacles
```

## Obstacle Tracker Publishes Nothing

- Launch with `track_obstacles:=true`, or use `launch/indoor_autonomy_go2.launch.py`.
- Confirm point cloud input:

```bash
ros2 topic hz /points
```

- Confirm map TF exists:

```bash
ros2 run tf2_ros tf2_echo map base
```

## Gazebo RL Cannot Find Topics

Use matching isolation values for the sim and evaluator/trainer:

```bash
python3 training/train_gazebo.py --ros-domain-id 177 --gz-partition go2rltrain
python3 training/eval_gazebo.py --ros-domain-id 177 --gz-partition go2rltrain --no-launch
```

Do not reuse the same `ROS_DOMAIN_ID` and `GZ_PARTITION` as a separate SLAM/Nav2 launch unless the sessions are intentionally sharing topics.

## RViz Shows No Robot Or Cloud

- Source both ROS and this workspace:

```bash
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
```

- Keep `ROS_DOMAIN_ID` consistent across terminals.
- For the default indoor demo, use `ROS_DOMAIN_ID=157` unless overridden.
