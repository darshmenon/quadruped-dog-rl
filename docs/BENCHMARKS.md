# Benchmarks

Use this file to track repeatable results instead of relying on one-off RViz observations.

## Gazebo RL

Command:

```bash
python3 training/eval_gazebo.py \
  --model training/logs/gazebo/go2_gazebo_final.zip \
  --episodes 5 \
  --csv training/eval/gazebo_eval.csv
```

Metrics to report:

| Date | Model | Command | Episodes | Mean steps | Fall rate | Mean speed | Speed error | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | `0.5 0.0 0.0` | TBD | TBD | TBD | TBD | TBD | Train a fresh checkpoint after the Gazebo reward fixes. |

## Indoor Autonomy

Command:

```bash
ros2 launch launch/indoor_autonomy_go2.launch.py headless:=true rviz:=false
```

Metrics to report:

| Date | Locomotion | World | Map available | Exploration complete | Obstacle tracks | Notes |
|---|---|---|---|---|---|---|
| TBD | CHAMP | `go2_gz_world_room.sdf` | TBD | TBD | TBD | Verify after obstacle tracker launch wiring. |

## Locomotion Backends

| Backend | Flat world | Room world | Outdoor world | Rough world | Notes |
|---|---|---|---|---|---|
| CHAMP | Working baseline | Needs indoor verification | Needs tracker verification | TBD | Native `/cmd_vel` backend. |
| Quad-SDK NMPC | Working baseline | TBD | Working baseline | TBD | Goal-driven backend. |
| MuJoCo RL | Working with stability issue | N/A | N/A | Needs longer sighted/blind run | Fastest training loop. |
| Gazebo RL | Needs fresh training | TBD | TBD | TBD | Real ROS/Gazebo loop, slower but closer to deployment. |
