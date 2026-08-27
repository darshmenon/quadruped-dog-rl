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
| 2026-08-27 | CHAMP | `go2_gz_world_room.sdf` | Yes, 343 WM nodes over ~6.3min | No -- 20 frontier goals abandoned ("no progress... for 12s"), repeatedly re-targeting the same 2 unreached corners | Tracker started, 1 track seen | After the odom-transport and gait-damping fixes (see below): 0 TF-extrapolation warnings, 1 octomap queue-drop (startup-only, vs. constant before), idle drift down to ~0.1-0.15m x2 (was 2.78m). Map is real and growing but two corners never got reached -- likely a residual CHAMP path-following gap, not the TF/gait issue. |
| 2026-08-27 | Quad-SDK NMPC | `go2_room.sdf` (copy of the room world under quad_sim_scripts' worlds/, renamed to Quad-SDK's expected `default` world name, with a matching flat terrain mesh) | No -- WM stuck at 3 nodes | False positive: reported "exploration complete" after 30s of no frontiers, but only because the robot never left spawn (tiny map => no frontiers to find) | Not reached (robot never explored) | Spawns and loads cleanly now (model URIs + terrain mesh resolve, no resource errors) after wiring `ros2/quad_sdk_external/setup_env.sh`'s env vars into `slam3d_go2.launch.py`. But `local_planner_node` failed "NMPC solving fail, ApplicationReturnStatus = -4" continuously (260 times in ~90s) and the robot never moved. Needs its own investigation (solver config/IPOPT), separate from the world-wiring fix -- do not read the "exploration complete" line as success for this backend yet. |

## Locomotion Backends

| Backend | Flat world | Room world | Outdoor world | Rough world | Notes |
|---|---|---|---|---|---|
| CHAMP | Working baseline | Working (maps successfully; frontier nav struggles on 2 far corners) -- see Indoor Autonomy 2026-08-27 | Needs tracker verification | TBD | Native `/cmd_vel` backend. Not physics-tuned for Go2 (generic reference gait, see README) -- produced visible hopping on contact until the 2026-08-27 d_gain fix. |
| Quad-SDK NMPC | Working baseline | Spawns/loads cleanly (2026-08-27 world-wiring fix) but local planner fails to solve continuously in this world -- robot never walks. Needs solver-level debugging. | Working baseline | TBD | Goal-driven backend, verified end-to-end on flat/outdoor terrain per README; the room-world failure above is new information, not previously covered. |
| MuJoCo RL | Working with stability issue | N/A | N/A | Needs longer sighted/blind run | Fastest training loop. |
| Gazebo RL | Needs fresh training | TBD | TBD | TBD | Real ROS/Gazebo loop, slower but closer to deployment. |
