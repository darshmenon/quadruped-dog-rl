# Demos

## Indoor Autonomy

Start the flagship indoor autonomy stack:

```bash
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
ros2 launch launch/indoor_autonomy_go2.launch.py
```

Headless:

```bash
ros2 launch launch/indoor_autonomy_go2.launch.py headless:=true rviz:=false
```

This launches the CHAMP-backed Go2 in `training/envs/go2_gz_world_room.sdf` with RTAB-Map 3D SLAM, frontier exploration, obstacle tracking, and RViz.

With Foxglove streaming, if `foxglove_bridge` is installed:

```bash
ros2 launch launch/indoor_autonomy_go2.launch.py foxglove:=true
```

## Check The Stack

Run this after the launch has been up for 30-60 seconds:

```bash
python3 scripts/check_stack_go2.py
```

For runs without obstacle tracking:

```bash
python3 scripts/check_stack_go2.py --skip-obstacles
```

Expected healthy signals:

- `/odom` publishing
- `/points` publishing
- `/map` publishing after RTAB-Map starts
- `/obstacle_tracker/state` publishing when `track_obstacles:=true`
- `odom->base` and `map->base` TF available

## Record A Session

```bash
scripts/record_go2_session.sh
```

Or choose an output path:

```bash
scripts/record_go2_session.sh bags/indoor_autonomy_test
```

Replay later:

```bash
ros2 bag play bags/indoor_autonomy_test
```

## Stairs & Ledges

Usable worlds (world name is `go2_rl`, so CHAMP / stand / odom bridges work):

| Course | File | What you get |
|--------|------|--------------|
| `stairs` | `training/envs/go2_gz_world_stairs.sdf` | Easy 6 cm → mid 8 cm → hard 12 cm flights + landing |
| `ledges` | `training/envs/go2_gz_world_ledges.sdf` | Platforms (parkour climbs), gap crossings, hollow open-riser stairs, curb |
| MuJoCo | `training/envs/go2_stairs_scene.xml` | Same course for MuJoCo play / RL |

One-command CHAMP walk (teleop `/cmd_vel` after ~15 s):

```bash
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
ros2 launch launch/stairs_ledges_go2.launch.py course:=stairs
# or
ros2 launch launch/stairs_ledges_go2.launch.py course:=ledges
```

Sim-only (no CHAMP gait — for Gazebo RL / stand):

```bash
ros2 launch training/launch/gazebo_rl.launch.py \
  world:="$(pwd)/training/envs/go2_gz_world_stairs.sdf"
```

Regenerate SDF/MJCF (optional):

```bash
python3 scripts/generate_stairs_ledges.py
```

### MuJoCo RL

```bash
# Download stairs .pt model files
python3 scripts/download_pretrained.py --only stairs

# Train the flat walk SB3 zip on the stairs course (blind = same 76-dim obs)
./scripts/train_stairs.sh --blind --init-from-flat --n_envs 4 --timesteps 300000 --device cpu

# Sighted from scratch (94-dim height scan)
./scripts/train_stairs.sh --n_envs 4 --timesteps 300000 --device cpu

python3 training/play_policy.py \
  --model training/logs/stairs/best_model.zip --scene stairs --blind
```

See [docs/PRETRAINED.md](PRETRAINED.md) for model names and usage.

Quad-SDK also has step/gap/parkour SDFs (`step_20cm.sdf`, `gap_40cm.sdf`, `parkour_local_min.sdf`) via `./scripts/walk_quadsdk_go2.sh …`. Papers: [docs/papers/README.md](papers/README.md). Models: [docs/PRETRAINED.md](PRETRAINED.md).

## Model Policies

```bash
# SB3 walk
python3 training/play_policy.py --model training/logs/mujoco/best_model.zip --cmd 0.5 0 0

# Download Go2 .pt model files into training/pretrained/
python3 scripts/download_pretrained.py
```

Full table and usage notes: [PRETRAINED.md](PRETRAINED.md).

## Gazebo RL Evaluation

Evaluate a trained Gazebo PPO checkpoint:

```bash
python3 training/eval_gazebo.py \
  --model training/logs/gazebo/go2_gazebo_final.zip \
  --episodes 5 \
  --csv training/eval/gazebo_eval.csv
```

Evaluate against an already running Gazebo RL sim:

```bash
python3 training/eval_gazebo.py --no-launch --episodes 5
```

Current metrics include reward, episode length, fall/truncation status, mean planar speed, command speed error, and velocity-integrated distance. The environment does not yet expose world-frame base position directly, so distance is integrated from `/odom` velocity.
