# Training terrain tools

Stairs, ledges, gaps, and hollow steps for Go2 sim training.

## Quick generate (this repo)

Regenerate Gazebo + MuJoCo stair/ledge worlds (no extra pip deps):

```bash
python3 scripts/generate_stairs_ledges.py
```

Outputs:

| File | Use |
|------|-----|
| `training/envs/go2_gz_world_stairs.sdf` | Gazebo: easy / mid / hard solid stairs + descent |
| `training/envs/go2_gz_world_ledges.sdf` | Gazebo: platforms, gaps, hollow stairs, curb |
| `training/envs/go2_stairs_scene.xml` | MuJoCo: full Go2 scene with the same course |

### Gazebo

One-command CHAMP walk on stairs or ledges:

```bash
ros2 launch launch/stairs_ledges_go2.launch.py course:=stairs
ros2 launch launch/stairs_ledges_go2.launch.py course:=ledges
```

Sim-only (RL / stand, no CHAMP):

```bash
ros2 launch training/launch/gazebo_rl.launch.py \
  world:="$(pwd)/training/envs/go2_gz_world_stairs.sdf"

ros2 launch training/launch/gazebo_rl.launch.py \
  world:="$(pwd)/training/envs/go2_gz_world_ledges.sdf"
```

World `<name>` is `go2_rl` (required by the existing Gazebo bridges).

Quad-SDK walk over stairs (existing step worlds also work):

```bash
./scripts/walk_quadsdk_go2.sh 4.0 0.0 gui step_20cm.sdf
```

### MuJoCo

```bash
./scripts/train_stairs.sh
python3 training/play_policy.py --model training/logs/stairs/best_model.zip --scene stairs
```

Env: `training/envs/go2_mujoco_stairs_env.py` on `go2_stairs_scene.xml`
(progress reward along +X, terrain-relative height, optional 18-point height scan).

## Unitree `terrain_tool` (vendored)

`unitree_terrain_generator.py` is adapted from
[unitree_mujoco/terrain_tool](https://github.com/unitreerobotics/unitree_mujoco/tree/main/terrain_tool)
(BSD / Unitree license as upstream). Upstream helpers: `AddStairs`, `AddSuspendStairs`
(hollow), `AddRoughGround`, Perlin heightfields.

See `UNITREE_TERRAIN_README.md` for original usage. Paths in the vendored script still
assume Unitree's layout; prefer `scripts/generate_stairs_ledges.py` for this repo.

## Research references

Local PDFs and open-source links: [docs/papers/README.md](../../docs/papers/README.md).
