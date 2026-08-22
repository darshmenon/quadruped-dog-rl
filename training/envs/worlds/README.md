# Gazebo worlds for Go2

## Ready for Go2 launches (`world name="go2_rl"`)

| File | Use |
|------|-----|
| `../go2_gz_world.sdf` | Flat default |
| `../go2_gz_world_room.sdf` | Indoor SLAM / frontier |
| `../go2_gz_world_outdoor.sdf` | Outdoor |
| `../go2_gz_world_stairs.sdf` / `ledges` | Stair curriculum |
| `../go2_gz_world_arena.sdf` | **New** bounded obstacle arena (from go2-quadruped-sim) |
| `../go2_gz_world_warehouse.sdf` | **New** aisle/shelf warehouse (Fuel-inspired, self-contained) |
| `../go2_gz_world_moving.sdf` | **New** room + moving cylinder (`scripts/moving_obstacle_gz.py`) |

```bash
ros2 launch launch/champ_go2_gazebo.launch.py \
  world:=$(pwd)/training/envs/go2_gz_world_arena.sdf

ros2 launch launch/champ_go2_gazebo.launch.py \
  world:=$(pwd)/training/envs/go2_gz_world_warehouse.sdf

ros2 launch launch/champ_go2_gazebo.launch.py \
  world:=$(pwd)/training/envs/go2_gz_world_moving.sdf headless:=true
# other terminal:
python3 scripts/moving_obstacle_gz.py
```

## Downloaded Fuel / upstream (not auto-launched)

- `fuel/*.sdf` — industrial warehouse, cave, tugbot warehouse, garden, …
- `upstream/` — classic `.world` / unitree default copies
- `_src/` — shallow clones (gitignored); refresh via `python3 scripts/download_worlds.py`

```bash
python3 scripts/download_worlds.py --list
python3 scripts/download_worlds.py --only fuel
gz sim training/envs/worlds/fuel/industrial_warehouse.sdf
```
