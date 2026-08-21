# Papers — stairs, ledges & parkour

Local PDFs for Go2 stair climbing, ledge/box climbing, parkour, and adaptation.

**Runnable worlds:** `training/envs/go2_gz_world_stairs.sdf`, `go2_gz_world_ledges.sdf`, MuJoCo `go2_stairs_scene.xml`. Launch: `ros2 launch launch/stairs_ledges_go2.launch.py course:=stairs|ledges` — see [docs/DEMOS.md](../DEMOS.md) and [training/terrain/README.md](../../training/terrain/README.md).

**Pretrained policies:** [docs/PRETRAINED.md](../PRETRAINED.md) + `python3 scripts/download_pretrained.py`.

## Stairs & hollow steps

| File | Paper | Why it matters here |
|------|-------|---------------------|
| `stairmaster_hollow_stairs_2606.25765.pdf` | [StairMaster](https://arxiv.org/abs/2606.25765) | Hollow stairs, depth noise, Go2 → 55° open risers |
| `adaptive_stair_climbing_firefighting_2602.03087.pdf` | [Adaptive stair climbing](https://arxiv.org/abs/2602.03087) | Two-stage RL on Go2 / Isaac Lab; straight, L, spiral |
| `u_shaped_stair_climbing_transfer_2602.14473.pdf` | [U-shaped stair transfer](https://arxiv.org/abs/2602.14473) | Pyramid → U-stair transfer; Go2 / Isaac Lab |
| `blind_stair_climbing_2402.06143.pdf` | [Blind stair climbing](https://arxiv.org/abs/2402.06143) | Blind / wheeled-legged stairs; RSL PPO curriculum |
| `pgtt_phase_guided_terrain_2510.18348.pdf` | [PGTT](https://arxiv.org/abs/2510.18348) | Phase-guided heightmap traversal; stairs/obstacles; Go2 |
| `frnet_fall_recovery_2509.11504.pdf` | [FR-Net](https://arxiv.org/abs/2509.11504) | Fall recovery on steep stairs; Go2 |

## Parkour, gaps & ledges

| File | Paper | Why it matters here |
|------|-------|---------------------|
| `robot_parkour_learning_2309.05665.pdf` | [Robot Parkour Learning](https://arxiv.org/abs/2309.05665) | Climb high boxes, leap gaps, crawl, squeeze; depth → joints |
| `extreme_parkour_2309.14341.pdf` | [Extreme Parkour](https://arxiv.org/abs/2309.14341) | High/long jump ~2× body, ramps; single depth camera |
| `anymal_parkour_agile_nav_2306.14874.pdf` | [ANYmal Parkour](https://arxiv.org/abs/2306.14874) | Hierarchical climb/jump/crouch + nav; stairs & tall boxes |
| `leeps_perceptive_parkour_iros2024.pdf` | [LEEPS](https://sites.google.com/view/leeps) (IROS 2024) | End-to-end perceptive parkour; multi-layer scans |
| `soloparkour_constrained_rl_2409.13678.pdf` | [SoloParkour](https://arxiv.org/abs/2409.13678) | Constrained RL; climb/leap/crawl from depth; Solo-12 |

## Adaptation / recovery

| File | Paper | Why it matters here |
|------|-------|---------------------|
| `rma_rapid_motor_adaptation_2107.04034.pdf` | [RMA](https://arxiv.org/abs/2107.04034) | Rapid adaptation; stairs & rough (blind) |
| `walk_these_ways_2212.03238.pdf` | [Walk These Ways](https://arxiv.org/abs/2212.03238) | Gait multiplicity; curb / stair via behavior |
| `dreamriser_recovery_terrain_2306.12712.pdf` | [DreamRiser](https://arxiv.org/abs/2306.12712) | All-terrain recovery |

## Open-source code (not vendored)

| Project | URL | Notes |
|---------|-----|-------|
| Robot Parkour Learning | https://github.com/ZiwenZhuang/parkour | Gaps, high climb, crawl |
| Extreme Parkour | https://github.com/chengxuxin/extreme-parkour | Parkour stack + depth |
| IsaacLab-Quadruped-Tasks | https://github.com/felipemohr/IsaacLab-Quadruped-Tasks | Go2 blind/vision stairs tasks |
| go2-blind-stairs-newton | https://github.com/NMadhub/go2-blind-stairs-newton | Blind stairs on Newton / Isaac Lab |
| go2-sim2real-locomotion-rl | https://github.com/saifahmadgit/go2-sim2real-locomotion-rl | Genesis; stair curriculum for Go2 |

Sources: `https://arxiv.org/pdf/<id>`. LEEPS is the author-hosted IROS PDF.
