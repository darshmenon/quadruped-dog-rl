# Robot Descriptions

This folder contains documentation for each robot's URDF and configuration.

## Unitree Go1

- URDF: `urdf/go1_config/`
- DOF: 12 (3 per leg)
- Weight: ~12 kg
- Use case: Sim-to-real transfer, general locomotion research

## Unitree Go2

- URDF: `urdf/go2_unitree/urdf/`
- DOF: 12 (3 per leg)
- Weight: ~15 kg
- Use case: Latest Unitree platform, best RL support

## Boston Dynamics Spot

- URDF: `urdf/spot_config/`
- DOF: 12 (3 per leg)
- Weight: ~32 kg
- Use case: Industrial applications, navigation

## MIT Mini Cheetah

- URDF: `urdf/mini_cheetah_config/`
- DOF: 12 (3 per leg)
- Weight: ~9 kg
- Use case: High-speed locomotion research

## ANYmal B

- URDF: `urdf/anymal_b_config/`
- DOF: 12 (3 per leg)
- Weight: ~30 kg
- Use case: ETH Zurich research, rugged terrain

## ANYmal C

- URDF: `urdf/anymal_c_config/`
- DOF: 12 (3 per leg)
- Weight: ~50 kg
- Use case: Industrial inspection

## Mini Pupper

- URDF: `urdf/mini_pupper_config/`
- DOF: 12 (3 per leg)
- Weight: ~0.56 kg
- Use case: Low-cost education and experimentation

## URDF Joint Convention

All robots follow the standard quadruped joint naming:
- `FL` = Front Left
- `FR` = Front Right
- `RL` = Rear Left
- `RR` = Rear Right

Joint order per leg: Hip Abduction/Adduction (HAA), Hip Flexion/Extension (HFE), Knee Flexion/Extension (KFE)
