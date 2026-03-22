# quadruped-dog-rl

Quadruped robot dog simulation, walking control, and reinforcement learning policy training workspace.

Supports: Unitree Go1/Go2, Boston Dynamics Spot, MIT Mini Cheetah, ANYmal B/C, Mini Pupper.

![Unitree Go2 in RViz2](docs/images/go2_rviz2.png)

---

## Repository Structure

```
quadruped-dog-rl/
├── urdf/                    # Robot URDF and mesh files
│   ├── go1_config/          # Unitree Go1
│   ├── go2_unitree/         # Unitree Go2
│   ├── spot_config/         # Boston Dynamics Spot
│   ├── mini_cheetah_config/ # MIT Mini Cheetah
│   ├── mini_pupper_config/  # Mini Pupper
│   ├── anymal_b_config/     # ANYmal B
│   └── anymal_c_config/     # ANYmal C
├── ros2/                    # ROS2 packages (CHAMP framework)
│   ├── champ_bringup/       # Launch files for hardware and simulation
│   ├── champ_config/        # Robot-specific configurations
│   ├── champ_description/   # Robot description and URDF loading
│   ├── champ_gazebo/        # Gazebo simulation launch and worlds
│   └── champ_navigation/    # Navigation stack integration
└── training/                # RL policy training (Unitree RL Gym)
    ├── legged_gym/          # PPO training scripts and environments
    ├── deploy/              # Policy deployment to real robot
    └── setup.py             # Package install
```

---

## Requirements

### System
- Ubuntu 22.04
- ROS2 Humble or Jazzy
- Python 3.8+
- NVIDIA GPU (10GB+ VRAM for training)

### Install ROS2 dependencies

```bash
sudo apt install ros-humble-gazebo-ros-pkgs \
                 ros-humble-joint-state-publisher \
                 ros-humble-robot-state-publisher \
                 ros-humble-rviz2
```

### Install CHAMP

```bash
cd ros2
pip install -r champ_bringup/requirements.txt
```

### Install RL training

```bash
cd training
pip install -e .
```

For GPU training also install Isaac Gym:
- Download from https://developer.nvidia.com/isaac-gym
- Follow their `install.sh` instructions

---

## Walking Simulation (ROS2 + Gazebo)

### Launch Go1 in Gazebo

```bash
source /opt/ros/humble/setup.bash
ros2 launch champ_config gazebo.launch.py
```

### Teleoperate with keyboard

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Visualize in RViz2

```bash
ros2 launch champ_description description.launch.py
```

### Switch robot (Spot, Mini Cheetah, etc.)

Edit `ros2/champ_config/config/robot_config.yaml` and set the URDF path to your chosen robot under `urdf/`.

---

## RL Policy Training

### Train a Go2 walking policy

```bash
cd training
python legged_gym/scripts/train.py --task=go2 --headless
```

Training logs and checkpoints are saved under `training/logs/`.

### Play/visualize trained policy

```bash
python legged_gym/scripts/play.py --task=go2
```

### Deploy to real robot

```bash
cd training/deploy
python deploy.py --task=go2 --ckpt=<path_to_checkpoint>
```

---

## Available Robots and URDF Paths

| Robot | URDF Path | Notes |
|-------|-----------|-------|
| Unitree Go1 | `urdf/go1_config/` | Good for sim-to-real |
| Unitree Go2 | `urdf/go2_unitree/urdf/` | Latest Unitree model |
| Boston Dynamics Spot | `urdf/spot_config/` | |
| MIT Mini Cheetah | `urdf/mini_cheetah_config/` | Research platform |
| ANYmal B | `urdf/anymal_b_config/` | ETH Zurich |
| ANYmal C | `urdf/anymal_c_config/` | ETH Zurich |
| Mini Pupper | `urdf/mini_pupper_config/` | Low-cost platform |

---

## References

- [CHAMP Framework](https://github.com/chvmp/champ) — ROS2 locomotion controller
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) — PPO policy training
- [legged_gym (ETH)](https://github.com/leggedrobotics/legged_gym) — Original RL gym
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — Modern training framework
