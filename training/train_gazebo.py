"""
Train Unitree Go2 walking policy using Gazebo (headless) + ROS2 + SB3 PPO.

Requires ROS2 sourced. Gazebo is auto-launched headlessly.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2/install/setup.bash
    python3 training/train_gazebo.py
    python3 training/train_gazebo.py --timesteps 500000 --cmd 0.5 0.0 0.0
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))

# torch lazily imports triton when SB3 builds the Adam optimizer; a broken
# local triton/CUDA-driver combo segfaults there, so block the import.
sys.modules.setdefault("triton", None)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.go2_gazebo_env import Go2GazeboEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "gazebo")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-launch", action="store_true",
                        help="don't auto-launch Gazebo (use externally running sim)")
    parser.add_argument("--ros-domain-id", default="177",
                        help="ROS_DOMAIN_ID isolating this run from other concurrent "
                             "ROS2/Gazebo sessions on this machine")
    parser.add_argument("--gz-partition", default="go2rltrain",
                        help="GZ_PARTITION isolating this run's Gazebo transport")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    cmd = tuple(args.cmd)

    print(f"Training Go2 (Gazebo) | cmd={cmd} | steps={args.timesteps} | "
          f"ROS_DOMAIN_ID={args.ros_domain_id} GZ_PARTITION={args.gz_partition}")
    print("Launching Gazebo headlessly..." if not args.no_launch else "Using existing Gazebo...")

    env = Monitor(Go2GazeboEnv(cmd=cmd, auto_launch=not args.no_launch,
                                ros_domain_id=args.ros_domain_id, gz_partition=args.gz_partition))

    # No EvalCallback here: unlike the MuJoCo backend, Go2GazeboEnv wraps one
    # live Gazebo sim reached over global ROS2 topics (/joint_states, /odom,
    # ...), not a cheap in-process physics object -- there's no separate env
    # to evaluate on without launching a second Gazebo instance fighting the
    # first over the same topic names. Pointing EvalCallback at the same env
    # PPO is training on made it reset/step that env out from under the
    # rollout collector mid-episode every eval_freq steps, corrupting
    # whatever rollout was in progress. Rely on periodic checkpoints instead
    # and evaluate saved checkpoints out-of-band.
    callbacks = [
        CheckpointCallback(save_freq=10_000, save_path=CKPT_DIR, name_prefix="go2_gazebo"),
    ]

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=LOG_DIR)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=500,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not bool(args.resume))
    model.save(os.path.join(LOG_DIR, "go2_gazebo_final"))
    print("Training done. Model saved to", LOG_DIR)
    env.close()


if __name__ == "__main__":
    main()
