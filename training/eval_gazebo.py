#!/usr/bin/env python3
"""Evaluate a Go2 Gazebo RL policy with simple locomotion metrics."""

import argparse
import csv
import os
import time

import numpy as np

from envs.go2_gazebo_env import CTRL_DT, Go2GazeboEnv


def _load_model(path):
    if not path:
        return None
    from stable_baselines3 import PPO
    return PPO.load(path)


def _action(model, obs, env):
    if model is None:
        return np.zeros(env.action_space.shape, dtype=np.float32)
    act, _ = model.predict(obs, deterministic=True)
    return act


def evaluate(args):
    env = Go2GazeboEnv(
        cmd=tuple(args.cmd),
        auto_launch=not args.no_launch,
        ros_domain_id=args.ros_domain_id,
        gz_partition=args.gz_partition,
    )
    model = _load_model(args.model)
    rows = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            start = time.monotonic()
            start_pos = None
            end_pos = None
            rewards = []
            speeds = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                odom_vel = env._node.get_lin_vel()
                if start_pos is None:
                    start_pos = np.zeros(2, dtype=np.float32)
                act = _action(model, obs, env)
                obs, reward, terminated, truncated, _ = env.step(act)
                rewards.append(float(reward))
                speeds.append(float(np.linalg.norm(odom_vel[:2])))
                if len(rewards) >= args.max_steps:
                    truncated = True

            elapsed = time.monotonic() - start
            mean_speed = float(np.mean(speeds)) if speeds else 0.0
            commanded_speed = float(np.linalg.norm(args.cmd[:2]))
            # Gazebo env currently exposes velocity, not world position. Distance
            # is velocity-integrated, which is enough for repeatable policy eval.
            distance = float(sum(speeds) * CTRL_DT)
            end_pos = np.array([distance, 0.0], dtype=np.float32)
            cmd_error = abs(mean_speed - commanded_speed)
            rows.append({
                "episode": ep + 1,
                "steps": len(rewards),
                "reward": round(sum(rewards), 3),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "elapsed_s": round(elapsed, 3),
                "distance_m": round(float(np.linalg.norm(end_pos - start_pos)), 3),
                "mean_speed_mps": round(mean_speed, 3),
                "command_speed_mps": round(commanded_speed, 3),
                "speed_error_mps": round(cmd_error, 3),
            })
    finally:
        env.close()

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="SB3 PPO checkpoint; omitted evaluates zero actions")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--no-launch", action="store_true",
                        help="use an already running Gazebo RL sim")
    parser.add_argument("--ros-domain-id", default="177")
    parser.add_argument("--gz-partition", default="go2rleval")
    parser.add_argument("--csv", default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    rows = evaluate(args)
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
