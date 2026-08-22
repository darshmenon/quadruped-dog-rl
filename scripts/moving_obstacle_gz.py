#!/usr/bin/env python3
"""Oscillate a Gazebo Harmonic model pose (moving obstacle for Go2 demos).

Requires a running gz-sim world that contains ``--model`` (default
``moving_person`` from ``training/envs/go2_gz_world_moving.sdf``).

Usage:
  # terminal 1
  ros2 launch launch/champ_go2_gazebo.launch.py \\
    world:=$(pwd)/training/envs/go2_gz_world_moving.sdf headless:=true
  # terminal 2
  python3 scripts/moving_obstacle_gz.py --model moving_person --amp 2.0 --period 8
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time


def set_pose(world: str, model: str, x: float, y: float, z: float) -> int:
    # gz service set_pose — Harmonic: /world/<world>/set_pose
    req = (
        f"name: \"{model}\", "
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: 0, y: 0, z: 0, w: 1}}"
    )
    cmd = [
        "gz", "service", "-s", f"/world/{world}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="go2_rl")
    p.add_argument("--model", default="moving_person")
    p.add_argument("--amp", type=float, default=2.0, help="Y oscillation amplitude [m]")
    p.add_argument("--x", type=float, default=2.5)
    p.add_argument("--z", type=float, default=0.9)
    p.add_argument("--period", type=float, default=8.0)
    p.add_argument("--hz", type=float, default=20.0)
    args = p.parse_args()

    print(f"Moving {args.model} in world={args.world}  "
          f"x={args.x} y=±{args.amp} period={args.period}s")
    t0 = time.time()
    dt = 1.0 / args.hz
    while True:
        t = time.time() - t0
        y = args.amp * math.sin(2.0 * math.pi * t / args.period)
        rc = set_pose(args.world, args.model, args.x, y, args.z)
        if rc != 0 and int(t) % 5 == 0 and abs(t - int(t)) < dt:
            print(f"  warn: set_pose rc={rc} (is gz-sim running with world {args.world}?)")
        time.sleep(dt)


if __name__ == "__main__":
    raise SystemExit(main())
