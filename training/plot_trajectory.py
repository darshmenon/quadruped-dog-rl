#!/usr/bin/env python3
"""Plot one eval episode's base trajectory/height/tilt/reward, to see *how*
a policy falls (tips forward, drifts sideways, stumbles immediately, etc.)
rather than just the aggregate steps/fall-rate numbers from play_policy.py.

Usage:
    python3 training/plot_trajectory.py --model training/logs/mujoco/go2_mujoco_final.zip
    python3 training/plot_trajectory.py --model best_model.zip --out docs/images/trajectory.png
"""
import argparse
import os
import sys

sys.modules.setdefault("triton", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from play_policy import _make_env

# Same thresholds _is_terminated() in envs/go2_mujoco_env.py checks against.
Z_MIN, Z_MAX = 0.15, 0.8
TILT_TERM = -0.5  # 1 - 2*(w^2 + z_q^2) crossing this ~= 60 degrees roll/pitch


def _run_episode(env, model, core, max_steps):
    obs = env.reset()
    xs, ys, zs, tilts, rewards = [], [], [], [], []
    for _ in range(max_steps):
        x, y, z = core.data.qpos[0], core.data.qpos[1], core.data.qpos[2]
        w, qx, qy, qz = core.data.sensor("orientation").data
        tilt = 1 - 2 * (w * w + qz * qz)
        xs.append(float(x)); ys.append(float(y)); zs.append(float(z)); tilts.append(float(tilt))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward[0]))
        if done[0]:
            break
    return np.array(xs), np.array(ys), np.array(zs), np.array(tilts), np.array(rewards)


def plot(xs, ys, zs, tilts, rewards, out_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(xs, ys, "-o", markersize=2, linewidth=1)
    ax.plot(xs[0], ys[0], "go", label="start")
    ax.plot(xs[-1], ys[-1], "rx", label="end (fall/timeout)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("base xy trajectory"); ax.axis("equal"); ax.legend()

    steps = np.arange(len(zs))
    ax = axes[0, 1]
    ax.plot(steps, zs)
    ax.axhline(Z_MIN, color="r", linestyle="--", label=f"terminate z<{Z_MIN}")
    ax.axhline(Z_MAX, color="r", linestyle="--")
    ax.set_xlabel("step"); ax.set_ylabel("base height z (m)")
    ax.set_title("base height"); ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, tilts)
    ax.axhline(TILT_TERM, color="r", linestyle="--", label=f"terminate tilt<{TILT_TERM}")
    ax.set_xlabel("step"); ax.set_ylabel("1 - 2(w²+qz²)  [1=upright]")
    ax.set_title("tilt (roll/pitch proxy)"); ax.legend()

    ax = axes[1, 1]
    ax.plot(np.arange(len(rewards)), rewards)
    ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.set_title("per-step reward")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnorm", default=None)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--scene", choices=("flat", "stairs", "rough"), default="flat")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", default="training/logs/mujoco/trajectory.png")
    args = parser.parse_args()

    cmd = tuple(args.cmd)
    raw = _make_env(args.scene, cmd, use_vision=(args.scene != "flat"))
    env = DummyVecEnv([lambda: Monitor(raw)])

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        for candidate in [
            os.path.join(os.path.dirname(args.model), "vecnorm_final.pkl"),
            os.path.join(os.path.dirname(args.model), "..", "vecnorm_final.pkl"),
        ]:
            if os.path.exists(candidate):
                vecnorm_path = os.path.normpath(candidate)
                break
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize stats: {vecnorm_path}")

    model = PPO.load(args.model, env=env)

    core = raw
    while hasattr(core, "env"):
        core = core.env

    xs, ys, zs, tilts, rewards = _run_episode(env, model, core, args.max_steps)
    fell = len(zs) < args.max_steps
    title = (f"{os.path.basename(args.model)}  cmd={cmd}  "
             f"steps={len(zs)}  {'FELL' if fell else 'survived full episode'}")
    plot(xs, ys, zs, tilts, rewards, args.out, title)


if __name__ == "__main__":
    main()
