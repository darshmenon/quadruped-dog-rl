"""
Train high-level cmd_vel navigator over a frozen walk policy (HelixNav-lite).

Usage:
    python3 training/train_hl_nav.py \\
      --walk-model training/logs/mujoco/best_model.zip \\
      --timesteps 200000 --n-envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("triton", None)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go2_hl_nav_env import Go2HLNavEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "hl_nav")


def main():
    parser = argparse.ArgumentParser(description="Train HL nav over frozen walk policy")
    parser.add_argument("--walk-model", type=str, required=True,
                        help="SB3 .zip low-level walk checkpoint")
    parser.add_argument("--walk-vecnorm", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--ll-steps", type=int, default=25)
    args = parser.parse_args()

    walk = PPO.load(args.walk_model)
    vn_path = args.walk_vecnorm
    if vn_path is None:
        guess = os.path.join(os.path.dirname(args.walk_model), "vecnorm_final.pkl")
        if os.path.exists(guess):
            vn_path = guess

    walk_vn = None
    if vn_path and os.path.exists(vn_path):
        from envs.go2_mujoco_env import Go2MujocoEnv
        dummy = DummyVecEnv([lambda: Monitor(Go2MujocoEnv())])
        walk_vn = VecNormalize.load(vn_path, dummy)
        walk_vn.training = False
        walk_vn.norm_reward = False
        print(f"Walk VecNormalize: {vn_path}")

    def _make(rank):
        def _init():
            e = Go2HLNavEnv(walk, walk_vecnorm=walk_vn, ll_steps=args.ll_steps)
            return Monitor(e)
        return _init

    vec_env = DummyVecEnv([_make(i) for i in range(args.n_envs)])
    os.makedirs(LOG_DIR, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    out = os.path.join(LOG_DIR, "hl_nav_final.zip")
    model.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
