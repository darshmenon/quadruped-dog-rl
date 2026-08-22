"""
High-level navigation env: outputs velocity commands to a frozen walk policy.

HelixNav-style hierarchy without Isaac Lab: HL PPO learns (vx, vy, wz) to
reach a random planar goal; a pretrained low-level walk checkpoint executes
locomotion for ``ll_steps`` per HL step.

Usage (train):
    python3 training/train_hl_nav.py \\
      --walk-model training/logs/mujoco/best_model.zip \\
      --timesteps 200000
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from envs.go2_mujoco_env import Go2MujocoEnv, CTRL_DECIMATION, SIM_DT
except ImportError:
    from go2_mujoco_env import Go2MujocoEnv, CTRL_DECIMATION, SIM_DT  # type: ignore

LL_STEPS = 25          # low-level steps per HL decision (~0.5 s @ 50 Hz)
GOAL_RADIUS = 0.35
MAX_GOAL_DIST = 8.0
MIN_GOAL_DIST = 2.0
CMD_SCALE = np.array([1.0, 0.6, 1.2], dtype=np.float32)  # vx, vy, wz max


class Go2HLNavEnv(gym.Env):
    """Planar goal reaching via cmd_vel over a frozen SB3 walk policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        walk_model,
        walk_vecnorm=None,
        ll_steps: int = LL_STEPS,
        render_mode=None,
    ):
        super().__init__()
        self.env = Go2MujocoEnv(
            cmd=(0.0, 0.0, 0.0),
            render_mode=render_mode,
            randomize_domain=True,
            use_curriculum=False,
            push_robots=True,
        )
        self.walk_model = walk_model
        self.walk_vecnorm = walk_vecnorm
        self.ll_steps = int(ll_steps)
        self.goal = np.zeros(2, dtype=np.float32)

        # rel_goal(2) + heading_to_goal(2) + dist(1) + lin_vel(2) + gravity_z(1)
        self._obs_dim = 8
        obs_high = np.full(self._obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _heading(self) -> float:
        w, x, y, z = self.env.data.sensor("orientation").data
        # yaw from quaternion (z-axis rotation)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny, cosy))

    def _sample_goal(self) -> None:
        rng = self.env.np_random
        dist = float(rng.uniform(MIN_GOAL_DIST, MAX_GOAL_DIST))
        ang = float(rng.uniform(-np.pi, np.pi))
        px, py = float(self.env.data.qpos[0]), float(self.env.data.qpos[1])
        self.goal[0] = px + dist * np.cos(ang)
        self.goal[1] = py + dist * np.sin(ang)

    def _hl_obs(self) -> np.ndarray:
        px, py = float(self.env.data.qpos[0]), float(self.env.data.qpos[1])
        dx, dy = self.goal[0] - px, self.goal[1] - py
        dist = float(np.hypot(dx, dy))
        yaw = self._heading()
        bearing = float(np.arctan2(dy, dx))
        err = float(np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw)))
        lin = self.env.data.sensor("lin_vel").data.astype(np.float32)
        gz = float(self.env._gravity_vec()[2])
        return np.array(
            [dx, dy, np.cos(err), np.sin(err), dist, lin[0], lin[1], gz],
            dtype=np.float32)

    def _ll_predict(self, ll_obs: np.ndarray) -> np.ndarray:
        obs_in = ll_obs.reshape(1, -1)
        if self.walk_vecnorm is not None:
            obs_in = self.walk_vecnorm.normalize_obs(obs_in)
        action, _ = self.walk_model.predict(obs_in, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ll_obs, _ = self.env.reset(seed=seed)
        self._sample_goal()
        return self._hl_obs(), {"goal": self.goal.copy()}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        cmd = action * CMD_SCALE
        self.env.cmd = cmd.astype(np.float32)

        ep_reward = 0.0
        terminated = False
        truncated = False
        ll_obs = self.env._get_obs()

        for _ in range(self.ll_steps):
            ll_action = self._ll_predict(ll_obs)
            ll_obs, r, term, trunc, info = self.env.step(ll_action)
            ep_reward += float(r)
            if term:
                terminated = True
                break
            if trunc:
                truncated = True
                break

        px, py = float(self.env.data.qpos[0]), float(self.env.data.qpos[1])
        dist = float(np.hypot(self.goal[0] - px, self.goal[1] - py))
        r_goal = -0.05 * dist
        r_success = 5.0 if dist < GOAL_RADIUS else 0.0
        r_cmd = -0.02 * float(np.sum(action ** 2))
        reward = r_goal + r_success + r_cmd + 0.01 * ep_reward

        if dist < GOAL_RADIUS:
            self._sample_goal()

        obs = self._hl_obs()
        info = {
            "goal_dist": dist,
            "cmd": cmd.copy(),
            "ll_reward_sum": ep_reward,
            "success": dist < GOAL_RADIUS,
        }
        return obs, reward, terminated, truncated, info
