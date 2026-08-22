"""Split deployable proprio obs from critic-only extras (DreamWaQ-lite).

Wraps envs that put ``info['privileged']['lin_vel']`` on each step (see
``Go2MujocoEnv``). Actor sees ``obs['policy']``; critic sees
``obs['critic']`` = policy ∥ lin_vel(3).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PRIV_DIM = 3  # true base linear velocity (m/s)


class PrivilegedObsWrapper(gym.Wrapper):
    """Dict observation for asymmetric actor-critic training."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise TypeError(
                "PrivilegedObsWrapper expects a flat Box observation space")
        dim = int(base.shape[0])
        self._policy_space = base
        high = np.full(dim + PRIV_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "policy": base,
            "critic": spaces.Box(-high, high, dtype=np.float32),
        })
        self._last_lin_vel = np.zeros(PRIV_DIM, dtype=np.float32)

    def _pack(self, policy_obs: np.ndarray) -> dict:
        policy_obs = np.asarray(policy_obs, dtype=np.float32)
        critic_obs = np.concatenate([policy_obs, self._last_lin_vel], axis=0)
        return {"policy": policy_obs, "critic": critic_obs}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        priv = info.get("privileged", {}) if isinstance(info, dict) else {}
        self._last_lin_vel = np.asarray(
            priv.get("lin_vel", np.zeros(PRIV_DIM)), dtype=np.float32)
        return self._pack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        priv = info.get("privileged", {}) if isinstance(info, dict) else {}
        if "lin_vel" in priv:
            self._last_lin_vel = np.asarray(priv["lin_vel"], dtype=np.float32)
        return self._pack(obs), reward, terminated, truncated, info
