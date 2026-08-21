"""Stack the last N proprioceptive observations into one flat vector.

Gives the policy a short temporal window so it can infer velocities,
contact events, and rough terrain properties from proprioception alone
without an explicit estimator network. Opt-in via ObsHistoryWrapper;
default training stays single-frame for checkpoint compatibility.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ObsHistoryWrapper(gym.Wrapper):
    """Concatenate the last ``history_len`` observations along the feature axis.

    On reset the buffer is filled by repeating the first observation so the
    observation shape is constant from step 0.
    """

    def __init__(self, env: gym.Env, history_len: int = 5):
        super().__init__(env)
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}")
        self.history_len = int(history_len)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise TypeError(
                "ObsHistoryWrapper expects a flat Box observation space, "
                f"got {type(base)} shape={getattr(base, 'shape', None)}")
        self._obs_dim = int(base.shape[0])
        low = np.tile(base.low, self.history_len)
        high = np.tile(base.high, self.history_len)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=base.dtype)
        self._buf: deque[np.ndarray] = deque(maxlen=self.history_len)

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._buf), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        self._buf.clear()
        for _ in range(self.history_len):
            self._buf.append(obs.copy())
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        self._buf.append(obs.copy())
        return self._stack(), reward, terminated, truncated, info
