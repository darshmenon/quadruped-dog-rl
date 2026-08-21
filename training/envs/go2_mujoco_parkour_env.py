"""MuJoCo Go2 parkour env — jump curriculum with hurdles / platforms."""

from __future__ import annotations

import os
import sys

import numpy as np
import mujoco

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from intelligence.skills.agility_skills import AgilityCommand, command_from_jump_target
from intelligence.skills.jump_curriculum import JumpStage, sample_jump_target

# Prefer package import; fall back when run as a script from training/envs/
try:
    from envs.go2_mujoco_agility_env import Go2MujocoAgilityEnv, OBS_DIM
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from envs.go2_mujoco_agility_env import Go2MujocoAgilityEnv, OBS_DIM

PARKOUR_SCENE = os.path.join(os.path.dirname(__file__), "go2_parkour_scene.xml")
HURDLE_NAMES = ("hurdle_0", "hurdle_1", "hurdle_2")
PLATFORM_NAME = "platform_0"
PARKED = np.array([0.0, 0.0, -5.0], dtype=np.float64)


class Go2MujocoParkourEnv(Go2MujocoAgilityEnv):
    """
    Extends agility training with a staged jump curriculum and physical
    hurdles placed between the start pose and the landing pad.

    Curriculum stages (via jump_curriculum.stage_for_level):
      upward → forward → diagonal → over_obstacle
    """

    def __init__(
        self,
        render_mode=None,
        randomize_domain=True,
        use_curriculum=True,
        initial_curriculum_level=0.0,
        initial_command: AgilityCommand | None = None,
    ):
        super().__init__(
            render_mode=render_mode,
            randomize_domain=randomize_domain,
            use_curriculum=use_curriculum,
            initial_curriculum_level=initial_curriculum_level,
            initial_command=initial_command,
            jump_bias=0.65,
            scene_xml=PARKOUR_SCENE,
        )
        self._hurdle_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in HURDLE_NAMES
        ]
        self._platform_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, PLATFORM_NAME)
        self._last_jump_target = None
        self._cleared_obstacle = False

    def _park_all(self) -> None:
        for gid in self._hurdle_ids:
            self.model.geom_pos[gid] = PARKED
            self.model.geom_size[gid] = [0.05, 0.2, 0.05]
        self.model.geom_pos[self._platform_id] = PARKED
        self.model.geom_size[self._platform_id] = [0.2, 0.2, 0.05]

    def _place_hurdle(self, x: float, y: float, height: float,
                      width: float, span: float, slot: int = 0) -> None:
        gid = self._hurdle_ids[slot % len(self._hurdle_ids)]
        half_h = max(0.03, height * 0.5)
        self.model.geom_size[gid] = [width, span, half_h]
        self.model.geom_pos[gid] = [x, y, half_h]

    def _place_platform(self, x: float, y: float, height: float = 0.08) -> None:
        half_h = height * 0.5
        self.model.geom_size[self._platform_id] = [0.30, 0.30, half_h]
        self.model.geom_pos[self._platform_id] = [x, y, half_h]

    def _on_reset_scene(self) -> None:
        self._park_all()
        self._cleared_obstacle = False

        if self.use_curriculum or self.cmd is None or not self.cmd.is_jump:
            target = sample_jump_target(
                self.np_random, curriculum_level=self.curriculum_level)
            self._last_jump_target = target
            self.cmd = command_from_jump_target(target)
        else:
            # Fixed-command eval: synthesize a matching target from cmd
            from intelligence.skills.jump_curriculum import JumpTarget
            stage = JumpStage.FORWARD if abs(self.cmd.landing_dx) > 0.1 else JumpStage.UPWARD
            if self.curriculum_level >= 0.75:
                stage = JumpStage.OVER_OBSTACLE
            target = JumpTarget(
                stage=stage,
                landing_dx=self.cmd.landing_dx,
                landing_dy=self.cmd.landing_dy,
                jump_height=self.cmd.jump_height,
                obstacle_height=0.10 if stage == JumpStage.OVER_OBSTACLE else 0.0,
            )
            self._last_jump_target = target

        if target.stage == JumpStage.OVER_OBSTACLE and target.obstacle_height > 0.0:
            hx = 0.45 * target.landing_dx
            hy = 0.45 * target.landing_dy
            self._place_hurdle(
                hx, hy,
                height=target.obstacle_height,
                width=target.obstacle_width,
                span=target.obstacle_span,
                slot=0,
            )
            if self.curriculum_level > 0.85 and self.np_random.random() < 0.4:
                self._place_hurdle(
                    0.7 * target.landing_dx,
                    0.7 * target.landing_dy,
                    height=target.obstacle_height * 0.7,
                    width=target.obstacle_width,
                    span=target.obstacle_span * 0.8,
                    slot=1,
                )
        elif target.stage in (JumpStage.FORWARD, JumpStage.DIAGONAL):
            if self.np_random.random() < 0.35 * max(self.curriculum_level, 0.1):
                self._place_platform(
                    target.landing_dx * 0.9,
                    target.landing_dy * 0.9,
                    height=0.05 + 0.04 * self.curriculum_level,
                )

    def _compute_reward(self, action: np.ndarray):
        reward, components = super()._compute_reward(action)

        target = self._last_jump_target
        if (
            target is not None
            and target.stage == JumpStage.OVER_OBSTACLE
            and not self._cleared_obstacle
        ):
            hx = 0.45 * target.landing_dx
            height = float(self.data.qpos[2])
            x = float(self.data.qpos[0])
            if x > hx + 0.05 and height > target.obstacle_height + 0.18:
                components["clear"] = 5.0
                reward += 5.0
                self._cleared_obstacle = True

        return reward, components

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["jump_stage"] = (
            self._last_jump_target.stage.value if self._last_jump_target else "none"
        )
        if self._last_jump_target is not None:
            info["obstacle_height"] = self._last_jump_target.obstacle_height
        return obs, info


if __name__ == "__main__":
    env = Go2MujocoParkourEnv(
        randomize_domain=False, use_curriculum=True, initial_curriculum_level=0.9)
    obs, info = env.reset(seed=1)
    print(
        "obs", obs.shape,
        "stage", info.get("jump_stage"),
        "obst_h", info.get("obstacle_height"),
        "cmd", info["command"],
    )
    assert obs.shape == (OBS_DIM,)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            obs, info = env.reset()
    print("parkour env smoke-test passed")
