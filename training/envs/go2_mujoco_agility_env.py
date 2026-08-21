"""MuJoCo Go2 env for stance / gait-style / jump-landing agility training."""

from __future__ import annotations

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from intelligence.skills.agility_skills import AgilityCommand, sample_command

SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_scene.xml")

ARM_STOW = [0.0, 1.4, 0.8, 0.3, 0.0, 0.0, 0.0]

DEFAULT_QPOS = np.array([
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
] + ARM_STOW, dtype=np.float32)

ACT_DEFAULT = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
] + ARM_STOW, dtype=np.float32)

# ang_vel(3)+gravity(3)+vel_cmd(3)+height(1)+jump(1)+gait(4)+clock(2)
# +landing_cmd(2)+landing_err(2)+has_jumped(1)
# +dof_pos(19)+dof_vel(19)+prev_action(19)+contacts(4)
OBS_DIM = 83
ACT_DIM = 19
ACT_SCALE = 0.25

EPISODE_LEN_S = 12.0
SIM_DT = 0.005
CTRL_DECIMATION = 4

NOMINAL_HEIGHT = 0.27
ALIVE_BONUS = 0.25
FALL_PENALTY = -8.0
MAX_CURRICULUM_LEVEL = 0.9
LANDING_SUCCESS_DIST = 0.18


class Go2MujocoAgilityEnv(gym.Env):
    """
    Skill-conditioned Go2 locomotion:
      - body height offset (stand / crouch / sit)
      - gait frequency + phase/offset/bound style
      - jump height + XY landing target (upward → forward → diagonal)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode=None,
        randomize_domain=True,
        use_curriculum=True,
        initial_curriculum_level=0.0,
        initial_command: AgilityCommand | None = None,
        jump_bias: float = 0.0,
        scene_xml: str | None = None,
    ):
        super().__init__()
        xml = scene_xml or SCENE_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.render_mode = render_mode
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        self.jump_bias = float(jump_bias)
        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._max_steps = int(EPISODE_LEN_S / (SIM_DT * CTRL_DECIMATION))
        self._last_episode_steps = self._max_steps
        self._sim_time = 0.0
        self._peak_height = NOMINAL_HEIGHT
        self._was_airborne = False
        self._has_jumped = False
        self._landing_scored = False
        self._start_xy = np.zeros(2, dtype=np.float32)

        self.curriculum_level = float(initial_curriculum_level)
        self.cmd = (
            initial_command if initial_command is not None
            else AgilityCommand(vx=0.4, gait_freq=2.0, gait_phase=0.5)
        )

        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_mass = float(self.model.body_mass[self._base_body_id])
        self._base_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._base_gainprm = self.model.actuator_gainprm[:, 0].copy()
        self._base_biasprm1 = self.model.actuator_biasprm[:, 1].copy()

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._act_default = ACT_DEFAULT.copy()

    # ------------------------------------------------------------------ #

    def _gravity_vec(self) -> np.ndarray:
        w, x, y, z = self.data.sensor("orientation").data.astype(np.float32)
        return np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)

    def _get_contacts(self) -> np.ndarray:
        raw = np.array(
            [self.data.sensor(n).data[0]
             for n in ("FL_contact", "FR_contact", "RL_contact", "RR_contact")],
            dtype=np.float32)
        return np.clip(raw / 50.0, 0.0, 1.0)

    def _desired_contacts(self) -> np.ndarray:
        freq = float(self.cmd.gait_freq)
        if freq <= 1e-3:
            return np.ones(4, dtype=np.float32)
        phase = float(self.cmd.gait_phase)
        offset = float(self.cmd.gait_offset)
        bound = float(self.cmd.gait_bound)
        foot_phase = np.array(
            [0.0, phase, offset, (phase + offset + bound) % 1.0], dtype=np.float32)
        clock = (self._sim_time * freq + foot_phase) % 1.0
        return (clock < 0.55).astype(np.float32)

    def _gait_clock_features(self) -> np.ndarray:
        freq = float(self.cmd.gait_freq)
        if freq <= 1e-3:
            return np.zeros(2, dtype=np.float32)
        ang = 2.0 * np.pi * freq * self._sim_time
        return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)

    def _target_height(self) -> float:
        return NOMINAL_HEIGHT + float(self.cmd.height_offset)

    def _landing_error(self) -> np.ndarray:
        desired = self._start_xy + np.array(
            [self.cmd.landing_dx, self.cmd.landing_dy], dtype=np.float32)
        xy = self.data.qpos[:2].astype(np.float32)
        return xy - desired

    def _get_obs(self) -> np.ndarray:
        d = self.data
        ang_vel = d.sensor("ang_vel").data.astype(np.float32) * 0.25
        gravity = self._gravity_vec()
        vel_cmd = np.array(
            [self.cmd.vx, self.cmd.vy, self.cmd.wz], dtype=np.float32
        ) * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        height_cmd = np.array([self.cmd.height_offset * 5.0], dtype=np.float32)
        jump_cmd = np.array([self.cmd.jump_height * 4.0], dtype=np.float32)
        gait = np.array(
            [
                self.cmd.gait_freq * 0.3,
                self.cmd.gait_phase,
                self.cmd.gait_offset,
                self.cmd.gait_bound,
            ],
            dtype=np.float32,
        )
        clock = self._gait_clock_features()
        landing_cmd = np.array(
            [self.cmd.landing_dx * 2.0, self.cmd.landing_dy * 2.0], dtype=np.float32)
        landing_err = self._landing_error() * 2.0
        has_jumped = np.array([1.0 if self._has_jumped else 0.0], dtype=np.float32)
        dof_pos = d.qpos[7:].astype(np.float32) - DEFAULT_QPOS
        dof_vel = d.qvel[6:].astype(np.float32) * 0.05
        contacts = self._get_contacts()
        return np.concatenate(
            [
                ang_vel, gravity, vel_cmd, height_cmd, jump_cmd, gait, clock,
                landing_cmd, landing_err, has_jumped,
                dof_pos, dof_vel, self._prev_action, contacts,
            ]
        )

    def _compute_reward(self, action: np.ndarray):
        d = self.data
        lin_vel = d.sensor("lin_vel").data.astype(np.float32)
        ang_vel = d.sensor("ang_vel").data.astype(np.float32)
        gravity = self._gravity_vec()
        height = float(d.qpos[2])
        contacts = self._get_contacts()
        n_contact = float(np.sum(contacts > 0.25))
        airborne = n_contact < 0.5
        jumping = self.cmd.is_jump

        if airborne:
            self._has_jumped = True

        cmd_speed = float(np.hypot(self.cmd.vx, self.cmd.vy))
        if jumping:
            # During a jump episode, soft velocity tracking toward the landing ray
            desired_vx = float(np.clip(self.cmd.landing_dx * 1.5, -0.5, 1.8))
            desired_vy = float(np.clip(self.cmd.landing_dy * 1.5, -0.8, 0.8))
            r_lin = 0.8 * float(np.exp(
                -((lin_vel[0] - desired_vx) ** 2 + (lin_vel[1] - desired_vy) ** 2) / 0.35))
        else:
            r_lin = 2.0 * float(np.exp(
                -((lin_vel[0] - self.cmd.vx) ** 2 + (lin_vel[1] - self.cmd.vy) ** 2) / 0.1))
        r_ang = 0.5 * float(np.exp(-((ang_vel[2] - self.cmd.wz) ** 2) / 0.25))

        r_landing = 0.0
        r_peak = 0.0
        r_flight = 0.0

        if jumping:
            jump_target = NOMINAL_HEIGHT + float(self.cmd.jump_height)
            if airborne:
                r_height = 2.5 * float(np.exp(-((height - jump_target) ** 2) / 0.025))
                r_flight = 1.0
            else:
                r_height = 1.0 * float(np.exp(-((height - NOMINAL_HEIGHT) ** 2) / 0.01))

            self._peak_height = max(self._peak_height, height)
            err = self._landing_error()
            dist = float(np.linalg.norm(err))

            if airborne:
                # Dense progress toward the landing pad while in flight
                r_landing = 1.5 * float(np.exp(-(dist ** 2) / 0.15))
            elif self._has_jumped and not self._landing_scored:
                # One-shot landing score on first touchdown after flight
                cleared_h = self._peak_height >= jump_target * 0.8
                r_peak = 3.0 if cleared_h else -0.5
                r_landing = 8.0 * float(np.exp(-(dist ** 2) / (LANDING_SUCCESS_DIST ** 2)))
                if dist < LANDING_SUCCESS_DIST and cleared_h:
                    r_landing += 4.0
                self._landing_scored = True
            elif self._has_jumped:
                # Hold the landing pose afterward
                r_landing = 1.2 * float(np.exp(-(dist ** 2) / 0.08))
            self._was_airborne = airborne
        else:
            target_h = self._target_height()
            r_height = 2.0 * float(np.exp(-((height - target_h) ** 2) / 0.008))
            if self.cmd.gait_freq <= 1e-3:
                r_flight = 0.2 * (n_contact / 4.0)

        desired = self._desired_contacts()
        r_gait = 0.4 * float(
            1.0 - np.mean(np.abs(desired - (contacts > 0.25).astype(np.float32))))

        r_z = -0.2 * float(lin_vel[2] ** 2) if jumping else -1.5 * float(lin_vel[2] ** 2)
        r_orient = -0.5 * float(gravity[0] ** 2 + gravity[1] ** 2)
        r_torque = -2e-4 * float(np.sum(d.actuator_force ** 2))
        r_smooth = -5e-3 * float(np.sum((action - self._prev_action) ** 2))

        actual_speed = float(np.hypot(lin_vel[0], lin_vel[1]))
        r_stall = (
            -0.5 if (cmd_speed > 0.2 and actual_speed < 0.25 * cmd_speed and not jumping)
            else 0.0
        )

        components = dict(
            lin=r_lin, ang=r_ang, height=r_height, flight=r_flight, peak=r_peak,
            landing=r_landing, gait=r_gait, vz=r_z, orient=r_orient,
            torque=r_torque, smooth=r_smooth, stall=r_stall, alive=ALIVE_BONUS,
        )
        return float(sum(components.values())), components

    def _apply_domain_rand(self) -> None:
        if not self.randomize_domain:
            return
        rng = self.np_random
        self.model.body_mass[self._base_body_id] = (
            self._base_mass * float(rng.uniform(0.85, 1.15)))
        self.model.geom_friction[self._floor_geom_id] = (
            self._base_floor_friction * float(rng.uniform(0.7, 1.3)))
        kp_scale = float(rng.uniform(0.85, 1.15))
        self.model.actuator_gainprm[:, 0] = self._base_gainprm * kp_scale
        self.model.actuator_biasprm[:, 1] = self._base_biasprm1 * kp_scale

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        z_max = 1.1 if self.cmd.is_jump else 0.8
        if z < 0.12 or z > z_max:
            return True
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > -0.5)

    def _on_reset_scene(self) -> None:
        """Hook for subclasses (parkour) to place obstacles."""
        return None

    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_curriculum:
            success = self._last_episode_steps >= 0.7 * self._max_steps
            self.curriculum_level = float(np.clip(
                self.curriculum_level + (0.006 if success else -0.003),
                0.0, MAX_CURRICULUM_LEVEL))
            self.cmd = sample_command(
                self.np_random,
                curriculum_level=self.curriculum_level,
                jump_bias=self.jump_bias,
            )

        if options and "command" in options:
            self.cmd = options["command"]
            if not isinstance(self.cmd, AgilityCommand):
                self.cmd = AgilityCommand.from_array(self.cmd)

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()
        self._on_reset_scene()

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.42
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:19] = DEFAULT_QPOS[:12] + (self.np_random.random(12) - 0.5) * 0.1
        self.data.qpos[19:] = DEFAULT_QPOS[12:]
        self.data.ctrl[:] = self._act_default
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._last_episode_steps = 0
        self._sim_time = 0.0
        self._peak_height = float(self.data.qpos[2])
        self._was_airborne = False
        self._has_jumped = False
        self._landing_scored = False
        self._start_xy = self.data.qpos[:2].astype(np.float32).copy()
        return self._get_obs(), {"command": self.cmd.as_array()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = self._act_default + action * ACT_SCALE
        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
            self._sim_time += SIM_DT

        reward, components = self._compute_reward(action)
        self._prev_action = action.copy()
        self._step_count += 1
        self._last_episode_steps = self._step_count

        obs = self._get_obs()
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_steps

        if terminated:
            reward += FALL_PENALTY
            components["fall"] = FALL_PENALTY

        if self.render_mode == "human":
            self.render()

        err = self._landing_error()
        return obs, reward, terminated, truncated, {
            "reward_components": components,
            "command": self.cmd.as_array(),
            "height": float(self.data.qpos[2]),
            "landing_error": float(np.linalg.norm(err)),
            "has_jumped": self._has_jumped,
        }

    def set_command(self, command: AgilityCommand | np.ndarray) -> None:
        if isinstance(command, AgilityCommand):
            self.cmd = command
        else:
            self.cmd = AgilityCommand.from_array(command)
        # Changing landing mid-episode re-anchors relative to current start
        self._landing_scored = False

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            self._renderer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            if hasattr(self._renderer, "close"):
                self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = Go2MujocoAgilityEnv(randomize_domain=False, use_curriculum=True, jump_bias=0.4)
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape, "cmd:", info["command"])
    assert obs.shape == (OBS_DIM,), f"expected {OBS_DIM}, got {obs.shape[0]}"
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            obs, info = env.reset()
    print("agility env smoke-test passed")
