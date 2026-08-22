"""MuJoCo fall-recovery env for Unitree Go2 (FR-Net-style baseline).

Inspired by lu-yidan/FR-Net ``go2_recovery`` (IEEE RA-L 2025 baseline) and
iit-DLSLab/get-up-isaaclab: each episode starts from a random fallen pose;
the policy uses plain 45-dim proprioception + 12-DOF leg PD targets to flip
upright and stand. Arm actuators stay at the walk-env stow pose.

Refs (cloned under ~/quad_inspo_2026/): FR-Net, get-up-isaaclab.
"""

from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from envs.go2_mujoco_env import (
        ARM_STOW, SCENE_XML, SIM_DT, CTRL_DECIMATION, CTRL_DT)
except ImportError:
    from go2_mujoco_env import (  # type: ignore
        ARM_STOW, SCENE_XML, SIM_DT, CTRL_DECIMATION, CTRL_DT)

ROUGH_SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_rough_scene.xml")

# Actuator order for the 12 legs (matches go2_scene.xml <actuator> block).
ACT_DIM = 12
ACT_SCALE = 0.25
OBS_DIM = 45  # ang3 + grav3 + cmd3 + q12 + dq12 + a12  (FR-Net layout)

# Curled PD reference (FR-Net default_joint_angles), actuator order.
ACT_CURL = np.array([
    0.0, 0.0, 0.0, 0.0,          # hips FL FR RL RR
    1.5, 1.5, 1.5, 1.5,          # thighs
    -2.4, -2.4, -2.4, -2.4,      # calves
], dtype=np.float32)

# Standing targets in actuator order (from FR-Net stand_high / stand_low).
STAND_HIGH = np.array([
    -0.05, 0.05, -0.05, 0.05,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)
STAND_LOW = ACT_CURL.copy()

# qpos[7:19] is leg-by-leg FL/FR/RL/RR × (hip,thigh,calf). Map to actuator order.
_QPOS_TO_ACT = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32)

EPISODE_LEN_S = 24.0
SPAWN_HEIGHT = 0.50
BASE_HEIGHT_TARGET = 0.40
ORIENT_EPS = 0.15          # Gaussian upright width (gravity_z ≈ -1)
STAND_POSE_EPS = 0.25      # only score stand-pose when nearly upright
ROTATED_FAIL_STEPS = 400   # ~8 s at 50 Hz — give up if still inverted
UPRIGHT_HOLD_STEPS = 50    # ~1 s at 50 Hz — sustained-upright threshold for is_success
ONLY_POSITIVE_REWARDS = True

# Reward scales (FR-Net recovery table, single-env SB3 friendly).
W_ORIENT = -0.5
W_ORIENT_GAUSS = 6.0
W_HEIGHT = 1.0
W_FOOT = 0.1
W_STAND = 6.0
W_ANG_VEL = -0.05
W_TORQUE = -2e-4
W_ACTION = -1e-2
W_ACTION_RATE = -0.02
W_DOF_VEL = -2e-3
W_MAX_VEL = -0.1
W_SETTLE = -0.4            # -vz^2 when nearly upright (go2-rl get-up settle)
MAX_JOINT_VEL = 8.0


class Go2MujocoRecoveryEnv(gym.Env):
    """Fall recovery from randomized orientations; legs-only 12-DOF control."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, randomize_domain=True,
                 cmd=(0.0, 0.0, 0.0), test_mode=False,
                 initial_curriculum_level=0.0, scene_xml=None):
        super().__init__()
        self._scene_xml = scene_xml or SCENE_XML
        self._rough_terrain = os.path.basename(self._scene_xml) == os.path.basename(
            ROUGH_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(self._scene_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.render_mode = render_mode
        self.randomize_domain = randomize_domain
        self.test_mode = bool(test_mode)
        self.cmd = np.array(cmd, dtype=np.float32)
        self.curriculum_level = float(np.clip(initial_curriculum_level, 0.0, 1.0))

        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._last_last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._rotated_steps = 0
        self._max_steps = int(EPISODE_LEN_S / CTRL_DT)
        self._last_upright = False
        self._success_streak = 0
        self._upright_hold = 0

        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_mass = float(self.model.body_mass[self._base_body_id])
        self._base_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._base_gainprm = self.model.actuator_gainprm[:12, 0].copy()
        self._base_biasprm1 = self.model.actuator_biasprm[:12, 1].copy()
        self._hfield_id = -1
        self._hfield_nrow = self._hfield_ncol = 0
        self._hfield_size = np.zeros(4, dtype=np.float64)
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hid >= 0:
            self._hfield_id = hid
            self._hfield_nrow = int(self.model.hfield_nrow[hid])
            self._hfield_ncol = int(self.model.hfield_ncol[hid])
            self._hfield_size = self.model.hfield_size[hid].copy()

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ #

    def _gravity_vec(self) -> np.ndarray:
        w, x, y, z = self.data.sensor("orientation").data.astype(np.float32)
        return np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)

    def _leg_qpos_act(self) -> np.ndarray:
        return self.data.qpos[7:19].astype(np.float32)[_QPOS_TO_ACT]

    def _leg_qvel_act(self) -> np.ndarray:
        return self.data.qvel[6:18].astype(np.float32)[_QPOS_TO_ACT]

    def _get_contacts(self) -> np.ndarray:
        raw = np.array(
            [self.data.sensor(n).data[0]
             for n in ("FL_contact", "FR_contact", "RL_contact", "RR_contact")],
            dtype=np.float32)
        return np.clip(raw / 50.0, 0.0, 1.0)

    def _stand_target(self) -> np.ndarray:
        # Curriculum: curl → standing pose as recoveries succeed.
        t = float(self.curriculum_level)
        return STAND_LOW + t * (STAND_HIGH - STAND_LOW)

    def _terrain_height_at(self, x: float, y: float) -> float:
        if self._hfield_id < 0:
            return 0.0
        sx, sy, elev, _ = self._hfield_size
        u = (x / sx + 1.0) * 0.5
        v = (y / sy + 1.0) * 0.5
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        adr = int(self.model.hfield_adr[self._hfield_id])
        nrow, ncol = self._hfield_nrow, self._hfield_ncol
        row = int(np.clip(v * (nrow - 1), 0, nrow - 1))
        col = int(np.clip(u * (ncol - 1), 0, ncol - 1))
        idx = adr + row * ncol + col
        return float(self.model.hfield_data[idx]) * elev

    def _random_unit_quat(self) -> np.ndarray:
        q = self.np_random.normal(size=4).astype(np.float64)
        q /= np.linalg.norm(q) + 1e-12
        return q.astype(np.float32)

    def _apply_domain_rand(self) -> None:
        if not self.randomize_domain:
            return
        rng = self.np_random
        self.model.body_mass[self._base_body_id] = (
            self._base_mass * float(rng.uniform(0.85, 1.15)))
        self.model.geom_friction[self._floor_geom_id] = (
            self._base_floor_friction * float(rng.uniform(0.7, 1.3)))
        kp_scale = float(rng.uniform(0.85, 1.15))
        self.model.actuator_gainprm[:12, 0] = self._base_gainprm * kp_scale
        self.model.actuator_biasprm[:12, 1] = self._base_biasprm1 * kp_scale

    def _is_upright(self, gravity: np.ndarray | None = None) -> bool:
        # Termination/threshold curriculum: the hard -0.7 bar was never once
        # crossed even after 2M+ steps of orient_gauss/stand climbing (both
        # give continuous partial credit for *approaching* upright, but the
        # discrete is_success/upright_frac check requires fully crossing a
        # fixed threshold from step one -- a known PPO failure mode where a
        # too-strict binary condition stays at 0 no matter how much the
        # shaped reward improves). Loosen the bar early in curriculum_level
        # and tighten it back to the real target as the policy improves, same
        # curl->stand pattern _stand_target() already uses.
        g = self._gravity_vec() if gravity is None else gravity
        t = float(self.curriculum_level)
        gravity_thresh = -0.3 - 0.4 * t   # -0.3 (easy) -> -0.7 (true target)
        return bool(g[2] < gravity_thresh and float(self.data.qpos[2]) > 0.25)

    def _get_obs(self) -> np.ndarray:
        ang_vel = self.data.sensor("ang_vel").data.astype(np.float32) * 0.25
        gravity = self._gravity_vec()
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        dof_pos = self._leg_qpos_act() - ACT_CURL
        dof_vel = self._leg_qvel_act() * 0.05
        return np.concatenate(
            [ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action])

    def _compute_reward(self, action: np.ndarray):
        d = self.data
        gravity = self._gravity_vec()
        ang_vel = d.sensor("ang_vel").data.astype(np.float32)
        contacts = self._get_contacts()
        leg_q = self._leg_qpos_act()
        leg_dq = self._leg_qvel_act()
        stand_tgt = self._stand_target()

        # Orientation: penalty on tilt + Gaussian bonus when upright.
        r_orient = W_ORIENT * float(gravity[0] ** 2 + gravity[1] ** 2
                                     + (gravity[2] + 1.0) ** 2)
        r_orient_g = W_ORIENT_GAUSS * float(np.exp(
            -((gravity[2] + 1.0) ** 2) / (2.0 * ORIENT_EPS ** 2)))

        height_err = float(d.qpos[2]) - BASE_HEIGHT_TARGET
        r_height = W_HEIGHT * float(np.exp(-(height_err ** 2)))

        r_foot = W_FOOT * float(np.sum(contacts > 0.3))

        near_upright = abs(float(gravity[2]) + 1.0) < STAND_POSE_EPS
        if near_upright:
            pose_err = float(np.sum((leg_q - stand_tgt) ** 2))
            r_stand = W_STAND * float(np.exp(-pose_err))
        else:
            r_stand = 0.0

        r_ang = W_ANG_VEL * float(np.sum(ang_vel ** 2))
        r_torque = W_TORQUE * float(np.sum(d.actuator_force[:12] ** 2))
        r_action = W_ACTION * float(np.sum(action ** 2))
        r_rate = W_ACTION_RATE * float(np.sum((action - self._prev_action) ** 2))
        r_dof_vel = W_DOF_VEL * float(np.sum(leg_dq ** 2))
        over = np.maximum(np.abs(leg_dq) - MAX_JOINT_VEL, 0.0)
        r_max_vel = W_MAX_VEL * float(np.sum(over))

        # Anti-bounce settle once upright-ish (Genesis go2-rl get-up idea).
        lin_vel = d.sensor("lin_vel").data.astype(np.float32)
        if near_upright:
            r_settle = W_SETTLE * float(lin_vel[2] ** 2)
        else:
            r_settle = 0.0

        components = dict(
            orient=r_orient, orient_gauss=r_orient_g, height=r_height,
            foot=r_foot, stand=r_stand, ang_vel=r_ang, torque=r_torque,
            action=r_action, action_rate=r_rate, dof_vel=r_dof_vel,
            max_vel=r_max_vel, settle=r_settle,
        )
        total = float(sum(components.values()))
        if ONLY_POSITIVE_REWARDS:
            total = max(0.0, total)
        components["clipped"] = total
        return total, components

    def randomize_pose(self) -> None:
        """Drop the robot into a fresh random orientation (play / Backspace)."""
        self._spawn_fallen()
        self._prev_action[:] = 0.0
        self._last_last_action[:] = 0.0
        self._step_count = 0
        self._rotated_steps = 0

    def _spawn_fallen(self) -> None:
        if self._rough_terrain and self._hfield_id >= 0:
            n = self._hfield_nrow * self._hfield_ncol
            adr = int(self.model.hfield_adr[self._hfield_id])
            self.model.hfield_data[adr:adr + n] = 0.0

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()

        self.data.qpos[0:2] = self.np_random.uniform(-0.15, 0.15, size=2)
        terrain_z = self._terrain_height_at(
            float(self.data.qpos[0]), float(self.data.qpos[1]))
        self.data.qpos[2] = terrain_z + SPAWN_HEIGHT
        self.data.qpos[3:7] = self._random_unit_quat()
        # Legs start curled; arm stowed.
        curl_qpos = np.empty(12, dtype=np.float32)
        curl_qpos[0::3] = 0.0
        curl_qpos[1::3] = 1.5
        curl_qpos[2::3] = -2.4
        noise = (self.np_random.random(12) - 0.5) * 0.1
        self.data.qpos[7:19] = curl_qpos + noise
        self.data.qpos[19:] = ARM_STOW
        self.data.qvel[:] = 0.0
        self.data.qvel[0:6] = self.np_random.uniform(-0.3, 0.3, size=6)

        self.data.ctrl[:12] = ACT_CURL
        self.data.ctrl[12:] = ARM_STOW
        mujoco.mj_forward(self.model, self.data)
        # Settle briefly so random quat doesn't explode contacts.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Advance stand-pose curriculum from previous episode outcome.
        if self._last_upright:
            self._success_streak += 1
            self.curriculum_level = float(np.clip(
                self.curriculum_level + 0.01, 0.0, 1.0))
        else:
            self._success_streak = 0
            self.curriculum_level = float(np.clip(
                self.curriculum_level - 0.005, 0.0, 1.0))

        if options and options.get("cmd") is not None:
            self.cmd = np.array(options["cmd"], dtype=np.float32)

        self._spawn_fallen()
        self._prev_action[:] = 0.0
        self._last_last_action[:] = 0.0
        self._step_count = 0
        self._rotated_steps = 0
        self._last_upright = False
        self._upright_hold = 0
        return self._get_obs(), {"curriculum_level": self.curriculum_level}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:12] = ACT_CURL + action * ACT_SCALE
        self.data.ctrl[12:] = ARM_STOW
        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        reward, components = self._compute_reward(action)
        self._last_last_action = self._prev_action.copy()
        self._prev_action = action.copy()
        self._step_count += 1

        gravity = self._gravity_vec()
        upright = self._is_upright(gravity)
        self._last_upright = upright
        self._upright_hold = self._upright_hold + 1 if upright else 0

        if gravity[2] > -0.5:
            self._rotated_steps += 1
        else:
            self._rotated_steps = 0

        truncated = self._step_count >= self._max_steps
        terminated = False
        if not self.test_mode:
            # Fail if stuck inverted too long (FR-Net rotate_time threshold).
            if self._rotated_steps >= ROTATED_FAIL_STEPS:
                terminated = True
                components["fail_rotated"] = -1.0
            # Soft success: stayed upright near end of episode → truncate early
            # only in non-test when clearly recovered for a while.
            if upright and self._step_count > int(0.4 * self._max_steps):
                # Don't auto-end; let rewards accumulate standing. Timeout only.
                pass

        # Escape to infinity / under floor.
        z = float(self.data.qpos[2])
        if z < 0.05 or z > 1.5 or not np.isfinite(z):
            terminated = True
            components["fail_bounds"] = -1.0

        info = {
            "reward_components": components,
            "upright": upright,
            "curriculum_level": self.curriculum_level,
            # Sustained upright for UPRIGHT_HOLD_STEPS (~1s), not just "happened
            # to be upright on the literal last tick of a full-length episode"
            # -- the old `upright and truncated` check meant success_rate could
            # only ever register on the rare episode that both ran the full
            # length AND was upright on that exact final frame, staying ~0
            # throughout early training even once the policy started reliably
            # recovering (orient_gauss/stand rewards already climbing by then).
            "is_success": bool(self._upright_hold >= UPRIGHT_HOLD_STEPS),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

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
    env = Go2MujocoRecoveryEnv(randomize_domain=False)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,), obs.shape
    upright_hits = 0
    for i in range(300):
        a = env.action_space.sample() * 0.1
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (OBS_DIM,)
        assert r >= 0.0  # only_positive_rewards
        if info["upright"]:
            upright_hits += 1
        if term or trunc:
            obs, info = env.reset()
    print(f"recovery smoke ok | obs={OBS_DIM} act={ACT_DIM} "
          f"upright_steps={upright_hits} curr={info['curriculum_level']:.2f}")
    env.close()
