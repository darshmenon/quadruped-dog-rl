"""MuJoCo-based Gymnasium environment for Unitree Go2 locomotion training."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_scene.xml")

# Arm+gripper stow pose, matches scripts/make_go2_stand.py STANDING_POSE /
# intelligence/manipulation/arm_reach_node.py STOW_POSE (gripper closed).
ARM_STOW = [0.0, 1.4, 0.8, 0.3, 0.0, 0.0, 0.0]  # base, lower_arm, upper_arm, wrist1, wrist2, L/R finger

# qpos/qvel order follows go2_scene.xml's body tree (depth-first): the 4 legs
# leg-by-leg, then the arm+gripper chain appended last (see that file's
# comment on why it must stay last).
DEFAULT_QPOS = np.array([
    0.1,   # FL_hip
    0.8,   # FL_thigh
    -1.5,  # FL_calf
    -0.1,  # FR_hip
    0.8,   # FR_thigh
    -1.5,  # FR_calf
    0.1,   # RL_hip
    1.0,   # RL_thigh
    -1.5,  # RL_calf
    -0.1,  # RR_hip
    1.0,   # RR_thigh
    -1.5,  # RR_calf
] + ARM_STOW, dtype=np.float32)

# Actuator order (independent of qpos order -- set by <actuator> declaration
# order in go2_scene.xml): [FL_hip, FR_hip, RL_hip, RR_hip,
#                            FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#                            FL_calf, FR_calf, RL_calf, RR_calf,
#                            base, lower_arm, upper_arm, wrist1, wrist2, L/R finger]
# The arm+gripper segment happens to match DEFAULT_QPOS's order too, since
# (unlike the 4 parallel legs) it's a single serial chain declared once.
ACT_DEFAULT = np.array([
    0.1, -0.1,  0.1, -0.1,
    0.8,  0.8,  1.0,  1.0,
   -1.5, -1.5, -1.5, -1.5,
] + ARM_STOW, dtype=np.float32)

# 3 ang_vel + 3 gravity + 3 cmd + 19 dof_pos + 19 dof_vel + 19 prev_action
# + 4 contacts + 3 ee_pos + 3 reach_target
OBS_DIM = 76
# When gait_conditioned=True, append: 5 gait params + 4 foot clock sins
# (freq, phase, offset, bound, footswing_height, clock_FL/FR/RL/RR)
GAIT_OBS_DIM = 9
ACT_DIM = 19
ACT_SCALE = 0.25

EPISODE_LEN_S = 20.0
SIM_DT = 0.005
CTRL_DECIMATION = 4   # policy at 50 Hz, sim at 200 Hz
CTRL_DT = SIM_DT * CTRL_DECIMATION

# Named gait presets: (phase, offset, bound). Duration is fixed at 0.5
# (half the cycle in stance). Trotting is the default curriculum sample.
GAIT_PRESETS = {
    "trotting": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.0, 0.5),
    "pacing":   (0.0, 0.5, 0.0),
    "pronking": (0.0, 0.0, 0.0),
}
GAIT_FREQ_RANGE = (1.5, 3.5)       # Hz
GAIT_FOOTSWING_RANGE = (0.04, 0.18)  # metres, command only (clearance cue)
GAIT_CONTACT_WEIGHT = 0.25
GAIT_CONTACT_KAPPA = 0.08          # soft-step width for desired contact

# Step-quality terms (air-time encourages real strides; slip penalizes
# sliding while planted). Only score air-time when a nontrivial velocity
# command is active so standing still is not rewarded for "long stance".
AIR_TIME_TARGET_S = 0.25
AIR_TIME_WEIGHT = 0.35
FEET_SLIP_WEIGHT = 0.12
CONTACT_FORCE_THRESH = 0.3  # after /50 clip, ~15 N raw touch

TARGET_HEIGHT = 0.27  # nominal base height above ground while standing

ALIVE_BONUS  = 0.3    # per-step credit for still standing, so ending an
                       # episode early is never a shortcut to avoid penalties
FALL_PENALTY = -8.0    # one-time hit applied on the step that trips termination

# arm_base's pos= in go2_scene.xml, i.e. where the arm mounts relative to the
# "base" body -- reach targets are sampled around this point, in the same
# frame as the ee_pos sensor.
ARM_MOUNT_POS = np.array([0.08, 0.0, 0.057], dtype=np.float32)

REACH_MIN_RADIUS = 0.12   # closest a target is ever placed from the mount point
REACH_MAX_RADIUS_EASY = 0.22  # max radius at curriculum_level=0
REACH_MAX_RADIUS_HARD = 0.42  # max radius at curriculum_level=1 (< arm_ik.py's
                               # ~0.51 structural max, so targets stay solvable
                               # across the yaw/elevation range sampled below)
REACH_SUCCESS_DIST = 0.05  # fingertip-to-target distance counted as "reached"
REACH_SIGMA = 0.12         # width of the reach-distance reward kernel
REACH_WEIGHT = 0.8

# The ARM_STOW pose (see DEFAULT_QPOS) tucks the arm down and back so it
# doesn't interfere with walking -- its fingertip sits ~0.4m from
# ARM_MOUNT_POS, roughly opposite the direction targets are sampled in, so a
# typical starting reach_dist is ~0.5-0.7m. REACH_SIGMA=0.12 alone gives
# exp(-(d/sigma)^2) and its gradient both underflow to ~0 anywhere past
# ~0.3m -- confirmed by hand FK on the stow pose matching the observed
# frozen fingertip exactly, and by every recorded "best" checkpoint having
# reach reward ~1e-8 the whole episode. The policy was never once rewarded
# for moving the arm toward the target; this dense term gives a real
# (non-vanishing) gradient across the actual starting-distance range, with
# the sigma kernel above still providing the fine-precision signal near the
# goal.
REACH_DENSE_WEIGHT = 0.3
REACH_DENSE_NORM = 0.8    # roughly the max plausible stow-to-target distance

# Two training runs both saw eval reward peak (~460) with curriculum_level
# around 0.8-0.85 (max cmd speed ~1.0 m/s, reach radius ~0.37m) and then
# degrade as the episode-length-only success signal kept pushing curriculum_
# level toward 1.0 -- by ~0.99, reach reward had collapsed to ~0 (arm no
# longer tracking targets at all) and eval reward had dropped to ~140. The
# success metric (did the episode survive) doesn't require the walk+reach
# task to still be going well, so it kept escalating difficulty past the
# point the policy could actually hold both. Capping below 1.0 keeps
# training in the region both runs demonstrated actually works.
MAX_CURRICULUM_LEVEL = 0.85
REACH_SUCCESS_BONUS = 0.5


class Go2MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cmd=(0.5, 0.0, 0.0), render_mode=None,
                 randomize_domain=True, use_curriculum=True,
                 initial_curriculum_level=0.0, reach_target=None,
                 gait_conditioned=False, gait_name="trotting"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.cmd = np.array(cmd, dtype=np.float32)
        self.render_mode = render_mode
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        self.gait_conditioned = bool(gait_conditioned)
        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._max_steps = int(EPISODE_LEN_S / (SIM_DT * CTRL_DECIMATION))
        self._last_episode_steps = self._max_steps

        self.curriculum_level = float(initial_curriculum_level)
        # Mirrors `cmd`: a fixed reach target for non-curriculum use (e.g.
        # play_policy.py), resampled every reset when use_curriculum=True
        # (see reset()). Default is a modest forward reach, not a random
        # sample, so eval runs are reproducible unless the caller asks
        # otherwise.
        self.reach_target = (
            np.array(reach_target, dtype=np.float32) if reach_target is not None
            else ARM_MOUNT_POS + np.array([0.3, 0.0, 0.0], dtype=np.float32)
        )

        # Gait command: [freq_hz, phase, offset, bound, footswing_m].
        # Clocks and desired contacts are derived each step from these.
        self._gait_index = 0.0
        self._clock_inputs = np.zeros(4, dtype=np.float32)
        self._desired_contact = np.ones(4, dtype=np.float32)
        self.gait_cmd = self._default_gait_cmd(gait_name)

        # cache original model params for domain randomization
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_mass = float(self.model.body_mass[self._base_body_id])
        self._base_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._base_gainprm = self.model.actuator_gainprm[:, 0].copy()
        self._base_biasprm1 = self.model.actuator_biasprm[:, 1].copy()

        # Foot bodies for air-time / slip rewards (FL, FR, RL, RR).
        self._foot_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ], dtype=np.int32)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._foot_vel_buf = np.zeros(6, dtype=np.float64)

        obs_dim = OBS_DIM + (GAIT_OBS_DIM if self.gait_conditioned else 0)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
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

    def _foot_lin_vels(self) -> np.ndarray:
        """World-frame linear velocity of each foot body, shape (4, 3)."""
        out = np.empty((4, 3), dtype=np.float32)
        for i, bid in enumerate(self._foot_body_ids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                int(bid), self._foot_vel_buf, 0)
            out[i] = self._foot_vel_buf[:3]
        return out

    def _air_time_and_slip(self, contacts: np.ndarray):
        """Update air-time clocks; return (air_time_reward, slip_penalty)."""
        in_contact = contacts > CONTACT_FORCE_THRESH
        contact_filt = np.logical_or(in_contact, self._last_contacts)
        first_contact = (self._feet_air_time > 0.0) & contact_filt

        self._feet_air_time += CTRL_DT
        air = 0.0
        cmd_speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        if cmd_speed > 0.1 and np.any(first_contact):
            # Reward strides that stay aloft near AIR_TIME_TARGET_S.
            air = AIR_TIME_WEIGHT * float(np.sum(
                (self._feet_air_time - AIR_TIME_TARGET_S) * first_contact))

        self._feet_air_time = np.where(contact_filt, 0.0, self._feet_air_time)
        self._last_contacts = in_contact

        # Penalize horizontal foot velocity while planted.
        foot_vel = self._foot_lin_vels()
        slip_speeds = np.linalg.norm(foot_vel[:, :2], axis=1)
        slip = -FEET_SLIP_WEIGHT * float(np.sum(slip_speeds * in_contact))
        return air, slip

    def _default_gait_cmd(self, gait_name: str) -> np.ndarray:
        phase, offset, bound = GAIT_PRESETS.get(gait_name, GAIT_PRESETS["trotting"])
        freq = 0.5 * (GAIT_FREQ_RANGE[0] + GAIT_FREQ_RANGE[1])
        swing = 0.5 * (GAIT_FOOTSWING_RANGE[0] + GAIT_FOOTSWING_RANGE[1])
        return np.array([freq, phase, offset, bound, swing], dtype=np.float32)

    def _sample_gait_cmd(self) -> np.ndarray:
        """Sample a gait style + frequency. Early curriculum stays on
        trotting; higher levels mix in bounding/pacing/pronking."""
        rng = self.np_random
        if self.curriculum_level < 0.35:
            name = "trotting"
        else:
            names = list(GAIT_PRESETS.keys())
            # Bias toward trotting even late in curriculum.
            weights = [0.55, 0.2, 0.15, 0.1]
            name = names[int(rng.choice(len(names), p=weights))]
        phase, offset, bound = GAIT_PRESETS[name]
        freq_lo, freq_hi = GAIT_FREQ_RANGE
        freq = float(rng.uniform(freq_lo, freq_lo + (freq_hi - freq_lo) * max(
            0.4, self.curriculum_level)))
        swing_lo, swing_hi = GAIT_FOOTSWING_RANGE
        swing = float(rng.uniform(swing_lo, swing_lo + (swing_hi - swing_lo) * max(
            0.3, self.curriculum_level)))
        return np.array([freq, phase, offset, bound, swing], dtype=np.float32)

    def _step_gait(self) -> None:
        """Advance gait phase and refresh clock / desired-contact targets."""
        if not self.gait_conditioned:
            return
        freq, phase, offset, bound, _swing = self.gait_cmd
        duration = 0.5
        self._gait_index = (self._gait_index + CTRL_DT * float(freq)) % 1.0
        # Per-foot phase offsets: FL, FR, RL, RR
        foot_phases = np.array([
            self._gait_index + phase + offset + bound,
            self._gait_index + offset,
            self._gait_index + bound,
            self._gait_index + phase,
        ], dtype=np.float64) % 1.0

        clocks = np.empty(4, dtype=np.float32)
        desired = np.empty(4, dtype=np.float32)
        kappa = GAIT_CONTACT_KAPPA
        for i, p in enumerate(foot_phases):
            # Warp so stance occupies [0, 0.5) and swing [0.5, 1).
            if p < duration:
                warped = p * (0.5 / duration)
            else:
                warped = 0.5 + (p - duration) * (0.5 / (1.0 - duration))
            clocks[i] = np.sin(2.0 * np.pi * warped)
            # Soft stance indicator: high in first half of warped cycle.
            desired[i] = 1.0 / (1.0 + np.exp((warped - 0.5) / kappa))
        self._clock_inputs = clocks
        self._desired_contact = desired

    def _get_obs(self) -> np.ndarray:
        d = self.data
        ang_vel   = d.sensor("ang_vel").data.astype(np.float32) * 0.25
        gravity   = self._gravity_vec()
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        dof_pos   = (d.qpos[7:].astype(np.float32) - DEFAULT_QPOS)
        dof_vel   = d.qvel[6:].astype(np.float32) * 0.05
        contacts  = self._get_contacts()
        # Gripper fingertip position and its current target, both relative
        # to the base body (see go2_scene.xml's ee_pos sensor comment) --
        # given as two absolute points rather than a precomputed error
        # vector, matching how dof_pos/cmd are split into achieved vs.
        # desired elsewhere in this observation.
        ee_pos    = d.sensor("ee_pos").data.astype(np.float32)
        parts = [
            ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action,
            contacts, ee_pos, self.reach_target,
        ]
        if self.gait_conditioned:
            # Normalize freq into roughly [-1, 1]-ish range for the MLP.
            freq_n = (self.gait_cmd[0] - GAIT_FREQ_RANGE[0]) / (
                GAIT_FREQ_RANGE[1] - GAIT_FREQ_RANGE[0] + 1e-6)
            swing_n = self.gait_cmd[4] / GAIT_FOOTSWING_RANGE[1]
            gait_feat = np.array(
                [freq_n, self.gait_cmd[1], self.gait_cmd[2], self.gait_cmd[3],
                 swing_n],
                dtype=np.float32)
            parts.extend([gait_feat, self._clock_inputs])
        return np.concatenate(parts)

    def _compute_reward(self, action: np.ndarray):
        d = self.data
        lin_vel  = d.sensor("lin_vel").data.astype(np.float32)
        ang_vel  = d.sensor("ang_vel").data.astype(np.float32)
        gravity  = self._gravity_vec()

        r_lin    = 2.0 * float(np.exp(
            -((lin_vel[0] - self.cmd[0])**2 + (lin_vel[1] - self.cmd[1])**2) / 0.1))
        r_ang    = 0.5 * float(np.exp(-((ang_vel[2] - self.cmd[2])**2) / 0.25))
        r_z      = -2.0  * float(lin_vel[2]**2)
        r_height = -1.0  * (float(d.qpos[2]) - TARGET_HEIGHT)**2
        r_orient = -0.5  * float(gravity[0]**2 + gravity[1]**2)
        r_torque = -2e-4 * float(np.sum(d.actuator_force**2))
        r_smooth = -5e-3 * float(np.sum((action - self._prev_action)**2))

        contacts  = self._get_contacts()
        r_contact = 0.15 * min(float(np.sum(contacts > 0.3)) / 2.0, 1.0)

        ee_pos = d.sensor("ee_pos").data.astype(np.float32)
        reach_dist = float(np.linalg.norm(self.reach_target - ee_pos))
        r_reach = REACH_WEIGHT * float(np.exp(-(reach_dist ** 2) / (REACH_SIGMA ** 2)))
        r_reach_dense = REACH_DENSE_WEIGHT * max(0.0, 1.0 - reach_dist / REACH_DENSE_NORM)
        r_reach_bonus = REACH_SUCCESS_BONUS if reach_dist < REACH_SUCCESS_DIST else 0.0

        # Explicit stall penalty: standing still while a real command is
        # active must never out-earn walking, no matter how forgiving the
        # tracking kernel above is (previously the policy converged to
        # standing still — see README "Known issue").
        cmd_speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        actual_speed = float(np.hypot(lin_vel[0], lin_vel[1]))
        r_stall = -0.6 if (cmd_speed > 0.15 and actual_speed < 0.3 * cmd_speed) else 0.0

        r_air, r_slip = self._air_time_and_slip(contacts)

        components = dict(
            lin=r_lin, ang=r_ang, vz=r_z, height=r_height,
            orient=r_orient, torque=r_torque, smooth=r_smooth, contact=r_contact,
            stall=r_stall, alive=ALIVE_BONUS,
            reach=r_reach, reach_dense=r_reach_dense, reach_bonus=r_reach_bonus,
            air_time=r_air, slip=r_slip,
        )

        if self.gait_conditioned:
            # Match measured contacts to the commanded gait's stance/swing.
            # contacts are already in [0,1]; desired_contact is soft [0,1].
            match = 1.0 - np.abs(contacts - self._desired_contact)
            components["gait_contact"] = GAIT_CONTACT_WEIGHT * float(np.mean(match))

        return float(sum(components.values())), components

    def _sample_cmd(self) -> np.ndarray:
        max_vx = 0.15 + 1.05 * self.curriculum_level
        vx = float(self.np_random.uniform(-0.1, max_vx))
        vy = float(self.np_random.uniform(-0.2, 0.2)) * self.curriculum_level
        wz = float(self.np_random.uniform(-0.5, 0.5)) * self.curriculum_level
        return np.array([vx, vy, wz], dtype=np.float32)

    def _sample_reach_target(self) -> np.ndarray:
        """Sample a reach target around ARM_MOUNT_POS, in the base body
        frame. Radius range widens with curriculum_level (same level that
        drives _sample_cmd's walking speed), so training starts with the
        arm holding a near, easy point while standing close to still, and
        only asks for farther reaches once the body is also walking faster
        -- "standing and reaching" progressing to "reaching while walking",
        both gated by the one curriculum_level rather than two independent
        schedules that could drift out of sync.
        """
        max_radius = REACH_MAX_RADIUS_EASY + (
            REACH_MAX_RADIUS_HARD - REACH_MAX_RADIUS_EASY) * self.curriculum_level
        radius = float(self.np_random.uniform(REACH_MIN_RADIUS, max_radius))
        # yaw within base_joint's own +/-90deg limit (arm.urdf.xacro), with
        # margin so the shoulder/elbow aren't also pinned at their limits
        # trying to hit the same point; elevation modestly above/below the
        # mount plane.
        yaw = float(self.np_random.uniform(-1.1, 1.1))
        pitch = float(self.np_random.uniform(-0.5, 0.6))
        direction = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float32)
        return ARM_MOUNT_POS + radius * direction

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
        if z < 0.15 or z > 0.8:
            return True
        # 1 - 2*(w^2 + z_q^2) == 2*(x^2 + y^2) - 1 == -cos(tilt) for a unit
        # quaternion, so the old `> 0.5` threshold only tripped past 120
        # degrees of pitch/roll - permissive enough that the policy could
        # rear up onto its hind legs and sustain a ~70-80 degree "wheelie"
        # for many steps before actually falling. -0.5 corresponds to ~60
        # degrees, a sane cutoff for a quadruped.
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > -0.5)

    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_curriculum:
            success = self._last_episode_steps >= 0.75 * self._max_steps
            self.curriculum_level = float(np.clip(
                self.curriculum_level + (0.005 if success else -0.002),
                0.0, MAX_CURRICULUM_LEVEL))
            self.cmd = self._sample_cmd()
            self.reach_target = self._sample_reach_target()
            if self.gait_conditioned:
                self.gait_cmd = self._sample_gait_cmd()

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()

        self.data.qpos[2]   = 0.42
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # Domain-randomize the leg joints only: the arm+gripper stow pose
        # (DEFAULT_QPOS[12:]) starts exact, since the same +/-0.05 rad noise
        # scale would swing the finger joints (0.025m full range) past their
        # limits.
        self.data.qpos[7:19] = DEFAULT_QPOS[:12] + (self.np_random.random(12) - 0.5) * 0.1
        self.data.qpos[19:]  = DEFAULT_QPOS[12:]
        self.data.ctrl[:]   = self._act_default
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._last_episode_steps = 0
        self._gait_index = 0.0
        self._feet_air_time[:] = 0.0
        self._last_contacts[:] = False
        self._step_gait()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = self._act_default + action * ACT_SCALE
        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self._step_gait()
        reward, components = self._compute_reward(action)
        self._prev_action = action.copy()
        self._step_count += 1
        self._last_episode_steps = self._step_count

        obs = self._get_obs()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self._max_steps

        if terminated:
            reward += FALL_PENALTY
            components["fall"] = FALL_PENALTY

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"reward_components": components}

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
    for gait in (False, True):
        env = Go2MujocoEnv(render_mode=None, randomize_domain=False,
                           use_curriculum=False, gait_conditioned=gait)
        obs, _ = env.reset(seed=0)
        expected = OBS_DIM + (GAIT_OBS_DIM if gait else 0)
        print(f"gait_conditioned={gait}: obs shape {obs.shape}")
        assert obs.shape == (expected,), f"expected {expected}, got {obs.shape[0]}"
        saw_air = False
        for _ in range(200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            comps = info["reward_components"]
            assert "air_time" in comps and "slip" in comps
            if gait:
                assert "gait_contact" in comps
            if comps["air_time"] != 0.0:
                saw_air = True
            if term or trunc:
                obs, _ = env.reset()
        print(f"gait_conditioned={gait}: smoke-test passed (air_nonzero={saw_air})")
