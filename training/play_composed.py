"""Compose walk + fall-recovery policies (skill FSM).

States: WALK → RECOVER (on fall) → STAND_HOLD → WALK.

Walk policy: 76-D (or 85 w/ gait) → 19-DOF. Recovery: 45-D → 12-DOF legs.
Physics stays in one Go2MujocoEnv; recovery actions only write leg ctrls.

Usage:
  python3 training/play_composed.py \\
    --walk training/logs/mujoco/best_model.zip \\
    --recovery training/logs/recovery/best_model.zip \\
    --no-display --episodes 3

  # Random recovery actions if --recovery omitted (FSM still exercises)
  python3 training/play_composed.py --walk training/logs/mujoco/best_model.zip \\
    --no-display --max-steps 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.modules.setdefault("triton", None)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from envs.go2_mujoco_env import (
    Go2MujocoEnv, ACT_DEFAULT, ACT_SCALE, ARM_STOW, CTRL_DT)
from envs.go2_mujoco_recovery_env import ACT_CURL, ACT_DIM as REC_ACT, _QPOS_TO_ACT

WALK, RECOVER, STAND_HOLD = "WALK", "RECOVER", "STAND_HOLD"
STAND_HOLD_STEPS = 75      # ~1.5 s at 50 Hz
FALL_GZ = -0.45            # gravity_z above this ⇒ fallen / tipping
FALL_Z = 0.18
UPRIGHT_GZ = -0.75
UPRIGHT_Z = 0.28
UPRIGHT_NEED = 15          # consecutive upright steps before STAND_HOLD


def _gravity(env: Go2MujocoEnv) -> np.ndarray:
    return env._gravity_vec()


def _is_fallen(env: Go2MujocoEnv) -> bool:
    g = _gravity(env)
    z = float(env.data.qpos[2])
    return bool(g[2] > FALL_GZ or z < FALL_Z)


def _is_upright(env: Go2MujocoEnv) -> bool:
    g = _gravity(env)
    z = float(env.data.qpos[2])
    contacts = env._get_contacts()
    feet = float(np.sum(contacts > 0.3))
    return bool(g[2] < UPRIGHT_GZ and z > UPRIGHT_Z and feet >= 2.5)


def _recovery_obs(env: Go2MujocoEnv, prev_rec_action: np.ndarray,
                  cmd: np.ndarray) -> np.ndarray:
    """Build FR-Net-style 45-D obs from the walk env's sensors."""
    ang_vel = env.data.sensor("ang_vel").data.astype(np.float32) * 0.25
    gravity = _gravity(env)
    cmd_scaled = cmd.astype(np.float32) * np.array([2.0, 2.0, 0.25], dtype=np.float32)
    leg_q = env.data.qpos[7:19].astype(np.float32)[_QPOS_TO_ACT]
    leg_dq = env.data.qvel[6:18].astype(np.float32)[_QPOS_TO_ACT]
    dof_pos = leg_q - ACT_CURL
    dof_vel = leg_dq * 0.05
    return np.concatenate(
        [ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, prev_rec_action])


def _apply_recovery_action(env: Go2MujocoEnv, action12: np.ndarray) -> None:
    action12 = np.clip(action12, -1.0, 1.0).astype(np.float32)
    env.data.ctrl[:12] = ACT_CURL + action12 * ACT_SCALE
    env.data.ctrl[12:] = ARM_STOW
    for _ in range(4):
        import mujoco
        mujoco.mj_step(env.model, env.data)


def _apply_walk_action(env: Go2MujocoEnv, action19: np.ndarray) -> None:
    action19 = np.clip(action19, -1.0, 1.0).astype(np.float32)
    env.data.ctrl[:] = ACT_DEFAULT + action19 * ACT_SCALE
    for _ in range(4):
        import mujoco
        mujoco.mj_step(env.model, env.data)


def main():
    parser = argparse.ArgumentParser(description="Walk ↔ recovery composed play")
    parser.add_argument("--walk", type=str, default=None)
    parser.add_argument("--recovery", type=str, default=None)
    parser.add_argument("--walk-vecnorm", type=str, default=None)
    parser.add_argument("--recovery-vecnorm", type=str, default=None)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--force-fall-step", type=int, default=80,
                        help="inject a tip-over at this step (0=disable) to demo recovery")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--gait", action="store_true")
    args = parser.parse_args()

    env = Go2MujocoEnv(
        cmd=tuple(args.cmd), render_mode=None,
        randomize_domain=False, use_curriculum=False,
        gait_conditioned=args.gait, push_robots=False)
    cmd = np.array(args.cmd, dtype=np.float32)

    walk_model = rec_model = None
    walk_vn = rec_vn = None
    if args.walk:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        walk_model = PPO.load(args.walk)
        guess = args.walk_vecnorm or os.path.join(
            os.path.dirname(args.walk), "vecnorm_final.pkl")
        if guess and os.path.exists(guess):
            walk_vn = VecNormalize.load(
                guess, DummyVecEnv([lambda: Go2MujocoEnv(
                    cmd=tuple(args.cmd), randomize_domain=False,
                    use_curriculum=False, gait_conditioned=args.gait,
                    push_robots=False)]))
            walk_vn.training = False
            walk_vn.norm_reward = False
            print(f"walk VecNormalize: {guess}")
    if args.recovery:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from envs.go2_mujoco_recovery_env import Go2MujocoRecoveryEnv
        rec_model = PPO.load(args.recovery)
        guess = args.recovery_vecnorm or os.path.join(
            os.path.dirname(args.recovery), "vecnorm_final.pkl")
        if guess and os.path.exists(guess):
            rec_vn = VecNormalize.load(
                guess, DummyVecEnv([lambda: Go2MujocoRecoveryEnv(
                    randomize_domain=False, test_mode=True)]))
            rec_vn.training = False
            rec_vn.norm_reward = False
            print(f"recovery VecNormalize: {guess}")

    renderer = None
    if not args.no_display:
        import cv2
        import mujoco
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        state = WALK
        hold_left = 0
        upright_streak = 0
        prev_rec = np.zeros(REC_ACT, dtype=np.float32)
        transitions = []
        t0 = time.time()

        for step in range(args.max_steps):
            # Optional demo tip-over.
            if args.force_fall_step and step == args.force_fall_step and state == WALK:
                env.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # 180° pitch-ish
                env.data.qpos[2] = 0.25
                import mujoco
                mujoco.mj_forward(env.model, env.data)
                state = RECOVER
                upright_streak = 0
                transitions.append((step, "force_fall→RECOVER"))

            if state == WALK:
                if _is_fallen(env):
                    state = RECOVER
                    upright_streak = 0
                    transitions.append((step, "WALK→RECOVER"))
                else:
                    if walk_model is None:
                        action = env.action_space.sample() * 0.05
                    else:
                        obs_in = obs
                        if walk_vn is not None:
                            obs_in = walk_vn.normalize_obs(obs.reshape(1, -1))[0]
                        action, _ = walk_model.predict(obs_in, deterministic=True)
                    _apply_walk_action(env, action)
                    env._prev_action = np.asarray(action, dtype=np.float32)
                    env._step_gait()
                    obs = env._get_obs()

            elif state == RECOVER:
                rec_obs = _recovery_obs(env, prev_rec, np.zeros(3, dtype=np.float32))
                if rec_model is None:
                    action12 = env.np_random.uniform(-0.2, 0.2, size=REC_ACT).astype(
                        np.float32)
                else:
                    obs_in = rec_obs
                    if rec_vn is not None:
                        obs_in = rec_vn.normalize_obs(rec_obs.reshape(1, -1))[0]
                    action12, _ = rec_model.predict(obs_in, deterministic=True)
                    action12 = np.asarray(action12, dtype=np.float32)
                _apply_recovery_action(env, action12)
                prev_rec = action12
                if _is_upright(env):
                    upright_streak += 1
                else:
                    upright_streak = 0
                if upright_streak >= UPRIGHT_NEED:
                    state = STAND_HOLD
                    hold_left = STAND_HOLD_STEPS
                    transitions.append((step, "RECOVER→STAND_HOLD"))
                obs = env._get_obs()

            else:  # STAND_HOLD
                env.cmd = np.zeros(3, dtype=np.float32)
                hold_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                _apply_walk_action(env, hold_action)
                env._prev_action = hold_action
                hold_left -= 1
                if hold_left <= 0:
                    state = WALK
                    env.cmd = cmd.copy()
                    obs = env._get_obs()
                    transitions.append((step, "STAND_HOLD→WALK"))
                else:
                    obs = env._get_obs()

            if renderer is not None:
                import cv2
                import mujoco
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance = 2.2
                cam.elevation = -20
                cam.lookat[:] = env.data.xpos[env._base_body_id]
                renderer.update_scene(env.data, camera=cam)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                g = _gravity(env)
                cv2.putText(
                    frame,
                    f"{state}  gz={g[2]:+.2f} z={env.data.qpos[2]:.2f} step={step}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if state == WALK else (0, 180, 255), 1)
                cv2.imshow("composed", frame)
                if cv2.waitKey(max(1, int(CTRL_DT * 1000))) & 0xFF in (27, ord("q")):
                    args.episodes = ep + 1
                    break

            if env._is_terminated() and state == WALK:
                # Fall detector in FSM should have caught it; if walk termination
                # fires first, switch to recovery instead of ending episode.
                state = RECOVER
                upright_streak = 0
                transitions.append((step, "term→RECOVER"))

        dt = time.time() - t0
        print(f"ep {ep}: steps={step+1}  {dt:.1f}s  transitions={transitions}")

    env.close()
    if renderer is not None:
        import cv2
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
