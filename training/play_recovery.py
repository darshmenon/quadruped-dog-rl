"""
Play / smoke-eval a Go2 fall-recovery policy in MuJoCo.

Usage:
  python3 training/play_recovery.py --model training/logs/recovery/best_model.zip
  python3 training/play_recovery.py --model best_model.zip --no-display --episodes 5
  python3 training/play_recovery.py   # random actions smoke (no model)

Controls (GUI):
  R / Backspace — new random fallen pose
  ESC           — quit
"""

import argparse
import os
import sys
import time

sys.modules.setdefault("triton", None)
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import mujoco
import numpy as np

from envs.go2_mujoco_recovery_env import (
    Go2MujocoRecoveryEnv, CTRL_DT, OBS_DIM, ACT_DIM)
from envs.obs_history import ObsHistoryWrapper


def _render_frame(renderer, data):
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -25.0
    if body_id >= 0:
        cam.lookat[:] = data.xpos[body_id]
    renderer.update_scene(data, camera=cam)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def _hud(frame, upright, reward, episode, step, gz, z, fps):
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"recovery  ep={episode}  step={step}  fps={fps:.0f}",
        f"upright={upright}  gz={gz:+.2f}  z={z:.2f}  r={reward:+.2f}",
        "R/Backspace=new fall   ESC=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 22), font, 0.55,
                    (0, 255, 0) if upright else (0, 180, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--obs-history", type=int, default=1)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    env = Go2MujocoRecoveryEnv(
        randomize_domain=False, test_mode=True)
    if args.obs_history > 1:
        env = ObsHistoryWrapper(env, history_len=args.obs_history)

    model = None
    vecnorm = None
    if args.model:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        def _mk():
            e = Go2MujocoRecoveryEnv(randomize_domain=False, test_mode=True)
            if args.obs_history > 1:
                e = ObsHistoryWrapper(e, history_len=args.obs_history)
            return Monitor(e)

        venv = DummyVecEnv([_mk])
        if args.vecnorm and os.path.exists(args.vecnorm):
            vecnorm = VecNormalize.load(args.vecnorm, venv)
            vecnorm.training = False
            vecnorm.norm_reward = False
        else:
            # Try sibling vecnorm_final.pkl
            guess = os.path.join(os.path.dirname(args.model), "vecnorm_final.pkl")
            if os.path.exists(guess):
                vecnorm = VecNormalize.load(guess, venv)
                vecnorm.training = False
                vecnorm.norm_reward = False
                print(f"Loaded {guess}")
            else:
                vecnorm = VecNormalize(venv, norm_obs=False, norm_reward=False,
                                       training=False)
        model = PPO.load(args.model)
        env = vecnorm.venv.envs[0] if hasattr(vecnorm, "venv") else env

    renderer = None
    if not args.no_display:
        # Unwrap to raw mujoco env for rendering.
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        renderer = mujoco.Renderer(raw.model, height=480, width=640)

    upright_eps = 0
    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        # Reach underlying recovery env for randomize / sensors.
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        ep_ret = 0.0
        saw_upright = False
        t0 = time.time()
        steps = 0

        for steps in range(args.max_steps):
            if model is None:
                action = env.action_space.sample() * 0.05
            else:
                if vecnorm is not None:
                    obs_in = vecnorm.normalize_obs(obs.reshape(1, -1))[0]
                else:
                    obs_in = obs
                action, _ = model.predict(obs_in, deterministic=args.deterministic)

            obs, reward, term, trunc, info = env.step(action)
            ep_ret += float(reward)
            upright = bool(info.get("upright", False))
            saw_upright = saw_upright or upright

            if renderer is not None:
                frame = _render_frame(renderer, raw.data)
                gz = float(raw._gravity_vec()[2])
                z = float(raw.data.qpos[2])
                fps = (steps + 1) / max(time.time() - t0, 1e-6)
                frame = _hud(frame, upright, reward, ep, steps, gz, z, fps)
                cv2.imshow("go2_recovery", frame)
                key = cv2.waitKey(max(1, int(CTRL_DT * 1000))) & 0xFF
                if key in (27, ord("q")):
                    args.episodes = ep + 1
                    term = True
                    trunc = True
                elif key in (ord("r"), ord("R"), 8):  # Backspace=8
                    raw.randomize_pose()
                    obs = raw._get_obs()
                    if args.obs_history > 1 and hasattr(env, "reset"):
                        # refresh history buffer cheaply
                        obs, _ = env.reset()
                    continue

            if term or trunc:
                break

        returns.append(ep_ret)
        upright_eps += int(saw_upright)
        print(f"ep {ep}: return={ep_ret:.1f} steps={steps+1} "
              f"saw_upright={saw_upright}")

    print(f"mean_return={np.mean(returns):.1f} "
          f"upright_rate={upright_eps / max(args.episodes, 1):.2f} "
          f"obs={OBS_DIM} act={ACT_DIM}")

    if renderer is not None:
        cv2.destroyAllWindows()
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
