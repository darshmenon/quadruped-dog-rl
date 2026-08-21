"""
Keyboard teleop for Go2 agility skills in MuJoCo.

Skills (number keys):
  1 stand   2 crouch   3 sit     4 walk
  5 trot    6 pace     7 bound   8 pronk
  9 jump    0 jump_fwd  - jump_diag

Motion:
  W/S  forward / back     A/D  strafe
  Q/E  yaw                Z/C  raise / lower body
  J/K  more / less jump   R    reset    ESC quit

Usage:
    python3 training/teleop_agility.py
    python3 training/teleop_agility.py --model training/logs/agility/best_model.zip
    python3 training/teleop_agility.py --parkour --model training/logs/parkour/best_model.zip
"""

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules.setdefault("triton", None)

import numpy as np
import mujoco
import mujoco.viewer

from envs.go2_mujoco_agility_env import Go2MujocoAgilityEnv
from intelligence.skills.agility_skills import (
    AgilityCommand,
    Skill,
    command_from_skill,
)

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


SKILL_KEYS = {
    "1": Skill.STAND,
    "2": Skill.CROUCH,
    "3": Skill.SIT,
    "4": Skill.WALK,
    "5": Skill.TROT,
    "6": Skill.PACE,
    "7": Skill.BOUND,
    "8": Skill.PRONK,
    "9": Skill.JUMP,
    "0": Skill.JUMP_FORWARD,
    "-": Skill.JUMP_DIAGONAL,
}


def _get_key_reader():
    import termios
    import tty
    import select

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def read():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore():
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return read, restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--parkour", action="store_true",
        help="use parkour scene with hurdle curriculum resets",
    )
    args = parser.parse_args()

    model = None
    if args.model and HAS_SB3:
        model = PPO.load(args.model)
        print(f"Loaded policy from {args.model}")
    else:
        print("No model — random actions. Pass --model <path> for a trained policy.")

    if args.parkour:
        from envs.go2_mujoco_parkour_env import Go2MujocoParkourEnv
        env = Go2MujocoParkourEnv(
            render_mode=None, randomize_domain=False, use_curriculum=True,
            initial_curriculum_level=0.6,
        )
    else:
        env = Go2MujocoAgilityEnv(
            render_mode=None, randomize_domain=False, use_curriculum=False,
            initial_command=command_from_skill(Skill.STAND),
        )
    mj_model, mj_data = env.model, env.data

    cmd = command_from_skill(Skill.STAND)
    skill_name = Skill.STAND.value
    cmd_lock = threading.Lock()
    do_reset = threading.Event()
    do_quit = threading.Event()
    STEP = 0.1

    def key_thread():
        nonlocal cmd, skill_name
        read_key, restore = _get_key_reader()
        try:
            while not do_quit.is_set():
                k = read_key()
                if k is None:
                    continue
                with cmd_lock:
                    if k in SKILL_KEYS:
                        skill = SKILL_KEYS[k]
                        skill_name = skill.value
                        base = command_from_skill(skill)
                        cmd = AgilityCommand(
                            vx=base.vx if abs(cmd.vx) < 0.05 else cmd.vx,
                            vy=cmd.vy,
                            wz=cmd.wz,
                            height_offset=base.height_offset,
                            jump_height=base.jump_height,
                            gait_freq=base.gait_freq,
                            gait_phase=base.gait_phase,
                            gait_offset=base.gait_offset,
                            gait_bound=base.gait_bound,
                            landing_dx=base.landing_dx,
                            landing_dy=base.landing_dy,
                        )
                    elif k in ("w", "W"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "vx": min(cmd.vx + STEP, 1.5)})
                    elif k in ("s", "S"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "vx": max(cmd.vx - STEP, -1.0)})
                    elif k in ("a", "A"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "vy": min(cmd.vy + STEP, 0.8)})
                    elif k in ("d", "D"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "vy": max(cmd.vy - STEP, -0.8)})
                    elif k in ("q", "Q"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "wz": min(cmd.wz + STEP, 1.2)})
                    elif k in ("e", "E"):
                        cmd = AgilityCommand(**{**cmd.__dict__, "wz": max(cmd.wz - STEP, -1.2)})
                    elif k in ("z", "Z"):
                        cmd = AgilityCommand(**{
                            **cmd.__dict__,
                            "height_offset": min(cmd.height_offset + 0.02, 0.08),
                        })
                    elif k in ("c", "C"):
                        cmd = AgilityCommand(**{
                            **cmd.__dict__,
                            "height_offset": max(cmd.height_offset - 0.02, -0.18),
                        })
                    elif k in ("j", "J"):
                        cmd = AgilityCommand(**{
                            **cmd.__dict__,
                            "jump_height": min(cmd.jump_height + 0.02, 0.28),
                        })
                    elif k in ("k", "K"):
                        cmd = AgilityCommand(**{
                            **cmd.__dict__,
                            "jump_height": max(cmd.jump_height - 0.02, 0.0),
                        })
                    elif k in ("r", "R"):
                        do_reset.set()
                        cmd = command_from_skill(Skill.STAND)
                        skill_name = Skill.STAND.value
                    elif k == "\x1b":
                        do_quit.set()
        finally:
            restore()

    print(
        "\nAgility teleop:\n"
        "  1-9/0/- skills   W/S fwd  A/D strafe  Q/E yaw\n"
        "  Z/C height   J/K jump   R reset   ESC quit\n"
    )
    threading.Thread(target=key_thread, daemon=True).start()

    obs, info = env.reset(seed=0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and not do_quit.is_set():
            if do_reset.is_set():
                obs, info = env.reset()
                do_reset.clear()

            with cmd_lock:
                if not args.parkour:
                    env.set_command(cmd)
                stage = info.get("jump_stage", skill_name)
                print(
                    f"\rskill={stage:14s}  vx={cmd.vx:+.2f}  "
                    f"land=({cmd.landing_dx:+.2f},{cmd.landing_dy:+.2f})  "
                    f"jump={cmd.jump_height:.2f}  ",
                    end="",
                )

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, info = env.step(action)
            if args.parkour:
                # Keep displayed cmd in sync with env's curriculum sample
                cmd = AgilityCommand.from_array(info.get("command", cmd.as_array()))
            viewer.sync()
            if terminated or truncated:
                obs, info = env.reset()

    do_quit.set()
    env.close()
    print("\nQuitting.")


if __name__ == "__main__":
    main()
