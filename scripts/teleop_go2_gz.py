#!/usr/bin/env python3
"""Keyboard teleop for the Go2 Gazebo Harmonic joint controllers."""

import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.gait.gait_scheduler import Gait, GaitScheduler, GAITS


JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

HIP_DEFAULT = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)
THIGH_DEFAULT = np.array([0.8, 0.8, 1.0, 1.0], dtype=np.float64)
CALF_DEFAULT = np.array([-1.5, -1.5, -1.5, -1.5], dtype=np.float64)

L1 = 0.213
L2 = 0.213
PZ_NOM = 0.306
# Per-leg sagittal stand bias from FK(thigh/calf defaults). Centering the
# stride on 0 yanked rear feet ~6cm back at gait start and yawed the body.
PX_NOM = np.array([0.016, 0.016, 0.077, 0.077], dtype=np.float64)
STRIDE_LEN = 0.10
YAW_STRIDE_GAIN = 0.08

VX_MAX = 0.35
VX_MIN = -0.20
VY_MAX = 0.15
WZ_MAX = 0.40
CMD_STEP = 0.03
CTRL_DT = 0.02

STEP_HEIGHT = {
    Gait.STAND: 0.00,
    Gait.WALK: 0.05,
    Gait.TROT: 0.06,
    Gait.CANTER: 0.07,
    Gait.BOUND: 0.08,
    Gait.PRONK: 0.10,
}
STRIDE_SCALE = {
    Gait.STAND: 0.0,
    Gait.WALK: 0.7,
    Gait.TROT: 1.0,
    Gait.CANTER: 1.2,
    Gait.BOUND: 1.4,
    Gait.PRONK: 0.8,
}


def _leg_ik(px, pz):
    r = np.sqrt(px * px + pz * pz)
    r = np.clip(r, abs(L1 - L2) + 0.005, L1 + L2 - 0.005)
    cos_k = (L1 * L1 + L2 * L2 - r * r) / (2.0 * L1 * L2)
    phi_k = np.arccos(np.clip(cos_k, -1.0, 1.0))
    calf = phi_k - np.pi
    sin_hip = np.clip(L2 * np.sin(phi_k) / r, -1.0, 1.0)
    thigh = np.arctan2(px, pz) + np.arcsin(sin_hip)
    return float(thigh), float(calf)


def _foot_target(phi, stride, duty_factor, step_height, px_nom=0.0):
    boundary = duty_factor * 2.0 * np.pi
    if phi < boundary:
        t = phi / boundary
        return px_nom + stride * (0.5 - t), PZ_NOM
    t = (phi - boundary) / (2.0 * np.pi - boundary)
    return px_nom + stride * (-0.5 + t), PZ_NOM - step_height * np.sin(t * np.pi)


def _joint_targets(leg_phases, cmd, gait_params):
    vx, vy, wz = map(float, cmd)
    speed = float(np.hypot(vx, wz * 0.3))

    if speed < 0.02 or gait_params.name == "stand":
        return np.array([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1.0, -1.5,
            -0.1, 1.0, -1.5,
        ])

    gait = Gait(gait_params.name)
    stride = STRIDE_LEN * STRIDE_SCALE[gait] * np.clip(vx / VX_MAX, -1.0, 1.0)
    step_height = STEP_HEIGHT[gait]
    targets = []

    for i in range(4):
        # Right legs +, left legs −: longer right stride → CCW (positive wz).
        yaw_sign = 1.0 if i % 2 == 1 else -1.0
        px, pz = _foot_target(
            leg_phases[i],
            stride + yaw_sign * YAW_STRIDE_GAIN * wz,
            gait_params.duty_factor,
            step_height,
            float(PX_NOM[i]),
        )
        thigh, calf = _leg_ik(px, pz)
        lat_sign = 1.0 if i % 2 == 0 else -1.0
        # Strafe + small yaw hip assist so turns don't rely on stride alone.
        hip = (
            HIP_DEFAULT[i]
            + lat_sign * 0.12 * vy / VY_MAX
            + yaw_sign * 0.06 * wz / WZ_MAX
        )
        targets.extend([hip, thigh, calf])

    return np.array(targets)


def _make_key_reader():
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


class Teleop(Node):
    def __init__(self):
        super().__init__("go2_gz_keyboard_teleop")
        self.pubs = [self.create_publisher(Float64, f"/go2/cmd/{name}", 10) for name in JOINTS]

    def publish_targets(self, targets):
        for pub, value in zip(self.pubs, targets):
            msg = Float64()
            msg.data = float(value)
            pub.publish(msg)


def main():
    rclpy.init()
    node = Teleop()
    scheduler = GaitScheduler()
    cycle_phase = 0.0
    cmd = np.zeros(3, dtype=np.float64)
    do_quit = threading.Event()
    read_key, restore = _make_key_reader()

    def keys():
        try:
            while not do_quit.is_set():
                key = read_key()
                if key is None:
                    time.sleep(0.005)
                    continue
                if key in ("w", "W"):
                    cmd[0] = min(cmd[0] + CMD_STEP, VX_MAX)
                elif key in ("s", "S"):
                    cmd[0] = max(cmd[0] - CMD_STEP, VX_MIN)
                elif key in ("a", "A"):
                    cmd[1] = min(cmd[1] + CMD_STEP, VY_MAX)
                elif key in ("d", "D"):
                    cmd[1] = max(cmd[1] - CMD_STEP, -VY_MAX)
                elif key in ("q", "Q"):
                    cmd[2] = min(cmd[2] + CMD_STEP, WZ_MAX)
                elif key in ("e", "E"):
                    cmd[2] = max(cmd[2] - CMD_STEP, -WZ_MAX)
                elif key == " ":
                    cmd[:] = 0.0
                elif key == "\x1b":
                    do_quit.set()
        finally:
            restore()

    threading.Thread(target=keys, daemon=True).start()
    print("Go2 Gazebo teleop: W/S forward | A/D strafe | Q/E yaw | Space stop | ESC quit")

    try:
        while rclpy.ok() and not do_quit.is_set():
            speed = float(np.hypot(cmd[0], cmd[2] * 0.3))
            gait = scheduler.get_gait_params(speed)
            cycle_phase = (cycle_phase + 2.0 * np.pi * gait.frequency * CTRL_DT) % (2.0 * np.pi)
            leg_phases = (cycle_phase + np.array(gait.phase_offsets) * 2.0 * np.pi) % (2.0 * np.pi)
            targets = np.clip(_joint_targets(leg_phases, cmd, gait), -2.7, 2.7)
            node.publish_targets(targets)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(CTRL_DT)
    finally:
        do_quit.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
