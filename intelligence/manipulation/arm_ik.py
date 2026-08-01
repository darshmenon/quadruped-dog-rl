"""
Analytic forward/inverse kinematics for the 5-DOF manipulator arm
(ros2/champ_description/urdf/arm.urdf.xacro).

Joint order: [base_joint, lower_arm_joint, upper_arm_joint, wrist1_joint, wrist2_joint]
Axes:        [z (yaw),    y (pitch),       y (pitch),       y (pitch),    x (roll)]

All positions are in the arm's own base_mount frame (the champ_arm macro's
<origin> on the parent link, e.g. (0.08, 0, 0.057) on Go2's "base" link per
urdf/go2_unitree/urdf/go2_gz.urdf.xacro) -- not the robot body frame.

wrist2_joint (roll about the arm's own pointing axis) sits at zero link
length beyond its own rotation axis, so it only sets end-effector
orientation and never moves the reachable position -- it's solved for
directly from the caller's wrist_roll argument, not part of the position IK.

The end effector is the gripper's fingertip centerline, which sits
GRIPPER_LENGTH past wrist2_joint (see arm.urdf.xacro's wrist2_to_gripper_joint
+ left/right_finger_joint origins). The gripper's own open/close motion is
side-to-side about that centerline, so it doesn't move the reach point --
GRIPPER_LENGTH is just a fixed extension of the same wrist2-forward segment
used for WRIST_LENGTH below, not a new DOF in this model.
"""

import math
from dataclasses import dataclass

SHOULDER_HEIGHT = 0.035    # arm_base -> lower_arm_joint, z offset
UPPER_ARM_LENGTH = 0.195   # lower_arm_joint -> upper_arm_joint
FOREARM_LENGTH = 0.195     # upper_arm_joint -> wrist1_joint
WRIST_LENGTH = 0.065       # wrist1_joint -> wrist2_joint
GRIPPER_LENGTH = 0.055     # wrist2_joint -> fingertip centerline (0.015 + 0.04)
EE_LENGTH = WRIST_LENGTH + GRIPPER_LENGTH  # wrist1_joint -> end effector
MAX_REACH = UPPER_ARM_LENGTH + FOREARM_LENGTH + EE_LENGTH

JOINT_LIMIT = math.pi / 2  # all 5 joints: +/-90 deg (arm.urdf.xacro)


@dataclass
class ArmPose:
    base: float
    shoulder: float
    elbow: float
    wrist_pitch: float
    wrist_roll: float

    def as_list(self):
        return [self.base, self.shoulder, self.elbow, self.wrist_pitch, self.wrist_roll]


def _within_limits(*angles):
    return all(-JOINT_LIMIT - 1e-9 <= a <= JOINT_LIMIT + 1e-9 for a in angles)


def forward_kinematics(pose: ArmPose):
    """End-effector (gripper fingertip centerline) position in the base_mount frame."""
    phi = pose.shoulder + pose.elbow + pose.wrist_pitch
    r = (
        UPPER_ARM_LENGTH * math.cos(pose.shoulder)
        + FOREARM_LENGTH * math.cos(pose.shoulder + pose.elbow)
        + EE_LENGTH * math.cos(phi)
    )
    z = (
        SHOULDER_HEIGHT
        - UPPER_ARM_LENGTH * math.sin(pose.shoulder)
        - FOREARM_LENGTH * math.sin(pose.shoulder + pose.elbow)
        - EE_LENGTH * math.sin(phi)
    )
    x = r * math.cos(pose.base)
    y = r * math.sin(pose.base)
    return x, y, z


def inverse_kinematics(x, y, z, wrist_pitch=0.0, wrist_roll=0.0):
    """
    Solve for joint angles reaching (x, y, z) in the base_mount frame, holding
    the end-effector pitch at `wrist_pitch` (rad, 0 = last link horizontal).

    Returns an ArmPose, or None if the target is unreachable given the arm's
    link lengths and +/-90 deg joint limits (tries both elbow-up and
    elbow-down solutions before giving up).

    Known gap: targets with |atan2(y, x)| > 90 deg (behind/beside the arm's
    forward cone) are reported unreachable even though a "folded" shoulder
    configuration bent back past the base axis could technically reach them
    -- resolving that redundancy isn't implemented, since it isn't the
    reach-forward-and-grab case this is built for.
    """
    base = math.atan2(y, x)
    if not _within_limits(base):
        return None

    r = math.hypot(x, y)
    # Subtract the fixed wrist+gripper offset to reduce this to a 2-link
    # shoulder/elbow problem targeting the wrist1_joint position.
    r2 = r - EE_LENGTH * math.cos(wrist_pitch)
    z2 = (z - SHOULDER_HEIGHT) + EE_LENGTH * math.sin(wrist_pitch)

    d_sq = r2 * r2 + z2 * z2
    cos_elbow = (d_sq - UPPER_ARM_LENGTH ** 2 - FOREARM_LENGTH ** 2) / (
        2.0 * UPPER_ARM_LENGTH * FOREARM_LENGTH
    )
    if abs(cos_elbow) > 1.0:
        return None  # outside link-length reach

    for elbow_sign in (1.0, -1.0):
        elbow = elbow_sign * math.acos(cos_elbow)
        k1 = UPPER_ARM_LENGTH + FOREARM_LENGTH * math.cos(elbow)
        k2 = FOREARM_LENGTH * math.sin(elbow)
        shoulder = math.atan2(-z2, r2) - math.atan2(k2, k1)
        wrist = wrist_pitch - shoulder - elbow
        if _within_limits(shoulder, elbow, wrist, wrist_roll):
            return ArmPose(base, shoulder, elbow, wrist, wrist_roll)
    return None
