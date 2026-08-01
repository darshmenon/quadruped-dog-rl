"""
Round-trip and sanity checks for arm_ik.py. Pure math, no ROS/Gazebo needed:

    python3 intelligence/manipulation/test_arm_ik.py
"""

import math
import sys

from arm_ik import (
    ArmPose,
    MAX_REACH,
    forward_kinematics,
    inverse_kinematics,
)

FAILURES = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name}")


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# Round-trip: for a grid of reachable targets, IK -> FK should return to the target.
# r values chosen past the near-arm dead zone (see arm_ik.py: with a level
# wrist_pitch=0 and +/-90deg joint limits, targets much closer than this are
# unreachable even though they're within total arm+gripper length).
grid = []
for base in (-0.6, 0.0, 0.6):
    for r in (0.40, 0.425, 0.45):
        for z in (-0.1, 0.0, 0.15):
            grid.append((r * math.cos(base), r * math.sin(base), z))

reachable_checked = 0
for (x, y, z) in grid:
    pose = inverse_kinematics(x, y, z)
    if pose is None:
        continue
    reachable_checked += 1
    fx, fy, fz = forward_kinematics(pose)
    check(
        f"round-trip ({x:.3f},{y:.3f},{z:.3f}) -> joints -> ({fx:.3f},{fy:.3f},{fz:.3f})",
        close(fx, x, 1e-4) and close(fy, y, 1e-4) and close(fz, z, 1e-4),
    )

check(f"round-trip grid exercised >=5 reachable targets (got {reachable_checked})", reachable_checked >= 5)

# Targets clearly beyond MAX_REACH must be rejected.
check("far target rejected", inverse_kinematics(MAX_REACH * 2, 0, 0) is None)

# Target directly behind the arm's forward cone must be rejected (documented
# gap in arm_ik.py: no folded-shoulder redundancy resolution).
check("behind-arm target rejected", inverse_kinematics(-0.2, 0, 0) is None)

# Straight-ahead, arm-height target should be reachable with base ~= 0.
# (Closer-in targets with a level wrist_pitch=0 can violate the +/-90deg
# elbow limit -- see arm_ik.py's joint-limit check -- so this uses a target
# past that threshold rather than an arbitrary close one.)
straight = inverse_kinematics(0.4, 0.0, 0.0)
check("straight-ahead target reachable", straight is not None)
if straight is not None:
    check("straight-ahead base joint ~= 0", close(straight.base, 0.0, 1e-6))

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("All checks passed.")
