"""
Scripted pick demo -- drives arm_reach_node.py through open -> pre-grasp ->
descend -> close -> lift to pick up the cylinder spawned by
spawn_pick_scene.py. No locomotion involved: the object is placed within the
stationary arm's reach, matching spawn_pick_scene.py's CYLINDER_ARM_FRAME.

Each phase waits for /joint_states to actually confirm convergence (within
POSITION_TOLERANCE / GRIPPER_TOLERANCE) before advancing, instead of a blind
fixed-duration hold -- so a stalled joint (e.g. a finger jammed against its
sibling, see arm.urdf.xacro's self-collision note) shows up as a loud
"did not converge" warning rather than silently being papered over by a
timer. Falls back to PHASE_TIMEOUT so a stuck joint can't hang the sequence
forever.

Requires /joint_states to actually be publishing -- see
scripts/make_go2_stand.py's JointStatePublisher plugin.

Usage (with Gazebo + arm_reach_node running and the pick scene spawned,
e.g. via `ros2 launch training/launch/gazebo_rl.launch.py
enable_arm_reach:=true` then `python3 spawn_pick_scene.py`):

    python3 intelligence/manipulation/pick_demo.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

from intelligence.manipulation.arm_ik import inverse_kinematics, forward_kinematics

# Must match spawn_pick_scene.py's CYLINDER_ARM_FRAME so the gripper actually
# closes around the object instead of empty air.
GRASP_XYZ = (0.42, 0.0, 0.0)
PRE_GRASP_XYZ = (0.42, 0.0, 0.15)   # clear of the cylinder before descending
LIFT_XYZ = (0.42, 0.0, 0.20)        # holds the grasp radius/base so it stays reachable

GRIPPER_OPEN = 0.025  # matches arm_reach_node.GRIPPER_MAX_WIDTH
GRIPPER_CLOSED = 0.0

ARM_JOINTS = ["base_joint", "lower_arm_joint", "upper_arm_joint", "wrist1_joint", "wrist2_joint"]
GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]

POSITION_TOLERANCE = 0.03    # rad, arm joints
GRIPPER_TOLERANCE = 0.004    # m, finger joints
PHASE_TIMEOUT = 6.0          # seconds -- give up waiting and move on regardless
CHECK_PERIOD = 0.05          # seconds between convergence checks

# (phase name, arm target xyz or None, gripper width or None)
PHASES = [
    ("open gripper, move to pre-grasp",  PRE_GRASP_XYZ, GRIPPER_OPEN),
    ("descend to grasp height",          GRASP_XYZ,     None),
    ("close gripper",                    None,          GRIPPER_CLOSED),
    ("lift",                             LIFT_XYZ,      None),
]


class PickDemo(Node):
    def __init__(self):
        super().__init__("pick_demo")
        self._target_pub = self.create_publisher(Point, "/arm/target", 10)
        self._gripper_pub = self.create_publisher(Float64, "/gripper/command", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

        self._latest_positions = {}
        self._phase_index = 0
        self._phase_start_time = None
        self._phase_targets = {}  # joint name -> target value, for the current phase
        self._done = False

        self.get_logger().info(f"pick_demo starting: {len(PHASES)} phases")
        for i, (name, xyz, width) in enumerate(PHASES):
            if xyz is not None and inverse_kinematics(*xyz) is None:
                self.get_logger().error(
                    f"phase {i + 1} '{name}' target {xyz} is unreachable by arm_ik -- "
                    "aborting before sending any commands"
                )
                raise SystemExit(1)

        self._run_phase()
        self.create_timer(CHECK_PERIOD, self._check_convergence)

    def _on_joint_states(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            self._latest_positions[name] = position

    def _run_phase(self):
        if self._phase_index >= len(PHASES):
            self._done = True
            self.get_logger().info("pick sequence complete, holding lift pose")
            return

        name, xyz, width = PHASES[self._phase_index]
        self.get_logger().info(f"phase {self._phase_index + 1}/{len(PHASES)}: {name}")

        self._phase_targets = {}

        if xyz is not None:
            pose = inverse_kinematics(*xyz)
            fx, fy, fz = forward_kinematics(pose)
            self.get_logger().info(
                f"  arm target=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) "
                f"fk_check=({fx:.3f}, {fy:.3f}, {fz:.3f}) "
                f"joints=[base={pose.base:.3f}, shoulder={pose.shoulder:.3f}, "
                f"elbow={pose.elbow:.3f}, wrist_pitch={pose.wrist_pitch:.3f}]"
            )
            self._target_pub.publish(Point(x=xyz[0], y=xyz[1], z=xyz[2]))
            for jname, jval in zip(ARM_JOINTS, pose.as_list()):
                self._phase_targets[jname] = jval

        if width is not None:
            self.get_logger().info(f"  gripper width -> {width:.4f}")
            self._gripper_pub.publish(Float64(data=width))
            self._phase_targets["left_finger_joint"] = width
            self._phase_targets["right_finger_joint"] = -width

        self._phase_index += 1
        self._phase_start_time = self.get_clock().now()

    def _check_convergence(self):
        if self._done or self._phase_start_time is None:
            return

        elapsed = (self.get_clock().now() - self._phase_start_time).nanoseconds / 1e9

        unconverged = []
        for jname, target in self._phase_targets.items():
            actual = self._latest_positions.get(jname)
            tol = GRIPPER_TOLERANCE if jname in GRIPPER_JOINTS else POSITION_TOLERANCE
            if actual is None or abs(actual - target) > tol:
                unconverged.append((jname, actual, target))

        if not unconverged:
            self.get_logger().info(f"  converged after {elapsed:.2f}s")
            self._run_phase()
            return

        if elapsed >= PHASE_TIMEOUT:
            details = ", ".join(
                f"{jname}={'?' if actual is None else f'{actual:.4f}'} "
                f"(target {target:.4f})"
                for jname, actual, target in unconverged
            )
            self.get_logger().warn(
                f"  did NOT converge within {PHASE_TIMEOUT}s, moving on anyway -- "
                f"stalled: {details}"
            )
            self._run_phase()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    rclpy.init()
    node = PickDemo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
