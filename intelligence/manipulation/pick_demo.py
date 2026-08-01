"""
Scripted pick demo -- drives arm_reach_node.py through open -> pre-grasp ->
descend -> close -> lift to pick up the cylinder spawned by
spawn_pick_scene.py. No locomotion involved: the object is placed within the
stationary arm's reach, matching spawn_pick_scene.py's CYLINDER_ARM_FRAME.

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

from intelligence.manipulation.arm_ik import inverse_kinematics, forward_kinematics

# Must match spawn_pick_scene.py's CYLINDER_ARM_FRAME so the gripper actually
# closes around the object instead of empty air.
GRASP_XYZ = (0.42, 0.0, 0.0)
PRE_GRASP_XYZ = (0.42, 0.0, 0.15)   # clear of the cylinder before descending
LIFT_XYZ = (0.42, 0.0, 0.20)        # holds the grasp radius/base so it stays reachable

GRIPPER_OPEN = 0.025  # matches arm_reach_node.GRIPPER_MAX_WIDTH
GRIPPER_CLOSED = 0.0

# (phase name, arm target or None, gripper width or None, hold seconds)
# None means "leave as previously commanded" for that channel.
PHASES = [
    ("open gripper, move to pre-grasp",  PRE_GRASP_XYZ, GRIPPER_OPEN,   2.5),
    ("descend to grasp height",          GRASP_XYZ,     None,           2.0),
    ("close gripper",                    None,          GRIPPER_CLOSED, 1.5),
    ("lift",                             LIFT_XYZ,      None,           3.0),
]


class PickDemo(Node):
    def __init__(self):
        super().__init__("pick_demo")
        self._target_pub = self.create_publisher(Point, "/arm/target", 10)
        self._gripper_pub = self.create_publisher(Float64, "/gripper/command", 10)
        self._phase_index = 0
        self._done = False

        self.get_logger().info(f"pick_demo starting: {len(PHASES)} phases")
        for i, (name, xyz, width, hold) in enumerate(PHASES):
            if xyz is not None and inverse_kinematics(*xyz) is None:
                self.get_logger().error(
                    f"phase {i + 1} '{name}' target {xyz} is unreachable by arm_ik -- "
                    "aborting before sending any commands"
                )
                raise SystemExit(1)

        self._run_phase()

    def _run_phase(self):
        if self._phase_index >= len(PHASES):
            self._done = True
            self.get_logger().info("pick sequence complete, holding lift pose")
            return

        name, xyz, width, hold = PHASES[self._phase_index]
        self.get_logger().info(
            f"phase {self._phase_index + 1}/{len(PHASES)}: {name} (hold {hold}s)"
        )

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

        if width is not None:
            self.get_logger().info(f"  gripper width -> {width:.4f}")
            self._gripper_pub.publish(Float64(data=width))

        self._phase_index += 1
        self._phase_timer = self.create_timer(hold, self._on_phase_timer)

    def _on_phase_timer(self):
        # create_timer has no one-shot mode in this rclpy version; cancel
        # immediately so each phase only fires once.
        self._phase_timer.cancel()
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
