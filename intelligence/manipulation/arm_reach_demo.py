"""
Scripted reach demo -- drives arm_reach_node.py through a fixed sequence of
targets to sanity-check the arm's control path (IK -> /arm/target ->
per-joint /go2/cmd/<joint> -> Gazebo) without any RL involved.

Usage (with Gazebo + arm_reach_node already running, e.g. via
`ros2 launch training/launch/gazebo_rl.launch.py enable_arm_reach:=true`):

    python3 intelligence/manipulation/arm_reach_demo.py
    python3 intelligence/manipulation/arm_reach_demo.py --loop
    python3 intelligence/manipulation/arm_reach_demo.py --hold-seconds 3.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from intelligence.manipulation.arm_ik import inverse_kinematics, forward_kinematics

# (x, y, z) targets in the arm's base_mount frame, in meters -- each verified
# reachable (within the arm+gripper's length and +/-90deg joint limits) by
# test_arm_ik.py's grid before being used here. Values further out than they
# look necessary: with a level default wrist_pitch, the arm can't fold in
# close (see arm_ik.py's docstring on the near-arm dead zone).
WAYPOINTS = [
    ("forward-up",    (0.42, 0.0, 0.15)),
    ("right",         (0.35, -0.24, 0.0)),
    ("left",          (0.35, 0.24, 0.0)),
    ("forward-down",  (0.42, 0.0, -0.1)),
    ("forward-level", (0.42, 0.0, 0.0)),
]


class ArmReachDemo(Node):
    def __init__(self, hold_seconds, loop):
        super().__init__("arm_reach_demo")
        self._pub = self.create_publisher(Point, "/arm/target", 10)
        self._loop = loop
        self._index = 0
        self._pass_count = 1
        self._done = False
        self.get_logger().info(
            f"starting arm reach demo: {len(WAYPOINTS)} waypoints, "
            f"hold_seconds={hold_seconds}, loop={loop}"
        )
        self.create_timer(hold_seconds, self._advance)
        self._advance()  # publish the first waypoint immediately

    def _advance(self):
        if self._done:
            return
        name, (x, y, z) = WAYPOINTS[self._index]
        pose = inverse_kinematics(x, y, z)
        # Reachability was already checked at startup for every waypoint in
        # main() -- this can only fail here if arm_ik's model/limits changed
        # underneath the running process, so surface it loudly.
        if pose is None:
            self.get_logger().error(
                f"waypoint {self._index + 1}/{len(WAYPOINTS)} '{name}' "
                f"({x:.3f}, {y:.3f}, {z:.3f}) became unreachable, skipping"
            )
        else:
            fx, fy, fz = forward_kinematics(pose)
            self.get_logger().info(
                f"waypoint {self._index + 1}/{len(WAYPOINTS)} (pass {self._pass_count}) "
                f"'{name}' target=({x:.3f}, {y:.3f}, {z:.3f}) "
                f"fk_check=({fx:.3f}, {fy:.3f}, {fz:.3f}) "
                f"joints=[base={pose.base:.3f}, shoulder={pose.shoulder:.3f}, "
                f"elbow={pose.elbow:.3f}, wrist_pitch={pose.wrist_pitch:.3f}, "
                f"wrist_roll={pose.wrist_roll:.3f}]"
            )
            self._pub.publish(Point(x=x, y=y, z=z))

        self._index += 1
        if self._index >= len(WAYPOINTS):
            if self._loop:
                self._index = 0
                self._pass_count += 1
                self.get_logger().info(f"sequence complete, looping (pass {self._pass_count})")
            else:
                self._done = True
                self.get_logger().info(
                    f"sequence complete after {self._pass_count} pass(es), holding final waypoint"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-seconds", type=float, default=2.5,
                         help="seconds to hold each waypoint before advancing")
    parser.add_argument("--loop", action="store_true",
                         help="repeat the waypoint sequence instead of stopping after one pass")
    args = parser.parse_args()

    # Fail fast with a clear error if a waypoint isn't actually reachable,
    # rather than relying on arm_reach_node's runtime "unreachable" warning.
    for name, (x, y, z) in WAYPOINTS:
        if inverse_kinematics(x, y, z) is None:
            parser.error(f"waypoint '{name}' ({x}, {y}, {z}) is not reachable by arm_ik")

    rclpy.init()
    node = ArmReachDemo(args.hold_seconds, args.loop)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
