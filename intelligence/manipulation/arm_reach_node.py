"""
Arm Reach Node -- drives the 5-DOF manipulator arm to a commanded 3D point
using the closed-form IK in arm_ik.py, plus the 2-finger gripper bolted onto
its end (see ros2/champ_description/urdf/arm.urdf.xacro).

Subscriptions:
    /arm/target       geometry_msgs/Point   target position, arm base_mount frame (m)
    /gripper/command  std_msgs/Float64      desired half-gap width, meters
                                             (0.0 = closed .. GRIPPER_MAX_WIDTH = fully open)

Publications:
    /go2/cmd/base_joint         std_msgs/Float64
    /go2/cmd/lower_arm_joint    std_msgs/Float64
    /go2/cmd/upper_arm_joint    std_msgs/Float64
    /go2/cmd/wrist1_joint       std_msgs/Float64
    /go2/cmd/wrist2_joint       std_msgs/Float64
    /go2/cmd/left_finger_joint  std_msgs/Float64
    /go2/cmd/right_finger_joint std_msgs/Float64
        Native per-joint JointPositionController topics that
        scripts/make_go2_stand.py wires up for every arm+gripper joint
        (bridged to Gazebo by training/launch/gazebo_rl.launch.py). Both
        finger joints are commanded directly here (equal and opposite) since
        right_finger_joint's URDF <mimic> isn't enforced along this path --
        `gz sdf -p` drops <ros2_control> tags during URDF->SDF conversion
        (so gz_ros2_control/arm_position_controller in
        training/config/go2_ros2_control.yaml never actually starts here
        either), and mimic constraints aren't carried through that
        conversion.

Parameters:
    wrist_pitch  (float, default 0.0)   end-effector pitch held while reaching (rad)
    wrist_roll   (float, default 0.0)   end-effector roll held while reaching (rad)
    rate         (float, default 20.0)  command publish rate (Hz)

On an unreachable target, logs a warning and keeps holding the last valid pose.
Gripper widths outside [0, GRIPPER_MAX_WIDTH] are clamped, with a warning.

Usage:
    python3 intelligence/manipulation/arm_reach_node.py

    ros2 topic pub /arm/target geometry_msgs/msg/Point "{x: 0.3, y: 0.0, z: 0.05}"
    ros2 topic pub /gripper/command std_msgs/msg/Float64 "{data: 0.025}"  # open
    ros2 topic pub /gripper/command std_msgs/msg/Float64 "{data: 0.0}"   # close
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

from intelligence.manipulation.arm_ik import inverse_kinematics

ARM_JOINTS = ["base_joint", "lower_arm_joint", "upper_arm_joint", "wrist1_joint", "wrist2_joint"]
STOW_POSE = [0.0, 1.4, 0.8, 0.3, 0.0]  # matches scripts/make_go2_stand.py STANDING_POSE

# left_finger_joint / right_finger_joint limits in arm.urdf.xacro: [0, 0.025]
# and [-0.025, 0] respectively, so a single "half-gap width" drives both.
GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]
GRIPPER_MAX_WIDTH = 0.025
GRIPPER_STOW_WIDTH = 0.0  # closed, matches scripts/make_go2_stand.py STANDING_POSE


class ArmReachNode(Node):
    def __init__(self):
        super().__init__("arm_reach_node")

        self.declare_parameter("wrist_pitch", 0.0)
        self.declare_parameter("wrist_roll", 0.0)
        self.declare_parameter("rate", 20.0)

        self._wrist_pitch = float(self.get_parameter("wrist_pitch").value)
        self._wrist_roll = float(self.get_parameter("wrist_roll").value)
        rate = float(self.get_parameter("rate").value)

        self.get_logger().info(
            f"arm_reach_node starting: wrist_pitch={self._wrist_pitch}, "
            f"wrist_roll={self._wrist_roll}, rate={rate} Hz"
        )

        self._joint_targets = list(STOW_POSE)
        self._gripper_width = GRIPPER_STOW_WIDTH

        self._cmd_pubs = {
            joint: self.create_publisher(Float64, f"/go2/cmd/{joint}", 10)
            for joint in ARM_JOINTS + GRIPPER_JOINTS
        }
        self.create_subscription(Point, "/arm/target", self._target_cb, 10)
        self.create_subscription(Float64, "/gripper/command", self._gripper_cb, 10)
        self.create_timer(1.0 / rate, self._publish)

    def _target_cb(self, msg: Point):
        pose = inverse_kinematics(
            msg.x, msg.y, msg.z,
            wrist_pitch=self._wrist_pitch, wrist_roll=self._wrist_roll,
        )
        if pose is None:
            self.get_logger().warn(
                f"target ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}) unreachable, "
                "holding last pose"
            )
            return
        self._joint_targets = pose.as_list()
        self.get_logger().info(
            f"new arm target ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}) -> "
            f"joints=[{', '.join(f'{j:.3f}' for j in self._joint_targets)}]"
        )

    def _gripper_cb(self, msg: Float64):
        width = msg.data
        clamped = max(0.0, min(GRIPPER_MAX_WIDTH, width))
        if clamped != width:
            self.get_logger().warn(
                f"gripper width {width:.4f} out of range [0, {GRIPPER_MAX_WIDTH}], "
                f"clamping to {clamped:.4f}"
            )
        self._gripper_width = clamped
        self.get_logger().info(f"new gripper width {clamped:.4f}")

    def _publish(self):
        for joint, target in zip(ARM_JOINTS, self._joint_targets):
            self._cmd_pubs[joint].publish(Float64(data=target))
        self._cmd_pubs["left_finger_joint"].publish(Float64(data=self._gripper_width))
        self._cmd_pubs["right_finger_joint"].publish(Float64(data=-self._gripper_width))


def main():
    rclpy.init()
    node = ArmReachNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
