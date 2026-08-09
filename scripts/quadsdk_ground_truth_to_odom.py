#!/usr/bin/env python3
"""Publish odom + odom->body TF from Quad-SDK's ground-truth robot state.

Quad-SDK has no nav_msgs/Odometry publisher anywhere (its own state
estimator is disabled by default -- see robot_driver's "estimator_id='none'"
log line) and no odom->body TF chain either; the only existing dynamic TF
(rviz_interface_node's "map -> <namespace>_ground_truth/body") is a
visualization side effect of the `rviz` arg, not a stable API to depend on.
The actual ground truth comes straight from a Gazebo plugin
(GroundTruthEstimator) as quad_msgs/RobotState on .../state/ground_truth, so
this republishes that -- verbatim, no filtering -- as the odom this repo's
other SLAM path (scripts/gz_pose_to_odom.py) already establishes for the
CHAMP/go2_gz.urdf backend.

Usage:
    python3 scripts/quadsdk_ground_truth_to_odom.py \\
        --ros-args -p use_sim_time:=true \\
        -p ground_truth_topic:=/robot_1/state/ground_truth \\
        -p odom_topic:=/robot_1/odom -p base_frame:=body
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from quad_msgs.msg import RobotState
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class QuadSdkGroundTruthToOdom(Node):
    def __init__(self):
        super().__init__('quadsdk_ground_truth_to_odom')
        self.declare_parameter('ground_truth_topic', '/robot_1/state/ground_truth')
        self.declare_parameter('odom_topic', '/robot_1/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'body')

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            RobotState, self.get_parameter('ground_truth_topic').value, self._on_state, 10)

        self.get_logger().info(
            f'quadsdk_ground_truth_to_odom started: '
            f'{self.get_parameter("ground_truth_topic").value} -> '
            f'{self.odom_frame}->{self.base_frame} TF + {self.get_parameter("odom_topic").value}')

    def _on_state(self, msg: RobotState):
        stamp = msg.header.stamp

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = msg.body.pose
        odom.twist.twist = msg.body.twist
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = msg.body.pose.position.x
        tf.transform.translation.y = msg.body.pose.position.y
        tf.transform.translation.z = msg.body.pose.position.z
        tf.transform.rotation = msg.body.pose.orientation
        self.tf_broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = QuadSdkGroundTruthToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
