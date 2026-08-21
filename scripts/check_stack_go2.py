#!/usr/bin/env python3
"""Check the core Go2 autonomy topics and TF links."""

import argparse
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


DEFAULT_TOPICS = {
    "/odom": Odometry,
    "/points": PointCloud2,
    "/map": OccupancyGrid,
    "/obstacle_tracker/state": String,
}


class StackCheck(Node):
    def __init__(self, topics):
        super().__init__("go2_stack_check")
        self.seen = {name: None for name in topics}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for name, msg_type in topics.items():
            self.create_subscription(msg_type, name, self._callback(name), qos)

    def _callback(self, topic):
        def cb(_msg):
            self.seen[topic] = time.monotonic()
        return cb

    def topic_report(self, max_age):
        now = time.monotonic()
        rows = []
        ok = True
        for topic, stamp in self.seen.items():
            if stamp is None:
                rows.append((False, topic, "no messages"))
                ok = False
                continue
            age = now - stamp
            fresh = age <= max_age
            rows.append((fresh, topic, f"last message {age:.1f}s ago"))
            ok = ok and fresh
        return ok, rows

    def tf_report(self, pairs):
        rows = []
        ok = True
        for target, source in pairs:
            try:
                self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.2))
                rows.append((True, f"{target}->{source}", "available"))
            except TransformException as exc:
                rows.append((False, f"{target}->{source}", str(exc).split("\n", 1)[0]))
                ok = False
        return ok, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0,
                        help="seconds to wait for first messages")
    parser.add_argument("--max-age", type=float, default=3.0,
                        help="maximum allowed topic age in seconds")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--skip-obstacles", action="store_true",
                        help="do not require /obstacle_tracker/state")
    args = parser.parse_args()

    topics = dict(DEFAULT_TOPICS)
    if args.skip_obstacles:
        topics.pop("/obstacle_tracker/state")

    rclpy.init()
    node = StackCheck(topics)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and any(stamp is None for stamp in node.seen.values()):
            rclpy.spin_once(node, timeout_sec=0.1)

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

        topics_ok, topic_rows = node.topic_report(args.max_age)
        tf_ok, tf_rows = node.tf_report([
            (args.odom_frame, args.base_frame),
            (args.map_frame, args.base_frame),
        ])
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("Go2 stack check")
    for good, name, detail in topic_rows:
        print(f"[{'OK' if good else 'FAIL'}] topic {name}: {detail}")
    for good, name, detail in tf_rows:
        print(f"[{'OK' if good else 'FAIL'}] tf {name}: {detail}")

    return 0 if topics_ok and tf_ok else 1


if __name__ == "__main__":
    sys.exit(main())
