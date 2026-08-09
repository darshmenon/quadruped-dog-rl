#!/usr/bin/env python3
"""Lightweight frontier exploration driver for the Go2 in Gazebo.

There's no Nav2 costmap/planner wired up for the Go2 yet (no footprint or
controller tuning exists for a legged base), so this doesn't do Nav2-style
frontier exploration with a global planner. Instead it directly scores
frontier cells on the /map occupancy grid (wavefront from the robot's cell,
frontier = reachable free cell adjacent to unknown) and hands the best
cluster's centroid off to whichever locomotion backend is driving:

- control_mode:=cmd_vel (default, CHAMP backend): a simple point-and-go
  Twist controller published to /cmd_vel, which CHAMP's
  quadruped_controller (cmd_vel/smooth) turns into a walking gait.
- control_mode:=nmpc_goal (Quad-SDK backend): publish the goal as a
  geometry_msgs/PointStamped to Quad-SDK's live goal_state topic (see
  global_body_planner.cpp's goal_state_sub_) and let global_body_planner /
  local_planner / nmpc_controller do the actual walking -- no Twist
  involved, so linear_speed/angular_speed/BLOCKED_LOOKAHEAD don't apply.

Good enough to grow the RTAB-Map map autonomously; not a substitute for a
real planner if obstacles crowd the direct line to a goal (cmd_vel mode
only -- NMPC mode gets real obstacle-aware planning from Quad-SDK itself).

Usage:
    python3 scripts/frontier_explorer_go2.py --ros-args -p use_sim_time:=true
    python3 scripts/frontier_explorer_go2.py --ros-args -p use_sim_time:=true \\
        -p control_mode:=nmpc_goal -p base_frame:=body \\
        -p goal_state_topic:=/robot_1/goal_state
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


FREE_MAX = 40          # occupancy value below which a cell counts as "free"
OCCUPIED_MIN = 65       # occupancy value at/above which a cell counts as "occupied"
MIN_FRONTIER_SIZE = 6    # ignore frontier clusters smaller than this many cells
GOAL_TOLERANCE = 0.45    # m, close enough to a frontier goal to replan
BLOCKED_LOOKAHEAD = 0.5  # m, don't drive forward if this close a cell is occupied
NO_FRONTIER_TIMEOUT = 30.0  # s with no frontier found before declaring done


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer_go2')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('linear_speed', 0.35)
        self.declare_parameter('angular_speed', 0.6)
        self.declare_parameter('control_period', 0.1)
        self.declare_parameter('control_mode', 'cmd_vel')  # 'cmd_vel' or 'nmpc_goal'
        self.declare_parameter('goal_state_topic', '/robot_1/goal_state')

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_mode = self.get_parameter('control_mode').value
        self.published_goal_xy = None  # last goal actually sent in nmpc_goal mode

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.control_mode == 'nmpc_goal':
            self.goal_state_pub = self.create_publisher(
                PointStamped, self.get_parameter('goal_state_topic').value, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)

        self.latest_map = None
        self.goal_xy = None
        self.last_frontier_seen = self.get_clock().now()

        # champ_joint_trajectory_to_go2_gz.py (cmd_vel mode only) treats
        # /cmd_vel as stale after 0.35s (COMMAND_TIMEOUT_S) and falls back to
        # its idle-drift-reset behaviour, which teleports the robot back
        # upright -- so this needs to keep publishing well under that, not
        # just whenever it replans a goal. nmpc_goal mode has no such
        # deadline (Quad-SDK's global_body_planner just keeps tracking
        # whatever goal it last got), but the expensive frontier search only
        # actually runs inside _step when goal_xy is None either way, so a
        # fast tick here stays cheap in both modes.
        period = self.get_parameter('control_period').value
        self.create_timer(period, self._step)
        self.get_logger().info(
            f'frontier_explorer_go2 started (control_mode={self.control_mode}, '
            f'map_frame={self.map_frame}, base_frame={self.base_frame})')

    def _on_map(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f'no {self.map_frame}->{self.base_frame} TF yet: {exc}',
                                    throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _find_frontier_goal(self, msg: OccupancyGrid, robot_x, robot_y):
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data, dtype=np.int16).reshape((h, w))

        rc = int((robot_x - ox) / res)
        rr = int((robot_y - oy) / res)
        if not (0 <= rr < h and 0 <= rc < w):
            self.get_logger().warn('robot cell is outside the current map bounds')
            return None

        # Wavefront BFS over free cells reachable from the robot's cell.
        # Unknown cells (-1) must NOT count as free -- excluding them here
        # is what makes a cell adjacent to unknown space a frontier at all;
        # otherwise the BFS floods straight through unmapped territory
        # (-1 < FREE_MAX is true) and never sees a boundary to stop at.
        known = grid >= 0
        free = known & (grid < FREE_MAX)
        reachable = np.zeros_like(free)
        if not free[rr, rc]:
            # Snap to the nearest free cell within a small radius so we can
            # still start the search right after spawn / a tight replan.
            found = False
            for radius in range(1, 6):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        r2, c2 = rr + dr, rc + dc
                        if 0 <= r2 < h and 0 <= c2 < w and free[r2, c2]:
                            rr, rc, found = r2, c2, True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None

        stack = [(rr, rc)]
        reachable[rr, rc] = True
        frontier_cells = []
        while stack:
            r, c = stack.pop()
            is_frontier = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                val = grid[nr, nc]
                if val < 0:  # unknown
                    is_frontier = True
                    continue
                if free[nr, nc] and not reachable[nr, nc] and val < OCCUPIED_MIN:
                    reachable[nr, nc] = True
                    stack.append((nr, nc))
            if is_frontier:
                frontier_cells.append((r, c))

        if not frontier_cells:
            return None

        # Cluster frontier cells (4-connected) and score each cluster by
        # size / (1 + distance from robot), same tradeoff explore_lite uses
        # (bigger unknown regions win, but nearer ones are cheaper to reach).
        frontier_set = set(frontier_cells)
        visited = set()
        best_score, best_centroid = -1.0, None
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster = []
            queue = [cell]
            visited.add(cell)
            while queue:
                r, c = queue.pop()
                cluster.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in frontier_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(cluster) < MIN_FRONTIER_SIZE:
                continue
            cr = sum(p[0] for p in cluster) / len(cluster)
            cc = sum(p[1] for p in cluster) / len(cluster)
            gx = ox + (cc + 0.5) * res
            gy = oy + (cr + 0.5) * res
            dist = math.hypot(gx - robot_x, gy - robot_y)
            score = len(cluster) / (1.0 + dist)
            if score > best_score:
                best_score, best_centroid = score, (gx, gy)

        return best_centroid

    def _path_blocked(self, msg: OccupancyGrid, robot_x, robot_y, yaw):
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape((h, w))
        lx = robot_x + BLOCKED_LOOKAHEAD * math.cos(yaw)
        ly = robot_y + BLOCKED_LOOKAHEAD * math.sin(yaw)
        c = int((lx - ox) / res)
        r = int((ly - oy) / res)
        if not (0 <= r < h and 0 <= c < w):
            return False
        return grid[r, c] >= OCCUPIED_MIN

    def _step(self):
        msg = self.latest_map
        if msg is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        rx, ry, yaw = pose

        if self.goal_xy is not None:
            dist_to_goal = math.hypot(self.goal_xy[0] - rx, self.goal_xy[1] - ry)
            if dist_to_goal < GOAL_TOLERANCE:
                self.goal_xy = None

        if self.goal_xy is None:
            self.goal_xy = self._find_frontier_goal(msg, rx, ry)

        if self.goal_xy is None:
            elapsed = (self.get_clock().now() - self.last_frontier_seen).nanoseconds / 1e9
            if elapsed > NO_FRONTIER_TIMEOUT:
                self.get_logger().info(
                    f'no frontiers left for {elapsed:.0f}s -- exploration complete, stopping.',
                    throttle_duration_sec=NO_FRONTIER_TIMEOUT)
            if self.control_mode == 'cmd_vel':
                self.cmd_pub.publish(Twist())
            return

        self.last_frontier_seen = self.get_clock().now()

        if self.control_mode == 'nmpc_goal':
            # Quad-SDK's global_body_planner already tracks whatever goal it
            # last received (see global_body_planner.cpp's goal_state_sub_)
            # and dedups by header.stamp, so only publish when the frontier
            # goal actually changes -- no need to re-send every tick.
            if self.goal_xy != self.published_goal_xy:
                gx, gy = self.goal_xy
                point = PointStamped()
                point.header.stamp = self.get_clock().now().to_msg()
                point.header.frame_id = self.map_frame
                point.point.x, point.point.y = gx, gy
                self.goal_state_pub.publish(point)
                self.published_goal_xy = self.goal_xy
            return

        gx, gy = self.goal_xy
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        angle_err = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))

        cmd = Twist()
        if abs(angle_err) > 0.35:
            cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_err))
        elif self._path_blocked(msg, rx, ry, yaw):
            # Something's in the way of a straight line to the goal -- this
            # explorer has no local planner, so just turn and re-score next
            # tick rather than drive into it.
            cmd.angular.z = self.angular_speed
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, 0.5 * angle_err))
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.control_mode == 'cmd_vel':
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
