#!/usr/bin/env python3
"""Moving-obstacle detection + Kalman tracking from the Go2's 3D lidar.

Adapted from rosnav_bot's obstacle_tracker.py (a different project, see
~/rosnav/src/rosnav_bot/scripts/obstacle_tracker.py) for this project's
sensor: rosnav tracks a 2D LaserScan and diffs range values ray-by-ray
across frames to find "closing" (approaching) rays. The Go2 only has the
3D gpu_lidar's PointCloud2 (see slam3d_go2.launch.py) -- there's no fixed
per-ray structure to diff the same way. So instead of detecting motion
directly, this clusters every point cloud, every frame, within a height
band above the ground plane (ground_z_min..ground_z_max, base_frame-
relative) -- i.e. it tracks all nearby obstacles, not just ones visibly
approaching. The downstream Kalman filter (constant-velocity, per-track
ID) is what reveals which tracked clusters are actually moving and how
fast; a static obstacle just settles into a near-zero-velocity track --
easy to filter downstream (e.g. only react above some speed) if that
distinction matters to a caller. This node itself makes no moving/static
distinction, unlike rosnav's original.

The Track class (constant-velocity Kalman filter + ellipse extent fit) is
ported unchanged -- it was already pure numpy with no ROS coupling.
Clustering is NOT a straight port, though: rosnav's single-linkage scan
over ~360 LaserScan rays doesn't scale to a 16-channel x ~1800-azimuth
point cloud (tens of thousands of points/frame, O(n^2) would be far too
slow). Instead this bins filtered points into a cluster_radius-sized
grid and flood-fills adjacent occupied cells (same grid+BFS idiom
frontier_explorer_go2.py already uses for frontier clustering), tracking
per-cell (count, sum_x, sum_y, sum_xx, sum_yy, sum_xy) so cluster
centroid + covariance-ellipse extent can be computed from the aggregates
without ever storing per-point data.

Usage:
    python3 scripts/obstacle_tracker_go2.py --ros-args -p use_sim_time:=true
    python3 scripts/obstacle_tracker_go2.py --ros-args -p use_sim_time:=true \\
        -p points_topic:=/robot_1/points -p base_frame:=body
"""

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


GROUND_Z_MIN = 0.08      # m, base_frame-relative -- excludes ground-plane returns
GROUND_Z_MAX = 1.3       # m, base_frame-relative -- excludes overhead/sky noise
CLUSTER_CELL = 0.4       # m, grid cell size for point binning + flood-fill grouping
TRACK_GATE_DIST = 0.75   # m, max association distance
TRACK_MAX_MISSES = 4     # consecutive misses before a track is dropped
TRACK_MIN_HITS = 2       # matched updates before a track is published
PROCESS_NOISE = 0.5      # accel noise std, m/s^2
MEASUREMENT_NOISE = 0.2  # centroid position std, m
MIN_EXTENT = 0.3         # m, min ellipse diameter
EXTENT_GAIN = 0.3        # extent smoothing gain, 0-1
MARKER_LIFETIME = 0.5    # s
MAX_TRACK_RANGE = 10.0   # m, base_frame-relative -- see _on_cloud's comment
PREDICT_HORIZONS = (0.4, 0.8)  # s, lead time(s) for the predicted-points costmap feed
PREDICT_RING_POINTS = 10        # points sampled around each predicted ellipse ring


class Track:
    """Constant-velocity Kalman filter track: state = [x, y, vx, vy].

    Ported unchanged from rosnav_bot's obstacle_tracker.py -- pure numpy,
    no ROS dependency in the original either.
    """

    _H = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, track_id: int, x: float, y: float,
                 l1: float = 0.3, l2: float = 0.3, alpha: float = 0.0):
        self.id = track_id
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([0.25, 0.25, 1.0, 1.0])
        self.hits = 1
        self.misses = 0
        self.count = 1
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha

    def update_extent(self, l1: float, l2: float, alpha: float, gain: float):
        # Ellipse orientation is only defined mod pi (the major axis looks
        # the same rotated 180deg), so resolve that ambiguity before
        # blending to avoid the smoothed angle snapping back and forth.
        diff = (alpha - self.alpha + math.pi / 2) % math.pi - math.pi / 2
        self.alpha = (self.alpha + gain * diff + math.pi / 2) % math.pi - math.pi / 2
        self.l1 = (1 - gain) * self.l1 + gain * l1
        self.l2 = (1 - gain) * self.l2 + gain * l2

    def predict(self, dt: float, process_noise: float):
        if dt <= 0.0:
            return
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        q = process_noise ** 2
        Q = q * np.array([
            [dt**3 / 3, 0.0,       dt**2 / 2, 0.0],
            [0.0,       dt**3 / 3, 0.0,       dt**2 / 2],
            [dt**2 / 2, 0.0,       dt,        0.0],
            [0.0,       dt**2 / 2, 0.0,       dt],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, zx: float, zy: float, measurement_noise: float, count: int):
        z = np.array([zx, zy])
        R = (measurement_noise ** 2) * np.eye(2)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self._H) @ self.P
        self.hits += 1
        self.misses = 0
        self.count = count

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self):
        return float(self.x[2]), float(self.x[3])


class ObstacleTrackerGo2(Node):
    def __init__(self):
        super().__init__('obstacle_tracker_go2')
        self.declare_parameter('points_topic', '/points')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('max_track_range', MAX_TRACK_RANGE)
        self.declare_parameter('ground_z_min', GROUND_Z_MIN)
        self.declare_parameter('ground_z_max', GROUND_Z_MAX)
        self.declare_parameter('cluster_cell', CLUSTER_CELL)
        self.declare_parameter('track_gate_dist', TRACK_GATE_DIST)
        self.declare_parameter('track_max_misses', TRACK_MAX_MISSES)
        self.declare_parameter('track_min_hits', TRACK_MIN_HITS)
        self.declare_parameter('process_noise', PROCESS_NOISE)
        self.declare_parameter('measurement_noise', MEASUREMENT_NOISE)
        self.declare_parameter('min_extent', MIN_EXTENT)
        self.declare_parameter('extent_gain', EXTENT_GAIN)
        self.declare_parameter('marker_lifetime', MARKER_LIFETIME)
        self.declare_parameter('predicted_points_topic', '/obstacle_tracker/predicted_points')
        self.declare_parameter('predict_horizons', list(PREDICT_HORIZONS))
        self.declare_parameter('predict_ring_points', PREDICT_RING_POINTS)

        points_topic = self.get_parameter('points_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.max_track_range = self.get_parameter('max_track_range').value
        self.z_min = self.get_parameter('ground_z_min').value
        self.z_max = self.get_parameter('ground_z_max').value
        self.cell = self.get_parameter('cluster_cell').value
        self.gate_dist = self.get_parameter('track_gate_dist').value
        self.max_misses = self.get_parameter('track_max_misses').value
        self.min_hits = self.get_parameter('track_min_hits').value
        self.process_noise = self.get_parameter('process_noise').value
        self.measurement_noise = self.get_parameter('measurement_noise').value
        self.min_extent = self.get_parameter('min_extent').value
        self.extent_gain = self.get_parameter('extent_gain').value
        self.marker_life = self.get_parameter('marker_lifetime').value
        predicted_points_topic = self.get_parameter('predicted_points_topic').value
        self.predict_horizons = list(self.get_parameter('predict_horizons').value)
        self.predict_ring_points = self.get_parameter('predict_ring_points').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_track_time = None

        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_tracker/markers', 10)
        self.state_pub = self.create_publisher(String, '/obstacle_tracker/state', 10)
        self.predicted_points_pub = self.create_publisher(
            PointCloud2, predicted_points_topic, 1)
        self.create_subscription(PointCloud2, points_topic, self._on_cloud, 1)

        self.get_logger().info(
            f'obstacle_tracker_go2 started (points_topic={points_topic}, '
            f'base_frame={self.base_frame}, map_frame={self.map_frame}, '
            f'z_band=[{self.z_min},{self.z_max}]m, max_track_range={self.max_track_range}m, '
            f'track_gate={self.gate_dist}m, min_hits={self.min_hits})')

    # -- Point cloud -> clusters (base_frame, top-down, grid + flood-fill) --

    def _cluster(self, msg: PointCloud2):
        xs, ys = [], []
        for x, y, z in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            if self.z_min <= z <= self.z_max:
                xs.append(x)
                ys.append(y)
        if not xs:
            return []

        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        ix = np.floor(x / self.cell).astype(np.int64)
        iy = np.floor(y / self.cell).astype(np.int64)

        # Fold (ix, iy) into one integer key so np.unique can group points
        # into cells in a single vectorized pass, then accumulate the
        # per-cell moment sums needed for centroid + covariance-ellipse
        # extent later, without ever holding onto per-point data.
        keys = ix.astype(np.int64) * np.int64(2**21) + (iy + np.int64(2**20))
        uniq, inv = np.unique(keys, return_inverse=True)
        # Recover each unique cell's (ix, iy) from one occurrence in the
        # original arrays (any point in the cell gives the same ix/iy).
        first_idx = np.zeros(len(uniq), dtype=np.int64)
        first_idx[inv] = np.arange(len(keys))
        cell_ix = ix[first_idx]
        cell_iy = iy[first_idx]

        count = np.bincount(inv, minlength=len(uniq))
        sum_x = np.bincount(inv, weights=x, minlength=len(uniq))
        sum_y = np.bincount(inv, weights=y, minlength=len(uniq))
        sum_xx = np.bincount(inv, weights=x * x, minlength=len(uniq))
        sum_yy = np.bincount(inv, weights=y * y, minlength=len(uniq))
        sum_xy = np.bincount(inv, weights=x * y, minlength=len(uniq))

        cell_lookup = {(int(cell_ix[i]), int(cell_iy[i])): i for i in range(len(uniq))}

        # 8-connected flood-fill over occupied grid cells (same grid+BFS
        # idiom frontier_explorer_go2.py uses for frontier clustering).
        visited = set()
        clusters = []
        for key, i in cell_lookup.items():
            if key in visited:
                continue
            comp = [i]
            visited.add(key)
            stack = [key]
            while stack:
                cx0, cy0 = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb = (cx0 + dx, cy0 + dy)
                        if nb in cell_lookup and nb not in visited:
                            visited.add(nb)
                            comp.append(cell_lookup[nb])
                            stack.append(nb)

            n = float(count[comp].sum())
            cx = float(sum_x[comp].sum() / n)
            cy = float(sum_y[comp].sum() / n)
            cxx = float(sum_xx[comp].sum() / n) - cx * cx
            cyy = float(sum_yy[comp].sum() / n) - cy * cy
            cxy = float(sum_xy[comp].sum() / n) - cx * cy
            l1, l2, alpha = self._fit_extent(cxx, cyy, cxy)
            clusters.append({'x': cx, 'y': cy, 'count': int(n),
                              'l1': l1, 'l2': l2, 'alpha': alpha})
        return clusters

    def _fit_extent(self, cxx: float, cyy: float, cxy: float):
        """Ellipse (major/minor axis lengths + orientation) from a
        cluster's point covariance -- extended-object tracking's 'extent'.
        """
        trace = cxx + cyy
        disc = max(trace * trace / 4.0 - (cxx * cyy - cxy * cxy), 0.0)
        half = math.sqrt(disc)
        lam_major = trace / 2.0 + half
        lam_minor = trace / 2.0 - half

        alpha = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) if (cxx != cyy or cxy != 0.0) else 0.0
        l1 = max(2.0 * math.sqrt(max(lam_major, 0.0)), self.min_extent)
        l2 = max(2.0 * math.sqrt(max(lam_minor, 0.0)), self.min_extent)
        return l1, l2, alpha

    # -- Kalman-filter tracking (map_frame) ---------------------------------

    def _update_tracks(self, clusters_map: list, stamp) -> list:
        now = stamp.sec + stamp.nanosec * 1e-9
        dt = 0.0 if self.last_track_time is None else now - self.last_track_time
        self.last_track_time = now

        for t in self.tracks:
            t.predict(dt, self.process_noise)

        pairs = []
        for ti, t in enumerate(self.tracks):
            tx, ty = t.pos
            for ci, c in enumerate(clusters_map):
                d = math.hypot(tx - c['x'], ty - c['y'])
                if d <= self.gate_dist:
                    pairs.append((d, ti, ci))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_clusters = set(), set()
        for _, ti, ci in pairs:
            if ti in matched_tracks or ci in matched_clusters:
                continue
            matched_tracks.add(ti)
            matched_clusters.add(ci)
            c = clusters_map[ci]
            self.tracks[ti].update(c['x'], c['y'], self.measurement_noise, c['count'])
            self.tracks[ti].update_extent(c['l1'], c['l2'], c['alpha'], self.extent_gain)

        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.misses += 1

        for ci, c in enumerate(clusters_map):
            if ci not in matched_clusters:
                nt = Track(self.next_id, c['x'], c['y'], c['l1'], c['l2'], c['alpha'])
                nt.count = c['count']
                self.tracks.append(nt)
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.hits >= self.min_hits]

    # -- Callback ------------------------------------------------------------

    @staticmethod
    def _tf_xyyaw(tf):
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _on_cloud(self, msg: PointCloud2):
        clusters_base = self._cluster(msg)

        # "latest" (rclpy.time.Time()), matching frontier_explorer_go2.py's
        # own _robot_pose lookup -- NOT the cloud's exact capture timestamp.
        # An exact-timestamp lookup was tried first and reverted: this
        # project's headless CHAMP stack (full EKF state-estimation chain
        # driving a 12+ DOF legged robot) doesn't sustain low-latency
        # odom->base publishing even on an otherwise idle machine --
        # confirmed empirically up to ~1.7s of lag between a cloud's
        # capture time and when the matching TF became available. Requiring
        # an exact match there means normal, sustained lookup failures, not
        # just an occasional startup race.
        #
        # That leaves CHAMP's real gait-cycle body sway uncorrected here
        # (a couple degrees of orientation error, lever-arm amplified for
        # distant points -- confirmed separately: with "latest", distant
        # static outdoor scenery generated thousands of spurious re-IDed
        # tracks with 0.3-1.3 m/s "speed" in 30s). Since the amplification
        # scales with range, MAX_TRACK_RANGE below bounds the worst case
        # instead: it doesn't fix the sway, it keeps points close enough
        # that gate_dist tolerates the resulting jitter.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f'no {self.map_frame}->{self.base_frame} TF yet: {exc}',
                                    throttle_duration_sec=5.0)
            return

        tx, ty, yaw = self._tf_xyyaw(tf)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        clusters_map = []
        for c in clusters_base:
            if math.hypot(c['x'], c['y']) > self.max_track_range:
                continue
            mx = tx + cos_y * c['x'] - sin_y * c['y']
            my = ty + sin_y * c['x'] + cos_y * c['y']
            clusters_map.append({**c, 'x': mx, 'y': my})

        tracks = self._update_tracks(clusters_map, msg.header.stamp)
        self._publish(tracks, msg.header.stamp)
        self._publish_predicted_points(tracks, msg.header.stamp)

    # -- Predicted-points costmap feed --------------------------------------

    def _publish_predicted_points(self, tracks: list, stamp):
        """Publish a synthetic PointCloud2 of each track's extrapolated
        near-future position(s), in map_frame, as an extra observation
        source for the local costmap's voxel3d_layer (see
        config/go2_navigation.yaml) -- giving MPPI a bit of lead time on
        fast movers instead of only seeing where they are *right now* (all
        the raw /points source in that layer can offer, since VoxelLayer's
        marking there is ephemeral/rolling-window with no persistence).
        Wired as marking-only (no clearing) in that config: these points
        aren't a real sensor return, so raytrace-clearing from map_frame's
        origin along the point would incorrectly punch bogus cleared rays
        through the costmap.

        A moving obstacle only produces points once its Track exists AND
        has speed above the tracker's own velocity noise floor; a static
        or newly-spawned obstacle contributes none here (it's already
        covered by the same-frame raw /points source at zero lead time).
        """
        points = []
        for t in tracks:
            x, y = t.pos
            vx, vy = t.vel
            if math.hypot(vx, vy) < 0.05:
                continue
            for h in self.predict_horizons:
                px, py = x + vx * h, y + vy * h
                for i in range(self.predict_ring_points):
                    theta = 2.0 * math.pi * i / self.predict_ring_points
                    ex = 0.5 * t.l1 * math.cos(theta)
                    ey = 0.5 * t.l2 * math.sin(theta)
                    rx = ex * math.cos(t.alpha) - ey * math.sin(t.alpha)
                    ry = ex * math.sin(t.alpha) + ey * math.cos(t.alpha)
                    points.append((px + rx, py + ry, 0.3))

        header_frame = self.map_frame
        cloud = point_cloud2.create_cloud_xyz32(
            self._header(header_frame, stamp), points)
        self.predicted_points_pub.publish(cloud)

    @staticmethod
    def _header(frame_id, stamp):
        h = Header()
        h.frame_id = frame_id
        h.stamp = stamp
        return h

    # -- Publish ---------------------------------------------------------------

    def _publish(self, tracks: list, stamp):
        markers = MarkerArray()

        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        del_marker.header.frame_id = self.map_frame
        del_marker.header.stamp = stamp
        markers.markers.append(del_marker)

        for t in tracks:
            x, y = t.pos
            vx, vy = t.vel
            speed = math.hypot(vx, vy)

            extent = Marker()
            extent.header.frame_id = self.map_frame
            extent.header.stamp = stamp
            extent.ns = 'moving_obstacles'
            extent.id = t.id
            extent.type = Marker.CYLINDER
            extent.action = Marker.ADD
            extent.pose.position.x = x
            extent.pose.position.y = y
            extent.pose.position.z = 0.3
            extent.pose.orientation.z = math.sin(t.alpha / 2.0)
            extent.pose.orientation.w = math.cos(t.alpha / 2.0)
            extent.scale.x = t.l1
            extent.scale.y = t.l2
            extent.scale.z = 0.5
            extent.color.r = 1.0
            extent.color.g = 0.2
            extent.color.b = 0.0
            extent.color.a = 0.85
            extent.lifetime.sec = int(self.marker_life)
            extent.lifetime.nanosec = int((self.marker_life % 1) * 1e9)
            markers.markers.append(extent)

            if speed > 0.03:
                arrow = Marker()
                arrow.header.frame_id = self.map_frame
                arrow.header.stamp = stamp
                arrow.ns = 'obstacle_velocity'
                arrow.id = t.id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=x, y=y, z=0.3),
                    Point(x=x + vx, y=y + vy, z=0.3),
                ]
                arrow.scale.x = 0.08
                arrow.scale.y = 0.16
                arrow.color.r = 1.0
                arrow.color.g = 0.9
                arrow.color.b = 0.0
                arrow.color.a = 0.9
                arrow.lifetime.sec = int(self.marker_life)
                arrow.lifetime.nanosec = int((self.marker_life % 1) * 1e9)
                markers.markers.append(arrow)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = stamp
            label.ns = 'obstacle_labels'
            label.id = t.id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.7
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.9
            label.text = f'#{t.id} {speed:.2f}m/s'
            label.lifetime.sec = int(self.marker_life)
            label.lifetime.nanosec = int((self.marker_life % 1) * 1e9)
            markers.markers.append(label)

        self.marker_pub.publish(markers)

        state_msg = String()
        state_msg.data = json.dumps({
            'moving_obstacles': [
                {
                    'id': t.id,
                    'x': round(t.pos[0], 2),
                    'y': round(t.pos[1], 2),
                    'vx': round(t.vel[0], 2),
                    'vy': round(t.vel[1], 2),
                    'speed': round(math.hypot(*t.vel), 2),
                    'points': t.count,
                    'length': round(t.l1, 2),
                    'width': round(t.l2, 2),
                    'orientation': round(t.alpha, 3),
                }
                for t in tracks
            ]
        })
        self.state_pub.publish(state_msg)


def main():
    rclpy.init()
    node = ObstacleTrackerGo2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
