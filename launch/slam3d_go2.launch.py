"""3D LiDAR SLAM (RTAB-Map) + optional frontier exploration for the Go2.

Builds on launch/champ_go2_gazebo.launch.py (Gazebo + CHAMP gait engine
driving Go2 from /cmd_vel) by bridging the 16-channel gpu_lidar point cloud
(see urdf/go2_unitree/urdf/go2_gz.urdf.xacro's "lidar3d" sensor) and running
RTAB-Map's ICP-registered lidar SLAM on it -- a real 3D map (and a 2D
occupancy grid projected from it) instead of slam_toolbox's single-plane
/scan mapping (see launch/slam_go2.launch.py for that 2D path).

Runs in its own ROS_DOMAIN_ID/GZ_PARTITION by default so it doesn't cross
talk with other ROS2/Gazebo sessions on this machine (see README) -- override
if you actually want to share a domain with another terminal.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2/install/setup.bash
    ros2 launch launch/slam3d_go2.launch.py
    ros2 launch launch/slam3d_go2.launch.py headless:=true explore:=true
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = REPO / "training" / "envs" / "go2_gz_world_outdoor.sdf"
FRONTIER_EXPLORER = REPO / "scripts" / "frontier_explorer_go2.py"


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    world = LaunchConfiguration("world")
    explore = LaunchConfiguration("explore")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    gz_partition = LaunchConfiguration("gz_partition")

    champ_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(REPO / "launch" / "champ_go2_gazebo.launch.py")),
        launch_arguments={
            "headless": headless,
            "rviz": "false",  # this launch's own rviz below uses the SLAM view instead
            "world": world,
        }.items(),
    )

    # gpu_lidar always publishes LaserScan on its own <topic> and the actual
    # PointCloud2 on a nested "<topic>/points" -- see the comment on the
    # lidar3d sensor block in go2_gz.urdf.xacro.
    points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="points_bridge",
        output="screen",
        arguments=["/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"],
        remappings=[("/points/points", "/points")],
        parameters=[{"use_sim_time": True}],
    )

    # Started after CHAMP/the joint-trajectory adapter are up (t=13s in
    # champ_go2_gazebo.launch.py) and ground-truth odom has been publishing
    # for a few seconds (t=9s), so RTAB-Map's TF lookups don't race startup.
    rtabmap_slam = TimerAction(
        period=16.0,
        actions=[Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "frame_id": "base",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "subscribe_depth": False,
                "subscribe_rgb": False,
                "subscribe_scan_cloud": True,
                "approx_sync": True,
                "wait_for_transform": 0.3,
                # RTAB-Map params are strings.
                "Reg/Strategy": "1",           # ICP -- no camera for Vis registration
                "Icp/PointToPlane": "true",
                "Grid/Sensor": "0",            # occupancy grid from the lidar cloud
                "Grid/3D": "false",            # projected 2D grid for the frontier explorer
                "Grid/CellSize": "0.05",
                "Grid/RangeMax": "20.0",
                # 16-channel vertical resolution is sparse enough that
                # normal-based ground segmentation sprinkles spurious
                # "obstacle" cells on flat ground -- filter isolated points
                # before classification (same fix rosnav's slam_nav.launch.py
                # lidar_type:=3d path uses).
                "Grid/NoiseFilteringRadius": "0.1",
                "Grid/NoiseFilteringMinNeighbors": "5",
                "Mem/IncrementalMemory": "true",
            }],
            remappings=[("odom", "/odom"), ("scan_cloud", "/points")],
            arguments=["-d"],  # fresh database each run
        )],
    )

    # launch_ros Node requires a ROS2 package; this repo's scripts/ isn't one
    # (see training/launch/gazebo_rl.launch.py's __REPO_ROOT__ comment for the
    # same constraint), so run it as a plain process like that file does.
    frontier_explorer = TimerAction(
        period=20.0,
        condition=IfCondition(explore),
        actions=[ExecuteProcess(
            cmd=["python3", str(FRONTIER_EXPLORER), "--ros-args", "-p", "use_sim_time:=true"],
            output="screen",
        )],
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(REPO / "ros2" / "champ_navigation" / "rviz" / "slam.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                               description="Skip Gazebo/RViz GUIs (server + mapping only)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("world", default_value=str(DEFAULT_WORLD),
                               description="SDF world -- defaults to the outdoor demo world "
                                           "(trees/rocks scattered on open ground)"),
        DeclareLaunchArgument("explore", default_value="false",
                               description="Auto-start scripts/frontier_explorer_go2.py"),
        DeclareLaunchArgument(
            "ros_domain_id", default_value="157",
            description="ROS_DOMAIN_ID for this launch tree, isolated from other concurrent "
                        "ROS2 workspaces on this machine (quad_sdk's real-robot scripts hardcode "
                        "42; unset/default is 0). Override if you want to share a domain."),
        DeclareLaunchArgument(
            "gz_partition", default_value="quad3dslam",
            description="GZ_PARTITION for this launch tree's Gazebo transport, isolated from "
                        "any other gz sim instance running concurrently."),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        SetEnvironmentVariable("GZ_PARTITION", gz_partition),
        champ_gazebo,
        points_bridge,
        rtabmap_slam,
        frontier_explorer,
        rviz2,
    ])
