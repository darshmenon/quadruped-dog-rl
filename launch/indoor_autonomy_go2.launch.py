"""One-command indoor autonomy demo for the Go2.

Starts the CHAMP-backed Go2 in the bounded room world with RTAB-Map 3D SLAM,
frontier exploration, obstacle tracking, and RViz enabled by default.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO = Path(__file__).resolve().parents[1]
ROOM_WORLD = REPO / "training" / "envs" / "go2_gz_world_room.sdf"


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("foxglove", default_value="false",
                              description="Start foxglove_bridge if installed"),
        DeclareLaunchArgument(
            "locomotion", default_value="champ",
            description="Locomotion backend: 'champ' (default, generic gait engine -- "
                        "not physics-tuned for Go2, see README) or 'nmpc' (Quad-SDK "
                        "global/local planner + NMPC controller, the backend verified "
                        "to actually walk cleanly; uses ros2/quad_sdk/quad_simulator/"
                        "quad_sim_scripts/worlds/go2_room.sdf, a copy of "
                        "go2_gz_world_room.sdf renamed to Quad-SDK's expected "
                        "'default' world name).",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(REPO / "launch" / "slam3d_go2.launch.py")),
            launch_arguments={
                "locomotion": LaunchConfiguration("locomotion"),
                "world": str(ROOM_WORLD),
                "nmpc_world": "go2_room.sdf",
                "headless": LaunchConfiguration("headless"),
                "rviz": LaunchConfiguration("rviz"),
                "explore": "true",
                "track_obstacles": "true",
            }.items(),
        ),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("foxglove")),
        ),
    ])
