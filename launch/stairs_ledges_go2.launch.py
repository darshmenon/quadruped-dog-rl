"""One-command Go2 stairs / ledges demo (CHAMP walk).

Usage:
  ros2 launch launch/stairs_ledges_go2.launch.py course:=stairs
  ros2 launch launch/stairs_ledges_go2.launch.py course:=ledges
  ros2 launch launch/stairs_ledges_go2.launch.py course:=ledges headless:=true
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


REPO = Path(__file__).resolve().parents[1]
COURSES = {
    "stairs": REPO / "training" / "envs" / "go2_gz_world_stairs.sdf",
    "ledges": REPO / "training" / "envs" / "go2_gz_world_ledges.sdf",
}


def _setup(context, *args, **kwargs):
    course = LaunchConfiguration("course").perform(context).strip().lower()
    if course not in COURSES:
        raise RuntimeError(
            f"Unknown course={course!r}; choose one of: {', '.join(sorted(COURSES))}"
        )
    world = str(COURSES[course])
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(REPO / "launch" / "champ_go2_gazebo.launch.py")
            ),
            launch_arguments={
                "world": world,
                "headless": LaunchConfiguration("headless"),
                "rviz": LaunchConfiguration("rviz"),
                "state_estimation": LaunchConfiguration("state_estimation"),
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "course",
            default_value="stairs",
            description="stairs = solid stair curriculum; ledges = platforms/gaps/hollow stairs",
        ),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "state_estimation",
            default_value="false",
            description="CHAMP odom EKF (set true if you will layer Nav2).",
        ),
        OpaqueFunction(function=_setup),
    ])
