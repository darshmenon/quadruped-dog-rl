#!/usr/bin/env python3
"""Launch Go2 in room world with moving obstacle + optional mover script."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

REPO = Path(__file__).resolve().parents[1]

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(REPO / "launch" / "champ_go2_gazebo.launch.py")),
            launch_arguments={
                "world": str(REPO / "training" / "envs" / "go2_gz_world_moving.sdf"),
                "headless": "false",
            }.items(),
        ),
        ExecuteProcess(
            cmd=["python3", str(REPO / "scripts" / "moving_obstacle_gz.py")],
            output="screen",
        ),
    ])
