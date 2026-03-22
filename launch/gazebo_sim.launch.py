"""
Launch file: Gazebo simulation for quadruped robot dog.

Usage:
    ros2 launch launch/gazebo_sim.launch.py robot:=go1
    ros2 launch launch/gazebo_sim.launch.py robot:=spot
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='go1',
        description='Robot to simulate: go1, spot, mini_cheetah, anymal_b, anymal_c, mini_pupper'
    )

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    champ_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('champ_gazebo'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'robot': LaunchConfiguration('robot'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items()
    )

    return LaunchDescription([
        robot_arg,
        use_sim_time,
        champ_gazebo_launch,
    ])
