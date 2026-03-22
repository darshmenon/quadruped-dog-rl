"""
Launch file: Deploy trained RL policy to real robot or simulation.

Usage:
    ros2 launch launch/policy_deploy.launch.py robot:=go2 checkpoint:=/path/to/policy.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='go2',
        description='Robot to deploy on: go1, go2'
    )

    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint',
        default_value='',
        description='Path to trained policy checkpoint (.pt file)'
    )

    policy_node = Node(
        package='quadruped_dog_rl',
        executable='policy_runner',
        name='policy_runner',
        output='screen',
        parameters=[{
            'robot': LaunchConfiguration('robot'),
            'checkpoint': LaunchConfiguration('checkpoint'),
        }]
    )

    return LaunchDescription([
        robot_arg,
        checkpoint_arg,
        policy_node,
    ])
