"""
Launch file: Visualize quadruped robot in RViz2.

Usage:
    ros2 launch launch/rviz_view.launch.py robot:=go1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='go1',
        description='Robot model to visualize'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        robot_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2,
    ])
