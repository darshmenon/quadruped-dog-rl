"""
Launch file: View Unitree Go2 URDF in RViz2.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch launch/view_go2.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    urdf_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'urdf', 'go2_unitree', 'urdf', 'go2.urdf'
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
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
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2,
    ])
