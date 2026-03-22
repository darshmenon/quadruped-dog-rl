"""
Launch file: Spawn Unitree Go2 in Gazebo Garden (gz-sim7) with ROS2 bridge.

Usage:
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ros2 launch launch/gazebo_go2.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'urdf', 'go2_unitree', 'urdf', 'go2.urdf'
)

with open(URDF_PATH, 'r') as f:
    robot_description = f.read()


def generate_launch_description():

    # Gazebo Garden (gz sim)
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', 'empty.sdf'],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # Spawn the robot into Gazebo Garden
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                name='spawn_go2',
                output='screen',
                arguments=[
                    '-name', 'go2',
                    '-topic', 'robot_description',
                    '-z', '0.5',
                ]
            )
        ]
    )

    # Bridge: Gazebo <-> ROS2 topics
    bridge = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gz_ros2_bridge',
                output='screen',
                arguments=[
                    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                    '/model/go2/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
                    '/model/go2/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                ]
            )
        ]
    )

    # RViz2 for visualization alongside Gazebo
    rviz2 = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz2,
    ])
