# Interfaces

This folder is for custom ROS2 message, service, and action definitions used in this project.

## Structure

```
interfaces/
├── msg/          # Custom ROS2 messages (.msg)
├── srv/          # Custom ROS2 services (.srv)
└── action/       # Custom ROS2 actions (.action)
```

## Existing standard interfaces used

- `sensor_msgs/JointState` — joint positions, velocities, efforts
- `geometry_msgs/Twist` — velocity commands (linear + angular)
- `nav_msgs/Odometry` — robot odometry
- `std_msgs/Float32MultiArray` — raw policy actions/observations

## Adding custom interfaces

1. Create your `.msg` / `.srv` / `.action` file in the correct subfolder
2. Add to `CMakeLists.txt` under `rosidl_generate_interfaces`
3. Build with `colcon build --packages-select interfaces`
