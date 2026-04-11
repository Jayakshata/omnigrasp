#!/bin/bash
set -e

# Source ROS2 Humble base installation
source /opt/ros/humble/setup.bash

# Source our custom workspace (if it's been built)
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Execute whatever command was passed to the container
exec "$@"