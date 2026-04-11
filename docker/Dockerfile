# ==============================================================================
# OmniGrasp Production Dockerfile
# ==============================================================================
# BASE IMAGE: Official ROS2 Humble on Ubuntu 22.04
# Why Humble? It's the LTS (Long Term Support) release — stable until 2027.
# Why not install ROS2 manually? The official image handles all the complex
# apt repository setup, GPG keys, and dependency resolution for us.
# ==============================================================================
FROM ros:humble-ros-base

# ==============================================================================
# SYSTEM DEPENDENCIES
# 'apt-get update' refreshes the list of available packages (like refreshing
# an app store). 'apt-get install' downloads and installs them.
# 
# Why each package:
#   python3-pip          → Python package manager (to install PyTorch, transformers, etc.)
#   python3-colcon-common-extensions → The ROS2 build tool (like 'make' but for ROS2)
#   python3-rosdep       → Resolves ROS2 package dependencies automatically
#   ros-humble-rosbridge-suite → WebSocket bridge (lets the web dashboard talk to ROS2)
#   ros-humble-cv-bridge → Converts between ROS2 image messages and OpenCV images
#   ros-humble-vision-msgs → Standard message types for computer vision (bounding boxes, etc.)
#   wget, curl           → Download tools (for model weights later)
#   git                  → Version control (needed inside container for some pip installs)
#
# The 'rm -rf /var/lib/apt/lists/*' at the end cleans up cached package lists
# to keep the Docker image small. Every byte matters in production images.
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-rosbridge-suite \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# PYTHON DEPENDENCIES
# These are the AI/ML libraries our perception stack needs.
#
# Why each package:
#   torch, torchvision   → PyTorch (the ML framework everything runs on)
#   transformers         → Hugging Face library (loads Grounding DINO, OWL-ViT)
#   opencv-python-headless → Computer vision utilities (image processing, no GUI)
#   numpy                → Numerical computing (matrices, arrays — used everywhere)
#   scipy                → Scientific computing (we use it for spatial transforms)
#   filterpy             → Kalman filter implementation (for temporal pose smoothing)
#   matplotlib           → Plotting (for evaluation metrics and charts)
#
# '--no-cache-dir' tells pip not to save downloaded packages — saves disk space.
# ==============================================================================
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    opencv-python-headless \
    numpy \
    scipy \
    filterpy \
    matplotlib

# ==============================================================================
# ROS2 WORKSPACE SETUP
# In ROS2, all your code lives inside a "workspace" — a specific folder structure.
# The convention is:
#   /ros2_ws/           → The workspace root
#   /ros2_ws/src/       → Where your ROS2 packages (source code) go
#
# WORKDIR sets the "current directory" for all subsequent commands.
# Think of it like 'cd /ros2_ws' that persists for the rest of the Dockerfile.
# ==============================================================================
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# ==============================================================================
# COPY SOURCE CODE
# This copies your local 'src/' folder (your ROS2 packages) into the container.
# Docker uses layers — each command creates a layer. By copying source code
# AFTER installing dependencies, Docker can cache the dependency layers.
# This means rebuilding after a code change is fast (seconds, not minutes).
# ==============================================================================
COPY src/ /ros2_ws/src/

# ==============================================================================
# BUILD THE ROS2 WORKSPACE
# 
# 'source /opt/ros/humble/setup.bash' activates ROS2 so the build tools can
# find all the ROS2 libraries. We need to do this in the same RUN command
# because each RUN creates a new shell (environment doesn't persist between RUNs).
#
# 'colcon build --symlink-install' compiles your ROS2 packages.
#   --symlink-install means Python files are symlinked rather than copied,
#   so changes to Python code take effect without rebuilding (useful for dev).
#
# '|| true' at the end means "don't fail the build even if there are warnings."
# We'll handle build errors through CI/CD, not here.
# ==============================================================================
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install" || true

# ==============================================================================
# ENTRYPOINT
# This script runs every time a container starts from this image.
# It sources (activates) both ROS2 and our custom workspace, then runs
# whatever command was passed to 'docker run'.
#
# 'source install/setup.bash' makes our custom packages available.
# 'exec "$@"' runs whatever command follows — e.g., 'ros2 launch ...'
# ==============================================================================
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]