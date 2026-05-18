#!/bin/bash
# One-shot ROS 2 Humble + IL pipeline setup inside Ubuntu 22.04 WSL.
# Run from the repo root inside WSL:
#   bash scripts/setup_ros2_wsl.sh
#
# What it does:
#   1. Sets locale to UTF-8
#   2. Adds ROS 2 apt repository
#   3. Installs ros-humble-ros-base + colcon + rosdep
#   4. Installs Python deps: torch (CUDA), pybullet, fastapi, etc.
#   5. Builds il_pipeline_msgs and il_pipeline via colcon
#   6. Writes a convenience source script

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS_ROOT="$HOME/il_ws"

echo "=== [1/6] Locale setup ==="
sudo apt-get update -q
sudo apt-get install -y -q locales
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "=== [2/6] ROS 2 Humble apt source ==="
sudo apt-get install -y -q software-properties-common curl
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update -q

echo "=== [3/6] Install ROS 2 Humble base + build tools ==="
sudo apt-get install -y -q \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-argcomplete
sudo rosdep init 2>/dev/null || true
rosdep update

echo "=== [4/6] Install Python deps ==="
pip3 install --upgrade pip
# CUDA 12.8 wheel — works with CUDA 13.0 driver
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install numpy pyarrow pybullet "fastapi>=0.110" "uvicorn[standard]" \
    "pydantic>=2.0" websockets datasets huggingface-hub lerobot pytest ruff

echo "=== [5/6] Build ROS 2 packages ==="
mkdir -p "$WS_ROOT/src"
# Symlink both ROS 2 packages from the repo into the workspace
ln -sfn "$REPO_ROOT/il_pipeline_msgs" "$WS_ROOT/src/il_pipeline_msgs"
ln -sfn "$REPO_ROOT/il_pipeline"      "$WS_ROOT/src/il_pipeline"

source /opt/ros/humble/setup.bash
cd "$WS_ROOT"
colcon build --packages-select il_pipeline_msgs
source install/setup.bash
colcon build --packages-select il_pipeline
source install/setup.bash

echo "=== [6/6] Shell integration ==="
# Add to .bashrc so every new WSL session is ready to go
grep -qxF "source /opt/ros/humble/setup.bash"       ~/.bashrc || \
    echo "source /opt/ros/humble/setup.bash"         >> ~/.bashrc
grep -qxF "source $WS_ROOT/install/setup.bash"       ~/.bashrc || \
    echo "source $WS_ROOT/install/setup.bash"         >> ~/.bashrc

echo ""
echo "Setup complete. To start a new session:"
echo "  source ~/.bashrc"
echo ""
echo "Quick verification:"
echo "  ros2 run il_pipeline pybullet_robot_node"
