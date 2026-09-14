#!/bin/bash
# ros2_ws 通用依赖安装脚本
# 创建独立 conda 环境，通过 ROS topic 与 IsaacLab 通信

set -e

echo "=========================================="
echo " Part 1: ROS2 apt 依赖"
echo "=========================================="

# ROS2 apt 包 (需要先 source humble)
# source /opt/ros/humble/setup.bash

sudo apt install -y \
    ros-humble-point-cloud-transport \
    ros-humble-tf-transformations \
    ros-humble-turtlebot3-msgs \
    ros-humble-grid-map \
    ros-humble-pcl-ros \
    ros-humble-image-transport \
    ros-humble-turtlebot3-teleop \
    ros-humble-turtlebot3-gazebo

echo "=========================================="
echo " Part 2: 创建 Conda 环境"
echo "=========================================="

# 初始化 conda (自动查找 conda 路径)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    # fallback: 尝试常见安装路径
    for _conda_path in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
        if [ -f "$_conda_path/etc/profile.d/conda.sh" ]; then
            . "$_conda_path/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# 创建新环境 (Python 3.10 兼容 ROS2 humble)
conda create -n ros2_env python=3.10 -y

echo "=========================================="
echo " Part 3: Pip 依赖 (在新环境中安装)"
echo "=========================================="

# 激活环境
conda activate ros2_env

pip install \
    catkin_pkg \
    "empy<4" \
    pyyaml \
    lark \
    cupy-cuda12x \
    scipy \
    shapely \
    ruamel.yaml \
    transforms3d \
    simple-parsing \
    scikit-image \
    networkx \
    matplotlib \
    numpy \
    opencv-python

echo "=========================================="
echo " Part 4: 安装 PyTorch (CUDA 12.8)"
pip install \
    torch==2.7.0 \
    torchaudio \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128 

echo "=========================================="
echo " 安装完成"
echo ""
echo " 编译步骤:"
echo "   conda activate ros2_env"
echo "   source /opt/ros/humble/setup.bash   # 重要: 用 humble，不是 ros_py311"
echo "   colcon build --symlink-install"
echo "=========================================="
