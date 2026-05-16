"""
ROS 2 launch file for the IL pipeline.

Brings up the simulated robot (pybullet_robot_node) for development, the data
logger and inference nodes, and the FastAPI web layer.

Usage:
    ros2 launch il_pipeline pipeline.launch.py
    ros2 launch il_pipeline pipeline.launch.py use_camera:=true

On the workstation, replace pybullet_robot_node with the actual MyBotShop
platform's controller — the data logger and inference nodes are unaffected
because they speak only the documented ROS 2 contracts.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("il_pipeline")
    config_root = os.path.join(pkg_share, "config")
    # Fallback to the repo-local configs directory when running pre-install
    if not os.path.isdir(config_root):
        config_root = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

    use_camera = LaunchConfiguration("use_camera")
    use_sim = LaunchConfiguration("use_sim")
    api_port = LaunchConfiguration("api_port")

    return LaunchDescription([
        DeclareLaunchArgument("use_camera", default_value="false",
                              description="Enable camera capture in data logger and inference."),
        DeclareLaunchArgument("use_sim", default_value="true",
                              description="Start the PyBullet simulator stand-in for the robot."),
        DeclareLaunchArgument("api_port", default_value="8011",
                              description="FastAPI port — set to 9000 to mirror MyBotShop's default."),

        Node(
            package="il_pipeline",
            executable="pybullet_robot_node",
            name="pybullet_robot_node",
            output="screen",
            condition=None,  # always start by default; gate with `--use-sim false` in real deployments
            parameters=[{"gui": False}],
        ),

        Node(
            package="il_pipeline",
            executable="data_logger_node",
            name="data_logger_node",
            parameters=[os.path.join(config_root, "data_logger.yaml")],
            output="screen",
        ),

        Node(
            package="il_pipeline",
            executable="inference_node",
            name="inference_node",
            parameters=[os.path.join(config_root, "inference.yaml")],
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "uvicorn",
                "il_pipeline.web_api.app:app",
                "--host", "0.0.0.0",
                "--port", api_port,
            ],
            output="screen",
        ),
    ])
