"""
ROS 2 launch file: bring up the data logger + inference nodes + web API
together with their parameter files.

Usage:
    ros2 launch configs/launch_pipeline.py
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CONFIGS = Path(__file__).resolve().parent


def generate_launch_description():
    use_camera = LaunchConfiguration("use_camera")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_camera",
            default_value="false",
            description="Enable camera capture in data logger and inference.",
        ),

        Node(
            package="il_pipeline",
            executable="data_logger_node",
            name="data_logger_node",
            parameters=[str(CONFIGS / "data_logger.yaml")],
            output="screen",
        ),

        Node(
            package="il_pipeline",
            executable="inference_node",
            name="inference_node",
            parameters=[str(CONFIGS / "inference.yaml")],
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "uvicorn",
                "il_pipeline.web_api.app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            output="screen",
            cwd=os.path.expanduser("~"),
        ),
    ])
