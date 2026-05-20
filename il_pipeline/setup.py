"""ament_python setup for the il_pipeline ROS 2 package."""

from glob import glob

from setuptools import setup

package_name = "il_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.nodes",
        f"{package_name}.dataset",
        f"{package_name}.training",
        f"{package_name}.inference",
        f"{package_name}.web_api",
    ],
    data_files=[
        # ament index marker — required so ros2 pkg can find this package
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        # Package manifest
        (f"share/{package_name}", ["package.xml"]),
        # Launch files
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Naeem Zain Uddin",
    maintainer_email="naeemzainuddin5253@gmail.com",
    description="Imitation learning pipeline for the MyBotShop robotic webserver",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Each entry point exposes a `ros2 run il_pipeline <name>` command.
            f"data_logger_node = {package_name}.nodes.data_logger_node:main",
            f"inference_node = {package_name}.nodes.inference_node:main",
            f"pybullet_robot_node = {package_name}.nodes.pybullet_robot_node:main",
        ],
    },
)
