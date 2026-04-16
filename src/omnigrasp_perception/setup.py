"""
setup.py — Tells ROS2 how to install this Python package.

This is like a manifest that says:
- What the package is called
- Where to find the code
- Which scripts should become runnable commands (entry points)

The 'console_scripts' section is KEY — it registers your ROS2 nodes
so you can run them with 'ros2 run omnigrasp_perception perception_node'
"""
from setuptools import find_packages, setup

package_name = "omnigrasp_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # These tell ROS2 where to find package metadata
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jayakshata",
    maintainer_email="jayakshata@example.com",
    description="Multi-model perception stack for OmniGrasp",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Format: 'command_name = package.module:function'
            # This means: 'ros2 run omnigrasp_perception perception_node'
            # will call the 'main()' function in perception_node.py
            "perception_node = omnigrasp_perception.perception_node:main",
        ],
    },
)