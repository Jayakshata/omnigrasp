from setuptools import find_packages, setup

package_name = "omnigrasp_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jayakshata",
    maintainer_email="jayakshata@example.com",
    description="RL-based robot control for OmniGrasp",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rl_controller_node = omnigrasp_control.rl_controller_node:main",
        ],
    },
)
