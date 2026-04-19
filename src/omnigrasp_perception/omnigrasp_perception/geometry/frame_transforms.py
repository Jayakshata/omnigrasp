"""
frame_transforms.py — Coordinate frame transformations.

WHY FRAMES MATTER:

The camera sees the world from its own perspective (camera frame).
The robot's arm operates in the robot's perspective (base frame).
A point at (0.1, 0.2, 0.5) in camera frame is NOT the same as
(0.1, 0.2, 0.5) in robot frame — the camera is mounted at a
specific position and angle relative to the robot.

We need to transform coordinates from camera frame → robot frame
so the robot knows where to actually move its arm.

THE MATH:

P_target = R × P_camera + t

Where:
  P_camera = 3D point in camera's coordinate system
  R = 3×3 rotation matrix (camera's orientation relative to robot)
  t = 3×1 translation vector (camera's position relative to robot)
  P_target = same point, expressed in robot's coordinate system
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transform:
    """
    A rigid transform (rotation + translation) between two coordinate frames.

    rotation: 3x3 rotation matrix
    translation: 3x1 translation vector (in meters)
    """

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)


class FrameTransformer:
    """
    Handles coordinate frame transformations.

    In our system, we have three frames:
    1. Camera frame — origin at the camera, Z pointing forward
    2. World frame — fixed reference frame of the simulation
    3. Robot base frame — origin at the robot's base

    The transforms between these frames are known because we placed
    the camera and robot in Isaac Sim at specific positions.
    """

    def __init__(
        self,
        camera_to_world: Transform = None,
        world_to_robot: Transform = None,
    ):
        if camera_to_world is None:
            # Default: camera is at (0, 0, 1) looking down at the table
            # This is a common setup for tabletop manipulation
            self.camera_to_world = Transform(
                rotation=np.eye(3),  # Identity = same orientation
                translation=np.array([0.0, 0.0, 1.0]),
            )
        else:
            self.camera_to_world = camera_to_world

        if world_to_robot is None:
            # Default: robot base is at world origin
            self.world_to_robot = Transform(
                rotation=np.eye(3),
                translation=np.array([0.0, 0.0, 0.0]),
            )
        else:
            self.world_to_robot = world_to_robot

    def camera_to_robot_frame(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera frame to robot base frame.

        Two-step transform:
        1. Camera → World: apply camera_to_world transform
        2. World → Robot: apply world_to_robot transform

        Args:
            point_camera: [X, Y, Z] in camera frame

        Returns:
            [X, Y, Z] in robot base frame
        """
        # Step 1: Camera → World
        point_world = self.apply_transform(point_camera, self.camera_to_world)

        # Step 2: World → Robot
        point_robot = self.apply_transform(point_world, self.world_to_robot)

        return point_robot

    @staticmethod
    def apply_transform(point: np.ndarray, transform: Transform) -> np.ndarray:
        """
        Apply a rigid transform to a 3D point.

        P_out = R × P_in + t

        This is the fundamental operation of coordinate transforms.
        Rotation changes the orientation, translation shifts the position.

        Args:
            point: [X, Y, Z] in source frame
            transform: Rotation + translation to apply

        Returns:
            [X, Y, Z] in target frame
        """
        return transform.rotation @ point + transform.translation

    @staticmethod
    def rotation_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Create a rotation matrix from Euler angles (in radians).

        Roll  = rotation around X axis (tilting left/right)
        Pitch = rotation around Y axis (tilting forward/backward)
        Yaw   = rotation around Z axis (turning left/right)

        This is useful when you know the camera's orientation in
        human-readable angles rather than a rotation matrix.
        """
        # Rotation around X (roll)
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        # Rotation around Y (pitch)
        ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        # Rotation around Z (yaw)
        rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation: Rz × Ry × Rx (applied right to left)
        return rz @ ry @ rx
