"""
camera_model.py — Pinhole camera model with deprojection.

THIS IS THE MOST IMPORTANT FILE FOR CV INTERVIEWS.

Every perception engineer must understand how cameras map 3D world
points to 2D pixel coordinates, and how to reverse that mapping.

THE PINHOLE CAMERA MODEL:

A camera projects 3D points onto a 2D sensor through a tiny hole (the pinhole).
The math is:

    u = fx * X/Z + cx       (horizontal pixel coordinate)
    v = fy * Y/Z + cy       (vertical pixel coordinate)

Where:
    (X, Y, Z) = 3D point in camera coordinate frame (meters)
    (u, v)    = 2D pixel coordinate
    fx, fy    = focal length in pixels (how "zoomed in" the camera is)
    cx, cy    = principal point (where the optical axis hits the sensor,
                usually near the image center)

DEPROJECTION (what we do — going backwards):

Given pixel (u, v) and depth Z:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z (from depth camera)

This converts a 2D detection back into a 3D position that the
robot can reach toward.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    These describe the camera's internal properties — how it
    converts 3D points to 2D pixels. They're determined by the
    camera's lens and sensor, not by where the camera is placed.

    In Isaac Sim, these come from the /camera/camera_info topic.
    In the real world, you'd get these from camera calibration.
    """

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels) — usually width/2
    cy: float  # Principal point y (pixels) — usually height/2
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    distortion_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5)
    )  # [k1, k2, p1, p2, k3]


class PinholeCamera:
    """
    Implements the pinhole camera model with optional lens distortion.

    This class is intentionally implemented FROM SCRATCH — not using
    cv2.projectPoints or cv2.undistortPoints. This demonstrates that
    you understand the math, not just the API.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Args:
            intrinsics: Camera intrinsic parameters
        """
        self.intrinsics = intrinsics

        # Build the 3x3 intrinsic matrix K
        # This matrix encodes the camera's projection in a single operation
        #
        #     ┌ fx   0   cx ┐
        # K = │  0  fy   cy │
        #     └  0   0    1 ┘
        self.K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ]
        )

        # Inverse of K (used for deprojection)
        self.K_inv = np.linalg.inv(self.K)

    def project(self, point_3d: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project a 3D point to 2D pixel coordinates.

        3D → 2D (how a camera sees the world)

        Args:
            point_3d: [X, Y, Z] in camera frame (meters)

        Returns:
            (u, v) pixel coordinates, or None if behind camera
        """
        x, y, z = point_3d

        # Can't project points behind the camera (z <= 0)
        if z <= 0:
            return None

        # Basic projection: divide by depth, scale by focal length
        u = self.intrinsics.fx * x / z + self.intrinsics.cx
        v = self.intrinsics.fy * y / z + self.intrinsics.cy

        return (u, v)

    def deproject(self, u: float, v: float, depth: float) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates + depth to 3D point.

        2D + depth → 3D (reversing the camera projection)

        THIS IS THE KEY FUNCTION. Understand this for interviews.

        The math:
            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth

        Intuition:
        - (u - cx) is how far the pixel is from the image center horizontally
        - Dividing by fx converts from pixels to angular offset
        - Multiplying by depth converts from angle to meters
        - Same logic for Y

        Args:
            u: Horizontal pixel coordinate
            v: Vertical pixel coordinate
            depth: Distance from camera in meters (from depth sensor)

        Returns:
            [X, Y, Z] in camera frame (meters), or None if depth invalid
        """
        # Validate depth
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None

        # Optionally undistort the pixel coordinates first
        if np.any(self.intrinsics.distortion_coeffs != 0):
            u, v = self.undistort_point(u, v)

        # Deprojection
        x = (u - self.intrinsics.cx) * depth / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth / self.intrinsics.fy
        z = depth

        return np.array([x, y, z])

    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """
        Remove lens distortion from pixel coordinates.

        Real camera lenses cause distortion — straight lines in the
        world appear curved in the image. The standard model uses
        5 coefficients: k1, k2 (radial), p1, p2 (tangential), k3 (radial).

        The distortion model:
            x_normalized = (u - cx) / fx
            y_normalized = (v - cy) / fy
            r² = x² + y²

            x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
                        + 2*p1*x*y + p2*(r² + 2*x²)
            y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶)
                        + p1*(r² + 2*y²) + 2*p2*x*y

        We need the INVERSE: given distorted coords, find undistorted.
        We use iterative refinement (Newton's method) because there's
        no closed-form inverse.
        """
        k1, k2, p1, p2, k3 = self.intrinsics.distortion_coeffs

        # Normalize pixel coordinates
        x = (u - self.intrinsics.cx) / self.intrinsics.fx
        y = (v - self.intrinsics.cy) / self.intrinsics.fy

        # Iterative undistortion (5 iterations is usually enough)
        x0, y0 = x, y
        for _ in range(5):
            r2 = x0 * x0 + y0 * y0
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

            # Tangential distortion
            dx = 2 * p1 * x0 * y0 + p2 * (r2 + 2 * x0 * x0)
            dy = p1 * (r2 + 2 * y0 * y0) + 2 * p2 * x0 * y0

            # Refined estimate
            x0 = (x - dx) / radial
            y0 = (y - dy) / radial

        # Convert back to pixel coordinates
        u_undistorted = x0 * self.intrinsics.fx + self.intrinsics.cx
        v_undistorted = y0 * self.intrinsics.fy + self.intrinsics.cy

        return u_undistorted, v_undistorted

    def deproject_mask_to_points(
        self, mask: np.ndarray, depth_image: np.ndarray, step: int = 4
    ) -> np.ndarray:
        """
        Deproject all masked pixels to 3D points (point cloud).

        Instead of deprojecting a single pixel, this deprojections
        every pixel where the mask is True. The result is a set of
        3D points on the object's surface — a "point cloud".

        We use this for:
        - Surface normal estimation (fit a plane to the points)
        - Object size estimation
        - Grasp pose computation

        Args:
            mask: Binary mask (H, W), True = object pixel
            depth_image: Depth map (H, W), float32, meters
            step: Sample every Nth pixel (for speed — we don't need ALL pixels)

        Returns:
            Nx3 array of 3D points in camera frame
        """
        # Find all mask pixel coordinates
        ys, xs = np.where(mask)

        if len(ys) == 0:
            return np.empty((0, 3))

        # Subsample for speed (every 'step'th pixel)
        ys = ys[::step]
        xs = xs[::step]

        # Look up depth for each masked pixel
        depths = depth_image[ys, xs]

        # Filter out invalid depths
        valid = (depths > 0) & (~np.isnan(depths)) & (~np.isinf(depths))
        xs = xs[valid]
        ys = ys[valid]
        depths = depths[valid]

        if len(depths) == 0:
            return np.empty((0, 3))

        # Vectorized deprojection (much faster than a for loop)
        x_3d = (xs.astype(float) - self.intrinsics.cx) * depths / self.intrinsics.fx
        y_3d = (ys.astype(float) - self.intrinsics.cy) * depths / self.intrinsics.fy
        z_3d = depths

        return np.column_stack([x_3d, y_3d, z_3d])
