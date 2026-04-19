"""
grasp_pose_estimator.py — Estimate 6-DOF grasp pose from mask + depth.

WHAT IS A 6-DOF GRASP POSE?

6-DOF = 6 Degrees of Freedom:
- 3 for position: (x, y, z) — WHERE to place the gripper
- 3 for orientation: (roll, pitch, yaw) — HOW to orient the gripper

Just publishing (x, y, z) means the robot reaches the right spot
but might approach at the wrong angle. Publishing a full 6-DOF
pose means the gripper approaches correctly — aligned with the
object's shape and approaching along the surface normal.

HOW WE ESTIMATE THE POSE:

1. POSITION: Centroid of the 3D point cloud (center of the object)

2. APPROACH DIRECTION (surface normal):
   Fit a plane to the 3D point cloud using least squares.
   The plane's normal vector = the direction to approach from.

3. GRIPPER ALIGNMENT (principal axis):
   Run PCA on the 2D mask to find the object's long axis.
   Align the gripper's fingers with this axis.
"""

from typing import Optional

import numpy as np


class GraspPoseEstimator:
    """Estimates 6-DOF grasp poses from segmentation masks and depth data."""

    def estimate(self, points_3d: np.ndarray, mask: np.ndarray) -> Optional[dict]:
        """
        Estimate a grasp pose from a 3D point cloud and 2D mask.

        Args:
            points_3d: Nx3 array of 3D points on the object surface
            mask: Binary mask (H, W) of the object

        Returns:
            Dictionary with:
                - position: [x, y, z] grasp center
                - approach_direction: [nx, ny, nz] surface normal
                - principal_axis: [ax, ay] object orientation in image
                - grasp_width: estimated object width in meters
            Or None if estimation fails
        """
        if len(points_3d) < 10:
            return None

        # Step 1: Position = centroid of 3D points
        position = np.mean(points_3d, axis=0)

        # Step 2: Approach direction = surface normal
        normal = self.estimate_surface_normal(points_3d)

        # Step 3: Principal axis from 2D mask (PCA)
        principal_axis = self.compute_principal_axis(mask)

        # Step 4: Estimate grasp width from point cloud spread
        grasp_width = self.estimate_grasp_width(points_3d)

        return {
            "position": position,
            "approach_direction": normal,
            "principal_axis": principal_axis,
            "grasp_width": grasp_width,
        }

    @staticmethod
    def estimate_surface_normal(points: np.ndarray) -> np.ndarray:
        """
        Estimate the surface normal by fitting a plane to the point cloud.

        THE MATH:

        A plane in 3D is defined by: ax + by + cz = d
        The normal vector is [a, b, c].

        To find the best-fit plane, we use least squares:
        1. Center the points (subtract mean)
        2. Compute the covariance matrix
        3. The eigenvector with the SMALLEST eigenvalue is the normal
           (it's the direction of least variance — perpendicular to the surface)

        This is equivalent to PCA in 3D, where the third component
        (least variance) is the normal direction.
        """
        if len(points) < 3:
            return np.array([0.0, 0.0, 1.0])  # Default: pointing up

        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # Covariance matrix
        cov = centered.T @ centered  # 3x3 matrix

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Smallest eigenvalue's eigenvector = surface normal
        # eigh returns eigenvalues in ascending order, so index 0 is smallest
        normal = eigenvectors[:, 0]

        # Ensure normal points toward camera (positive Z direction)
        if normal[2] < 0:
            normal = -normal

        return normal

    @staticmethod
    def compute_principal_axis(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the principal axis of the object from its 2D mask using PCA.

        THE INTUITION:

        PCA finds the direction of maximum spread in the mask pixels.
        For a bolt, this is along its length. For a washer, both
        directions are similar (it's round).

        HOW IT WORKS:
        1. Get all (x, y) coordinates where mask is True
        2. Center them (subtract mean)
        3. Compute 2x2 covariance matrix
        4. The eigenvector with the LARGEST eigenvalue is the principal axis

        The ratio of eigenvalues tells us the object's elongation:
        - ratio ≈ 1 → round (washer, ball)
        - ratio >> 1 → elongated (bolt, pen, handle)
        """
        ys, xs = np.where(mask)

        if len(xs) < 10:
            return None

        # Center the coordinates
        cx = np.mean(xs)
        cy = np.mean(ys)
        xs_centered = xs - cx
        ys_centered = ys - cy

        # 2x2 covariance matrix
        cov = np.cov(xs_centered, ys_centered)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Largest eigenvalue's eigenvector = principal axis
        # eigh returns ascending order, so index 1 (last) is largest
        principal = eigenvectors[:, 1]

        return principal

    @staticmethod
    def estimate_grasp_width(points: np.ndarray) -> float:
        """
        Estimate the width of the object for gripper opening.

        Uses the range of points projected onto the direction
        perpendicular to the surface (the "thickness" of the object).

        Returns width in meters.
        """
        if len(points) < 3:
            return 0.05  # Default 5cm

        # Use the spread perpendicular to the dominant direction
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # PCA in 3D
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Project onto the axis of least spread (thinnest dimension)
        thin_axis = eigenvectors[:, 0]
        projections = centered @ thin_axis

        width = float(np.max(projections) - np.min(projections))
        return max(width, 0.01)  # At least 1cm
