"""
Tests for the pinhole camera model.

WHAT THESE TESTS PROVE:
- You understand the pinhole camera model (THE core CV concept)
- You can implement deprojection from scratch
- You verify 3D accuracy against known ground truth
- You handle edge cases (invalid depth, image boundaries)

These tests are the strongest signal to a CV interviewer that you
understand 3D vision, not just 2D detection.
"""

import numpy as np
import pytest

from omnigrasp_perception.geometry.camera_model import PinholeCamera, CameraIntrinsics


class TestDeprojection:
    """Tests for 2D pixel + depth → 3D point conversion."""

    def setup_method(self):
        """Create a standard camera for testing."""
        self.intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480
        )
        self.camera = PinholeCamera(self.intrinsics)

    def test_center_pixel_projects_to_origin(self):
        """
        A pixel at the image center with any depth should deproject
        to (0, 0, depth) in camera frame.

        WHY: The principal point (cx, cy) is where the optical axis
        hits the sensor. A 3D point on the optical axis at distance Z
        should project to (cx, cy). Reversing this: (cx, cy, Z) → (0, 0, Z).
        """
        point = self.camera.deproject(320.0, 240.0, 1.0)
        assert point is not None
        assert abs(point[0]) < 1e-6, f"X should be 0 at center, got {point[0]}"
        assert abs(point[1]) < 1e-6, f"Y should be 0 at center, got {point[1]}"
        assert abs(point[2] - 1.0) < 1e-6, f"Z should be 1.0, got {point[2]}"

    def test_known_3d_point(self):
        """
        Verify deprojection against a manually calculated 3D point.

        Pixel (420, 240) at depth 2.0:
        X = (420 - 320) * 2.0 / 525 = 100 * 2.0 / 525 ≈ 0.3810
        Y = (240 - 240) * 2.0 / 525 = 0
        Z = 2.0
        """
        point = self.camera.deproject(420.0, 240.0, 2.0)
        expected_x = (420.0 - 320.0) * 2.0 / 525.0
        assert point is not None
        assert abs(point[0] - expected_x) < 1e-4
        assert abs(point[1]) < 1e-6
        assert abs(point[2] - 2.0) < 1e-6

    def test_roundtrip_project_deproject(self):
        """
        Project a 3D point to 2D, then deproject back.
        The result should match the original 3D point.

        This is the ultimate test of the camera model:
        3D → 2D → 3D should be identity (within floating point precision).
        """
        original = np.array([0.5, -0.3, 1.5])
        pixel = self.camera.project(original)
        assert pixel is not None
        recovered = self.camera.deproject(pixel[0], pixel[1], original[2])
        assert recovered is not None
        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_zero_depth_returns_none(self):
        """Depth of zero is invalid — cannot deproject."""
        result = self.camera.deproject(320.0, 240.0, 0.0)
        assert result is None

    def test_negative_depth_returns_none(self):
        """Negative depth is physically impossible."""
        result = self.camera.deproject(320.0, 240.0, -1.0)
        assert result is None

    def test_nan_depth_returns_none(self):
        """NaN depth (sensor failure) should be handled gracefully."""
        result = self.camera.deproject(320.0, 240.0, float("nan"))
        assert result is None

    def test_point_behind_camera_no_projection(self):
        """A point behind the camera (negative Z) cannot be projected."""
        result = self.camera.project(np.array([0.0, 0.0, -1.0]))
        assert result is None

    def test_different_depths_scale_linearly(self):
        """
        Doubling the depth should double the X and Y coordinates.

        This is a fundamental property of the pinhole model:
        deprojection is linear in depth.
        """
        point_1m = self.camera.deproject(400.0, 300.0, 1.0)
        point_2m = self.camera.deproject(400.0, 300.0, 2.0)
        assert point_1m is not None and point_2m is not None
        assert abs(point_2m[0] - 2 * point_1m[0]) < 1e-6
        assert abs(point_2m[1] - 2 * point_1m[1]) < 1e-6


class TestMaskDeprojection:
    """Tests for deprojecting a mask to a 3D point cloud."""

    def setup_method(self):
        self.intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480
        )
        self.camera = PinholeCamera(self.intrinsics)

    def test_mask_produces_3d_points(self):
        """A mask with valid depth should produce 3D points."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:250, 300:350] = True  # 50x50 region

        depth = np.full((480, 640), 1.0, dtype=np.float32)

        points = self.camera.deproject_mask_to_points(mask, depth, step=1)
        assert len(points) > 0
        assert points.shape[1] == 3  # Each point has X, Y, Z

    def test_empty_mask_returns_empty(self):
        """An empty mask should return no points."""
        mask = np.zeros((480, 640), dtype=bool)
        depth = np.full((480, 640), 1.0, dtype=np.float32)

        points = self.camera.deproject_mask_to_points(mask, depth)
        assert len(points) == 0

    def test_invalid_depth_filtered(self):
        """Points with NaN or zero depth should be excluded."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[240, 320] = True  # Single pixel

        depth = np.full((480, 640), 0.0, dtype=np.float32)  # All zero = invalid

        points = self.camera.deproject_mask_to_points(mask, depth, step=1)
        assert len(points) == 0
