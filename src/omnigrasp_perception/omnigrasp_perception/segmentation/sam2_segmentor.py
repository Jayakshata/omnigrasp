"""
sam2_segmentor.py — SAM 2 instance segmentation.

WHAT THIS DOES:

Takes a bounding box (from our detection fusion) and produces a
pixel-perfect binary mask of the object. The mask tells us the
EXACT SHAPE of the object, not just a rectangle around it.

WHY MASKS MATTER MORE THAN BOXES FOR GRASPING:

A bounding box around a bolt and a washer might be the same size.
But their SHAPES are completely different:
- Bolt: long and thin → grasp along its length
- Washer: round and flat → grasp from above

The mask gives us shape information. From shape + depth, we can
estimate the optimal grasp approach.

MOCK MODE:
Like the detectors, we use a mock locally that generates masks
from the bounding box. On RunPod, we swap in real SAM 2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SegmentationResult:
    """Output from the segmentation module."""

    mask: Optional[np.ndarray] = None  # Binary mask (H, W), True = object
    centroid: Optional[Tuple[float, float]] = None  # (cx, cy) in pixels
    area: int = 0  # Number of object pixels
    valid: bool = False


class SAM2Segmentor:
    """
    SAM 2 segmentor with mock fallback.

    Real SAM 2 takes an image + a prompt (bounding box, point, or
    previous mask) and outputs a high-quality segmentation mask.

    Our mock creates an elliptical mask inside the bounding box,
    which is a reasonable approximation for testing the downstream
    geometry pipeline.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None
        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Load real SAM 2 model (requires GPU)."""
        # from sam2.build_sam import build_sam2
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # self.model = SAM2ImagePredictor(build_sam2("sam2_hiera_small"))
        pass

    def segment(self, image: np.ndarray, box: np.ndarray) -> SegmentationResult:
        """
        Segment the object inside the bounding box.

        Args:
            image: RGB image, shape (H, W, 3)
            box: Bounding box [x_min, y_min, x_max, y_max]

        Returns:
            SegmentationResult with binary mask and metadata
        """
        if self.use_mock:
            return self._mock_segment(image, box)
        else:
            return self._real_segment(image, box)

    def _mock_segment(self, image: np.ndarray, box: np.ndarray) -> SegmentationResult:
        """
        Create an elliptical mask inside the bounding box.

        WHY ELLIPTICAL?
        A real SAM 2 mask follows the object's contour, which is
        roughly elliptical for most manufactured parts (bolts, washers,
        handles). A rectangle would be unrealistically perfect.
        An ellipse is a better approximation for testing.

        HOW IT WORKS:
        1. Extract box coordinates
        2. Calculate the center and radii of an ellipse that fits the box
        3. For each pixel in the image, check if it's inside the ellipse
        4. Return the binary mask
        """
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = box.astype(int)

        # Clamp to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)

        # Ellipse parameters
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        radius_x = (x_max - x_min) / 2.0
        radius_y = (y_max - y_min) / 2.0

        if radius_x <= 0 or radius_y <= 0:
            return SegmentationResult(valid=False)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        ellipse = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2
        mask = ellipse <= 1.0

        # Calculate centroid (center of mass of the mask)
        mask_pixels = np.where(mask)
        if len(mask_pixels[0]) == 0:
            return SegmentationResult(valid=False)

        centroid_y = float(np.mean(mask_pixels[0]))
        centroid_x = float(np.mean(mask_pixels[1]))

        return SegmentationResult(
            mask=mask,
            centroid=(centroid_x, centroid_y),
            area=int(np.sum(mask)),
            valid=True,
        )

    def _real_segment(self, image: np.ndarray, box: np.ndarray) -> SegmentationResult:
        """Real SAM 2 segmentation (requires GPU)."""
        raise NotImplementedError("Real SAM 2 requires GPU. Set use_mock=True.")
