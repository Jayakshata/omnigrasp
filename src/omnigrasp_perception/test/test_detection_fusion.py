"""
Tests for the detection fusion module.

WHAT THESE TESTS PROVE:
- You understand IoU calculation (the #1 CV interview question)
- Your fusion logic handles all 4 scenarios correctly
- Edge cases are covered (no overlap, identical boxes, single model)

TESTING PATTERN:
Each test follows: Arrange → Act → Assert
1. Arrange: set up known inputs
2. Act: call the function
3. Assert: verify the output matches expected values
"""

import numpy as np
import pytest

from omnigrasp_perception.detectors.base_detector import DetectionResult
from omnigrasp_perception.detectors.detection_fusion import DetectionFusion


class TestIoUCalculation:
    """Tests for IoU (Intersection over Union) — the core CV metric."""

    def test_identical_boxes_returns_one(self):
        """Two identical boxes should have IoU = 1.0 (perfect overlap)."""
        box = np.array([100.0, 100.0, 200.0, 200.0])
        iou = DetectionFusion.calculate_iou(box, box)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_no_overlap_returns_zero(self):
        """Two boxes that don't touch should have IoU = 0.0."""
        box_a = np.array([0.0, 0.0, 50.0, 50.0])
        box_b = np.array([100.0, 100.0, 200.0, 200.0])
        iou = DetectionFusion.calculate_iou(box_a, box_b)
        assert iou == 0.0, f"Expected IoU=0.0 for non-overlapping boxes, got {iou}"

    def test_partial_overlap(self):
        """Two partially overlapping boxes should have 0 < IoU < 1."""
        box_a = np.array([0.0, 0.0, 100.0, 100.0])  # area = 10000
        box_b = np.array([50.0, 50.0, 150.0, 150.0])  # area = 10000
        # Overlap: [50,50] to [100,100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429
        iou = DetectionFusion.calculate_iou(box_a, box_b)
        expected = 2500.0 / 17500.0
        assert abs(iou - expected) < 1e-4, f"Expected IoU≈{expected:.4f}, got {iou:.4f}"

    def test_one_box_inside_another(self):
        """A small box completely inside a large box."""
        big = np.array([0.0, 0.0, 200.0, 200.0])  # area = 40000
        small = np.array([50.0, 50.0, 100.0, 100.0])  # area = 2500
        # Overlap = 2500 (entire small box)
        # Union = 40000 + 2500 - 2500 = 40000
        # IoU = 2500/40000 = 0.0625
        iou = DetectionFusion.calculate_iou(big, small)
        expected = 2500.0 / 40000.0
        assert abs(iou - expected) < 1e-4

    def test_touching_edges_returns_zero(self):
        """Boxes that share an edge but don't overlap have IoU = 0."""
        box_a = np.array([0.0, 0.0, 100.0, 100.0])
        box_b = np.array([100.0, 0.0, 200.0, 100.0])  # Starts where A ends
        iou = DetectionFusion.calculate_iou(box_a, box_b)
        assert iou == 0.0


class TestDetectionFusion:
    """Tests for the multi-model fusion logic."""

    def setup_method(self):
        """Create a fusion instance for each test."""
        self.fusion = DetectionFusion(iou_threshold=0.5, single_model_penalty=0.7)

    def test_both_models_agree(self):
        """When both models detect with high IoU, status should be AGREED."""
        det_a = DetectionResult(
            box=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.9,
            detected=True,
            model_name="gdino",
        )
        det_b = DetectionResult(
            box=np.array([105.0, 95.0, 205.0, 195.0]),  # Slightly offset
            confidence=0.85,
            detected=True,
            model_name="owlvit",
        )
        result = self.fusion.fuse(det_a, det_b)

        assert result.detected is True
        assert result.fusion_status == "AGREED"
        assert result.confidence > 0.8
        assert result.fusion_iou > 0.5

    def test_neither_model_detects(self):
        """When neither model detects, status should be NO_DETECTION."""
        det_a = DetectionResult(detected=False, confidence=0.0, model_name="gdino")
        det_b = DetectionResult(detected=False, confidence=0.0, model_name="owlvit")
        result = self.fusion.fuse(det_a, det_b)

        assert result.detected is False
        assert result.fusion_status == "NO_DETECTION"

    def test_only_gdino_detects(self):
        """When only GDINO detects, use it but with penalty."""
        det_a = DetectionResult(
            box=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.9,
            detected=True,
            model_name="gdino",
        )
        det_b = DetectionResult(detected=False, confidence=0.0, model_name="owlvit")
        result = self.fusion.fuse(det_a, det_b)

        assert result.detected is True
        assert result.fusion_status == "SINGLE_GDINO"
        assert result.confidence == pytest.approx(0.9 * 0.7, abs=1e-4)

    def test_models_disagree(self):
        """When models detect in different locations, status should be DISAGREED."""
        det_a = DetectionResult(
            box=np.array([0.0, 0.0, 50.0, 50.0]),
            confidence=0.8,
            detected=True,
            model_name="gdino",
        )
        det_b = DetectionResult(
            box=np.array([400.0, 400.0, 500.0, 500.0]),  # Far away
            confidence=0.7,
            detected=True,
            model_name="owlvit",
        )
        result = self.fusion.fuse(det_a, det_b)

        assert result.detected is True
        assert result.fusion_status == "DISAGREED"
        assert result.fusion_iou < 0.5

    def test_weighted_average_box(self):
        """Higher-confidence model should have more weight in the fused box."""
        box_a = np.array([100.0, 100.0, 200.0, 200.0])
        box_b = np.array([120.0, 120.0, 220.0, 220.0])
        # conf_a = 0.9, conf_b = 0.1 → fused box should be very close to box_a
        fused = DetectionFusion.weighted_average_box(box_a, 0.9, box_b, 0.1)
        assert fused[0] < 105  # x_min closer to box_a's 100 than box_b's 120
