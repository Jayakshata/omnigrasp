"""
detection_fusion.py — Multi-model detection fusion.

THIS IS THE KEY MODULE FOR CV INTERVIEWS.

It takes detection results from two (or more) models, compares
them, and produces a single fused detection that's more reliable
than either model alone.

THE FUSION STRATEGY:
1. Both models detect (IoU > threshold) → AGREED
   Use confidence-weighted average of boxes
2. Only one model detects → SINGLE
   Use that model's result but with lower confidence
3. Neither model detects → NO_DETECTION
   Report failure, hold last known state

WHY MULTI-MODEL FUSION?
- No single model is perfect
- Two models agreeing = much higher reliability
- Disagreement is a useful signal (something is ambiguous)
- This is exactly how self-driving car perception works
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from omnigrasp_perception.detectors.base_detector import DetectionResult


@dataclass
class FusedDetection:
    """
    Result of fusing multiple detector outputs.

    Contains the fused box, combined confidence, agreement metrics,
    and which strategy was used — all of which feed into our
    Detection.msg and PerceptionDiagnostics.msg.
    """

    box: Optional[np.ndarray] = None  # Fused bounding box
    confidence: float = 0.0  # Combined confidence
    detected: bool = False
    gdino_confidence: float = 0.0
    owlvit_confidence: float = 0.0
    fusion_iou: float = 0.0  # How much the models agreed
    fusion_status: str = "NO_DETECTION"  # AGREED, SINGLE_GDINO, SINGLE_OWLVIT, NO_DETECTION


class DetectionFusion:
    """
    Fuses detections from multiple models using IoU-based agreement.
    """

    def __init__(self, iou_threshold: float = 0.5, single_model_penalty: float = 0.7):
        """
        Args:
            iou_threshold: Minimum IoU to consider models as "agreeing".
                           0.5 is the standard threshold used in PASCAL VOC
                           and most detection benchmarks.
            single_model_penalty: Multiply confidence by this when only
                                  one model detects. 0.7 means we're 30%
                                  less confident when models don't agree.
        """
        self.iou_threshold = iou_threshold
        self.single_model_penalty = single_model_penalty

    def fuse(self, detection_a: DetectionResult, detection_b: DetectionResult) -> FusedDetection:
        """
        Fuse two detection results into a single output.

        This method handles all four scenarios:
        1. Both detected, boxes agree (IoU ≥ threshold)
        2. Both detected, boxes disagree (IoU < threshold)
        3. Only one detected
        4. Neither detected
        """
        a_detected = detection_a.detected
        b_detected = detection_b.detected

        # Case 4: Neither model detected anything
        if not a_detected and not b_detected:
            return FusedDetection(
                detected=False,
                fusion_status="NO_DETECTION",
                gdino_confidence=detection_a.confidence,
                owlvit_confidence=detection_b.confidence,
            )

        # Case 3: Only one model detected
        if a_detected and not b_detected:
            return FusedDetection(
                box=detection_a.box.copy(),
                confidence=detection_a.confidence * self.single_model_penalty,
                detected=True,
                gdino_confidence=detection_a.confidence,
                owlvit_confidence=0.0,
                fusion_iou=0.0,
                fusion_status=f"SINGLE_{detection_a.model_name.upper()}",
            )

        if not a_detected and b_detected:
            return FusedDetection(
                box=detection_b.box.copy(),
                confidence=detection_b.confidence * self.single_model_penalty,
                detected=True,
                gdino_confidence=0.0,
                owlvit_confidence=detection_b.confidence,
                fusion_iou=0.0,
                fusion_status=f"SINGLE_{detection_b.model_name.upper()}",
            )

        # Both models detected — compute IoU to check agreement
        iou = self.calculate_iou(detection_a.box, detection_b.box)

        # Case 1: Models agree (IoU ≥ threshold)
        if iou >= self.iou_threshold:
            fused_box = self.weighted_average_box(
                detection_a.box,
                detection_a.confidence,
                detection_b.box,
                detection_b.confidence,
            )
            fused_confidence = max(detection_a.confidence, detection_b.confidence)

            return FusedDetection(
                box=fused_box,
                confidence=fused_confidence,
                detected=True,
                gdino_confidence=detection_a.confidence,
                owlvit_confidence=detection_b.confidence,
                fusion_iou=iou,
                fusion_status="AGREED",
            )

        # Case 2: Models disagree (IoU < threshold)
        # Use the higher-confidence detection but penalize
        if detection_a.confidence >= detection_b.confidence:
            better = detection_a
        else:
            better = detection_b

        return FusedDetection(
            box=better.box.copy(),
            confidence=better.confidence * self.single_model_penalty,
            detected=True,
            gdino_confidence=detection_a.confidence,
            owlvit_confidence=detection_b.confidence,
            fusion_iou=iou,
            fusion_status="DISAGREED",
        )

    @staticmethod
    def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        THIS IS THE MOST-ASKED CV INTERVIEW QUESTION.

        IoU = Area of Overlap / Area of Union

        Where Union = Area_A + Area_B - Overlap (to avoid double-counting)

        Args:
            box_a: [x_min, y_min, x_max, y_max]
            box_b: [x_min, y_min, x_max, y_max]

        Returns:
            IoU value between 0.0 (no overlap) and 1.0 (identical boxes)
        """
        # Find the coordinates of the intersection rectangle
        # The intersection's top-left is the MAX of the two top-lefts
        # The intersection's bottom-right is the MIN of the two bottom-rights
        x_min_inter = max(box_a[0], box_b[0])
        y_min_inter = max(box_a[1], box_b[1])
        x_max_inter = min(box_a[2], box_b[2])
        y_max_inter = min(box_a[3], box_b[3])

        # If the boxes don't overlap, intersection dimensions will be negative
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return 0.0

        # Intersection area
        inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

        # Individual areas
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # Union = sum of areas minus intersection (avoid double-counting overlap)
        union_area = area_a + area_b - inter_area

        # Avoid division by zero
        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)

    @staticmethod
    def weighted_average_box(
        box_a: np.ndarray,
        confidence_a: float,
        box_b: np.ndarray,
        confidence_b: float,
    ) -> np.ndarray:
        """
        Compute confidence-weighted average of two bounding boxes.

        The more confident model's box gets more weight. If model A
        is 90% confident and model B is 60% confident:
        - Weight A = 0.9 / (0.9 + 0.6) = 0.6
        - Weight B = 0.6 / (0.9 + 0.6) = 0.4
        - Fused box is 60% model A + 40% model B
        """
        total = confidence_a + confidence_b
        if total <= 0:
            return box_a.copy()

        weight_a = confidence_a / total
        weight_b = confidence_b / total

        return weight_a * box_a + weight_b * box_b
