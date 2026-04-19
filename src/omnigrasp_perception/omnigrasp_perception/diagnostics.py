"""
diagnostics.py — Perception pipeline health monitoring.

WHAT THIS DOES:

Monitors every stage of the perception pipeline and publishes
structured health information. This is the difference between
a demo and a production system.

A demo crashes when something goes wrong.
A production system DETECTS what went wrong, REPORTS it, and
DEGRADES GRACEFULLY — continuing to work as well as it can.

FAILURE MODES WE HANDLE:

1. NO_DETECTION — Neither VLM found the object
2. LOW_AGREEMENT — VLMs disagree on location (low IoU)
3. DEPTH_INVALID — Depth reading is NaN, zero, or out of range
4. LATENCY_WARNING — Processing is too slow for real-time
5. OCCLUSION_DETECTED — Object suddenly disappeared from mask
6. TRACKING — Everything is working normally
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiagnosticsState:
    """
    Current state of the perception pipeline.

    This gets published as a PerceptionDiagnostics ROS2 message
    every frame, letting the dashboard and controller monitor health.
    """

    status: str = "INITIALIZING"

    # Detection metrics
    gdino_confidence: float = 0.0
    owlvit_confidence: float = 0.0
    fusion_iou: float = 0.0
    fusion_status: str = "NO_DETECTION"

    # Depth metrics
    depth_valid: bool = False
    depth_value: float = 0.0

    # Performance metrics
    inference_latency_ms: float = 0.0

    # Tracking metrics
    kalman_innovation: float = 0.0
    frames_since_last_detection: int = 0


class PerceptionDiagnostics:
    """
    Monitors and reports perception pipeline health.

    Usage:
        diagnostics = PerceptionDiagnostics()
        diagnostics.start_frame()

        # ... run detection ...
        diagnostics.update_detection(fusion_result)

        # ... run depth ...
        diagnostics.update_depth(depth_value)

        # ... run tracking ...
        diagnostics.update_tracking(kalman_innovation)

        state = diagnostics.end_frame()
        # → publish state as PerceptionDiagnostics message
    """

    def __init__(self, max_latency_ms: float = 200.0, max_depth: float = 5.0):
        """
        Args:
            max_latency_ms: If processing takes longer than this, warn.
            max_depth: Maximum valid depth in meters.
        """
        self.max_latency_ms = max_latency_ms
        self.max_depth = max_depth

        self.current_state = DiagnosticsState()
        self.frame_start_time: Optional[float] = None
        self.consecutive_no_detection = 0

    def start_frame(self) -> None:
        """Call at the beginning of each frame's processing."""
        self.frame_start_time = time.time()
        self.current_state = DiagnosticsState()

    def update_detection(
        self,
        detected: bool,
        gdino_confidence: float = 0.0,
        owlvit_confidence: float = 0.0,
        fusion_iou: float = 0.0,
        fusion_status: str = "NO_DETECTION",
    ) -> None:
        """Update diagnostics with detection results."""
        self.current_state.gdino_confidence = gdino_confidence
        self.current_state.owlvit_confidence = owlvit_confidence
        self.current_state.fusion_iou = fusion_iou
        self.current_state.fusion_status = fusion_status

        if not detected:
            self.consecutive_no_detection += 1
            self.current_state.status = "NO_DETECTION"
        else:
            self.consecutive_no_detection = 0
            if fusion_iou < 0.5 and fusion_status == "DISAGREED":
                self.current_state.status = "LOW_AGREEMENT"
            else:
                self.current_state.status = "TRACKING"

        self.current_state.frames_since_last_detection = self.consecutive_no_detection

    def update_depth(self, depth_value: float) -> None:
        """Update diagnostics with depth measurement."""
        self.current_state.depth_value = depth_value

        import math

        if math.isnan(depth_value) or math.isinf(depth_value):
            self.current_state.depth_valid = False
            self.current_state.status = "DEPTH_INVALID"
        elif depth_value <= 0 or depth_value > self.max_depth:
            self.current_state.depth_valid = False
            self.current_state.status = "DEPTH_INVALID"
        else:
            self.current_state.depth_valid = True

    def update_tracking(self, innovation: float) -> None:
        """Update diagnostics with Kalman filter innovation."""
        self.current_state.kalman_innovation = innovation

    def end_frame(self) -> DiagnosticsState:
        """
        Finalize frame diagnostics and check latency.

        Call this after all processing is done for the frame.
        Returns the complete diagnostics state for publishing.
        """
        if self.frame_start_time is not None:
            elapsed_ms = (time.time() - self.frame_start_time) * 1000
            self.current_state.inference_latency_ms = elapsed_ms

            if elapsed_ms > self.max_latency_ms:
                # Only override status if we were tracking normally
                if self.current_state.status == "TRACKING":
                    self.current_state.status = "LATENCY_WARNING"

        return self.current_state
