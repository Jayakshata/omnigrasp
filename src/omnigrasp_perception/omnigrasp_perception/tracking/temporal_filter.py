"""
temporal_filter.py — Kalman filter for smoothing target pose over time.

THE PROBLEM:

VLM detections jitter frame-to-frame. Even if the object is perfectly
still, the bounding box shifts by a few pixels each frame due to:
- Model non-determinism
- Image noise
- Slight rounding in the detection pipeline

These small pixel shifts become real-world position noise after
deprojection. If the robot follows raw detections, its arm jerks.

THE SOLUTION: KALMAN FILTER

The Kalman filter maintains a BELIEF about where the object is,
combining two sources of information:

1. PREDICTION: "Based on physics (constant velocity model), the
   object should be HERE now"
2. MEASUREMENT: "The perception pipeline says the object is THERE"

The filter blends these optimally based on how much we trust each.
The result is a smooth, stable trajectory.

OUR STATE VECTOR: [x, y, z, vx, vy, vz]
- Position (x, y, z) in meters
- Velocity (vx, vy, vz) in meters/second

Even for a stationary object, tracking velocity lets the filter
detect when something starts moving and react faster.
"""

from typing import Optional, Tuple

import numpy as np


class TemporalFilter:
    """
    6-state Kalman filter for 3D position tracking.

    Uses a constant-velocity model: assumes the object moves at
    roughly constant speed between frames. This is a good assumption
    for robotic manipulation where objects are either stationary or
    being slowly moved.
    """

    def __init__(
        self, dt: float = 0.1, process_noise: float = 0.01, measurement_noise: float = 0.05
    ):
        """
        Args:
            dt: Time between frames in seconds (0.1 = 10 Hz camera)
            process_noise: How much we expect the object to randomly accelerate.
                           Higher = less trust in prediction, more responsive.
                           Lower = smoother output, slower to react to real motion.
            measurement_noise: How noisy we expect detections to be.
                               Higher = less trust in measurement, smoother output.
                               Lower = more trust in measurement, more responsive.
        """
        self.dt = dt

        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # State covariance matrix (uncertainty in our state estimate)
        # Start with high uncertainty — we don't know where the object is yet
        self.P = np.eye(6) * 1.0

        # State transition matrix F (constant velocity model)
        # x_new = x_old + vx * dt
        # y_new = y_old + vy * dt
        # z_new = z_old + vz * dt
        # vx_new = vx_old  (velocity stays the same)
        # vy_new = vy_old
        # vz_new = vz_old
        self.F = np.eye(6)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

        # Measurement matrix H (we observe position only, not velocity)
        # measurement = [x, y, z] = H × state
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1  # observe x
        self.H[1, 1] = 1  # observe y
        self.H[2, 2] = 1  # observe z

        # Process noise covariance Q (how much random motion we expect)
        self.Q = np.eye(6) * process_noise
        self.Q[3, 3] *= 10  # Velocity changes more than position
        self.Q[4, 4] *= 10
        self.Q[5, 5] *= 10

        # Measurement noise covariance R (how noisy detections are)
        self.R_base = np.eye(3) * measurement_noise

        # Track whether we've received our first measurement
        self.initialized = False

        # Innovation (surprise) — useful for diagnostics
        self.last_innovation = np.zeros(3)

    def predict(self) -> np.ndarray:
        """
        Prediction step: estimate where the object SHOULD be now.

        Uses the constant-velocity model:
        new_position = old_position + velocity * dt

        Also increases our uncertainty (P grows) because we're less
        sure about the prediction the further into the future we go.

        Returns:
            Predicted position [x, y, z]
        """
        # Predict state: x_predicted = F × x_old
        self.state = self.F @ self.state

        # Predict covariance: P_predicted = F × P × F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:3].copy()

    def update(self, measurement: np.ndarray, confidence: float = 1.0) -> np.ndarray:
        """
        Update step: incorporate a new measurement.

        Blends the prediction with the measurement based on their
        relative uncertainties. High confidence = trust measurement more.

        THE KALMAN GAIN (K):
        K determines the blend ratio. If K is large, we trust the
        measurement. If K is small, we trust the prediction.

        K = P × H^T × (H × P × H^T + R)^(-1)

        Args:
            measurement: Observed position [x, y, z] from perception
            confidence: Detection confidence (0-1). Lower confidence
                        increases measurement noise, making the filter
                        trust the prediction more.

        Returns:
            Filtered position [x, y, z]
        """
        if not self.initialized:
            # First measurement: initialize state directly
            self.state[:3] = measurement
            self.state[3:] = 0  # Zero initial velocity
            self.initialized = True
            self.last_innovation = np.zeros(3)
            return measurement.copy()

        # Adjust measurement noise based on confidence
        # Low confidence → high noise → trust prediction more
        confidence_clamped = max(confidence, 0.1)  # Prevent division by zero
        R = self.R_base / confidence_clamped

        # Innovation: difference between measurement and prediction
        # This is how "surprised" we are by the measurement
        y = measurement - self.H @ self.state
        self.last_innovation = y.copy()

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state: blend prediction with measurement
        self.state = self.state + K @ y

        # Update covariance: we're now more certain
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get the estimated velocity [vx, vy, vz] in m/s."""
        return self.state[3:].copy()

    def get_innovation_magnitude(self) -> float:
        """
        Get the magnitude of the last innovation (surprise).

        Large innovation = the measurement was far from the prediction.
        This is useful for diagnostics:
        - Consistently large innovation → detections are very noisy
        - Sudden large innovation → object might have moved quickly
        - Near-zero innovation → tracking is stable
        """
        return float(np.linalg.norm(self.last_innovation))

    def reset(self) -> None:
        """Reset the filter to uninitialized state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.initialized = False
        self.last_innovation = np.zeros(3)
