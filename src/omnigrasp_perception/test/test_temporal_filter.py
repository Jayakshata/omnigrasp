"""
Tests for the Kalman filter temporal smoother.

WHAT THESE TESTS PROVE:
- You understand temporal filtering (essential for real perception)
- The filter produces smoother output than raw input
- The filter tracks moving objects
- Confidence affects how much the filter trusts measurements
"""

import numpy as np
import pytest

from omnigrasp_perception.tracking.temporal_filter import TemporalFilter


class TestTemporalFilter:
    """Tests for the 6-state Kalman filter."""

    def test_first_measurement_accepted_directly(self):
        """First measurement should initialize state directly."""
        tf = TemporalFilter()
        measurement = np.array([1.0, 2.0, 3.0])
        result = tf.update(measurement)
        np.testing.assert_allclose(result, measurement, atol=1e-6)

    def test_stationary_object_converges(self):
        """
        Repeated measurements at the same position should converge.
        The output should be very close to the input.
        """
        tf = TemporalFilter()
        target = np.array([1.0, 2.0, 0.5])

        for _ in range(50):
            tf.predict()
            result = tf.update(target, confidence=0.9)

        np.testing.assert_allclose(result, target, atol=0.01)

    def test_noisy_input_produces_smoother_output(self):
        """
        Noisy measurements should produce smoother output.
        The variance of the output should be less than the input.
        """
        tf = TemporalFilter(measurement_noise=0.1)
        true_pos = np.array([1.0, 2.0, 0.5])

        rng = np.random.default_rng(seed=42)
        raw_positions = []
        filtered_positions = []

        for _ in range(100):
            noise = rng.normal(0, 0.05, size=3)
            measurement = true_pos + noise
            raw_positions.append(measurement.copy())

            tf.predict()
            result = tf.update(measurement, confidence=0.9)
            filtered_positions.append(result.copy())

        raw_var = np.var(raw_positions, axis=0).mean()
        filtered_var = np.var(filtered_positions, axis=0).mean()

        assert filtered_var < raw_var, (
            f"Filtered variance ({filtered_var:.6f}) should be less "
            f"than raw variance ({raw_var:.6f})"
        )

    def test_low_confidence_trusts_prediction_more(self):
        """
        With low confidence, the filter should trust its prediction
        more than the measurement. A sudden jump in measurement
        should be dampened.
        """
        tf = TemporalFilter()
        pos = np.array([1.0, 2.0, 0.5])

        # Initialize with stable position
        for _ in range(20):
            tf.predict()
            tf.update(pos, confidence=0.9)

        # Sudden jump with LOW confidence
        jumped = np.array([5.0, 5.0, 5.0])
        tf.predict()
        result = tf.update(jumped, confidence=0.1)

        # Result should be closer to original position than the jump
        dist_to_original = np.linalg.norm(result - pos)
        dist_to_jump = np.linalg.norm(result - jumped)
        assert dist_to_original < dist_to_jump, "Low confidence should dampen jumps"

    def test_innovation_magnitude(self):
        """Innovation should be near zero for expected measurements."""
        tf = TemporalFilter()
        pos = np.array([1.0, 2.0, 0.5])

        # Initialize
        tf.update(pos)

        # Consistent measurements → low innovation
        for _ in range(10):
            tf.predict()
            tf.update(pos, confidence=0.9)

        assert tf.get_innovation_magnitude() < 0.1

    def test_reset_clears_state(self):
        """After reset, filter should accept next measurement directly."""
        tf = TemporalFilter()
        tf.update(np.array([1.0, 2.0, 3.0]))
        tf.reset()

        new_pos = np.array([10.0, 20.0, 30.0])
        result = tf.update(new_pos)
        np.testing.assert_allclose(result, new_pos, atol=1e-6)
