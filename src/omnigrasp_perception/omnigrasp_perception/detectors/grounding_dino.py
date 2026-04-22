"""
grounding_dino.py — Grounding DINO object detector.

Wraps the Grounding DINO model from Hugging Face Transformers
into our BaseDetector interface. When running locally (no GPU),
it uses a MOCK that simulates detection by looking for colored
regions in the image. On RunPod with GPU, we swap in the real model.
"""

import numpy as np

from omnigrasp_perception.detectors.base_detector import BaseDetector, DetectionResult


class GroundingDINODetector(BaseDetector):
    """
    Grounding DINO detector with mock fallback.

    use_mock=True  → color-based mock (CPU, no GPU needed)
    use_mock=False → real Grounding DINO from HuggingFace (GPU required)
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load the model or initialize mock."""
        if self.use_mock:
            pass
        else:
            from transformers import pipeline

            self.model = pipeline(
                "zero-shot-object-detection",
                model="IDEA-Research/grounding-dino-base",
                device=0,
            )

    def detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        if self.use_mock:
            return self._mock_detect(image, prompt)
        else:
            return self._real_detect(image, prompt)

    def _mock_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """Mock detection: finds regions that differ from the background."""
        height, width = image.shape[:2]
        bg_color = image.mean(axis=(0, 1))
        diff = np.abs(image.astype(float) - bg_color)
        mask = diff.mean(axis=2) > 50
        nonzero_count = np.sum(mask)

        if nonzero_count < 100:
            return DetectionResult(detected=False, confidence=0.0, model_name="gdino")

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        area_ratio = nonzero_count / (height * width)
        confidence = min(0.95, 0.5 + area_ratio * 10)

        return DetectionResult(
            box=np.array([float(x_min), float(y_min), float(x_max), float(y_max)]),
            confidence=confidence,
            detected=True,
            model_name="gdino",
        )

    def _real_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """Real Grounding DINO detection using HuggingFace pipeline."""
        from PIL import Image

        pil_image = Image.fromarray(image)
        results = self.model(pil_image, candidate_labels=[prompt])

        # Filter by confidence threshold
        filtered = [r for r in results if r["score"] > 0.2]

        if not filtered:
            return DetectionResult(detected=False, confidence=0.0, model_name="gdino")

        best = max(filtered, key=lambda x: x["score"])
        box = best["box"]

        return DetectionResult(
            box=np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], dtype=float),
            confidence=best["score"],
            detected=True,
            model_name="gdino",
        )
