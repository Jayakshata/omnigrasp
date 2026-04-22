"""
owl_vit.py — OWL-ViT object detector.

Same interface as Grounding DINO but with a different underlying model.
In mock mode, adds slight random noise to simulate a second independent model.
"""

import numpy as np

from omnigrasp_perception.detectors.base_detector import BaseDetector, DetectionResult


class OWLViTDetector(BaseDetector):
    """
    OWL-ViT detector with mock fallback.

    OWL-ViT uses CLIP-style embedding similarity instead of
    cross-attention (like GDINO). In practice, it's faster
    but slightly less accurate on fine-grained descriptions.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        if self.use_mock:
            pass
        else:
            from transformers import pipeline

            self.model = pipeline(
                "zero-shot-object-detection",
                model="google/owlvit-base-patch32",
                device=0,
            )

    def detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        if self.use_mock:
            return self._mock_detect(image, prompt)
        else:
            return self._real_detect(image, prompt)

    def _mock_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """Mock detection with slight noise to simulate model disagreement."""
        height, width = image.shape[:2]
        bg_color = image.mean(axis=(0, 1))
        diff = np.abs(image.astype(float) - bg_color)
        mask = diff.mean(axis=2) > 50
        nonzero_count = np.sum(mask)

        if nonzero_count < 100:
            return DetectionResult(detected=False, confidence=0.0, model_name="owlvit")

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        rng = np.random.default_rng(seed=42)
        noise = rng.integers(-5, 6, size=4)
        x_min = max(0, x_min + noise[0])
        y_min = max(0, y_min + noise[1])
        x_max = min(width - 1, x_max + noise[2])
        y_max = min(height - 1, y_max + noise[3])

        area_ratio = nonzero_count / (height * width)
        confidence = min(0.90, 0.45 + area_ratio * 10)

        return DetectionResult(
            box=np.array([float(x_min), float(y_min), float(x_max), float(y_max)]),
            confidence=confidence,
            detected=True,
            model_name="owlvit",
        )

    def _real_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """Real OWL-ViT detection using HuggingFace pipeline."""
        from PIL import Image

        pil_image = Image.fromarray(image)
        results = self.model(pil_image, candidate_labels=[prompt])

        filtered = [r for r in results if r["score"] > 0.1]

        if not filtered:
            return DetectionResult(detected=False, confidence=0.0, model_name="owlvit")

        best = max(filtered, key=lambda x: x["score"])
        box = best["box"]

        return DetectionResult(
            box=np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], dtype=float),
            confidence=best["score"],
            detected=True,
            model_name="owlvit",
        )
