"""
grounding_dino.py — Grounding DINO object detector.

WHAT THIS FILE DOES:

Wraps the Grounding DINO model from Hugging Face Transformers
into our BaseDetector interface. When running locally (no GPU),
it uses a MOCK that simulates detection by looking for colored
regions in the image. On RunPod with GPU, we swap in the real model.

WHY MOCK FIRST?
- Grounding DINO needs ~2GB VRAM and a GPU for reasonable speed
- Our local machine has no NVIDIA GPU
- But we need to test the full pipeline NOW
- The mock produces output in the EXACT same format as the real model
- When we switch to real model on RunPod, zero code changes needed
  outside this file

This is a professional practice: develop and test with mocks,
deploy with real models. The interface stays the same.
"""

import numpy as np

from omnigrasp_perception.detectors.base_detector import BaseDetector, DetectionResult


class GroundingDINODetector(BaseDetector):
    """
    Grounding DINO detector with mock fallback.

    In production (GPU available):
        self.model = transformers.pipeline("zero-shot-object-detection",
                                           model="IDEA-Research/grounding-dino-base")
        results = self.model(image, candidate_labels=[prompt])

    In development (no GPU):
        Uses color-based detection as a stand-in.
        Finds the largest region of non-background pixels.
    """

    def __init__(self, use_mock: bool = True):
        """
        Args:
            use_mock: If True, use color-based mock detection.
                      If False, load the real Grounding DINO model (needs GPU).
        """
        self.use_mock = use_mock
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load the model or initialize mock."""
        if self.use_mock:
            # No model to load — mock uses numpy operations
            pass
        else:
            # Real model loading (uncomment on RunPod with GPU):
            # from transformers import pipeline
            # self.model = pipeline(
            #     "zero-shot-object-detection",
            #     model="IDEA-Research/grounding-dino-base",
            #     device=0  # GPU device index
            # )
            pass

    def detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """
        Detect an object matching the prompt in the image.

        Args:
            image: RGB image, shape (H, W, 3), dtype uint8
            prompt: Text description like "red bolt"

        Returns:
            DetectionResult with box coordinates and confidence
        """
        if self.use_mock:
            return self._mock_detect(image, prompt)
        else:
            return self._real_detect(image, prompt)

    def _mock_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """
        Mock detection: finds regions that differ from the background.

        HOW IT WORKS:
        1. Calculate the mean color of the image (assumed to be background)
        2. Find pixels that differ significantly from the background
        3. If enough different pixels exist, compute their bounding box
        4. Return the box with a simulated confidence score

        This works with our mock_camera_node because it publishes
        a dark gray image with a colored rectangle — the rectangle
        will be detected as the "object".
        """
        height, width = image.shape[:2]

        # Calculate background color (mean of entire image)
        bg_color = image.mean(axis=(0, 1))

        # Find pixels that differ from background by more than 50 intensity
        diff = np.abs(image.astype(float) - bg_color)
        mask = diff.mean(axis=2) > 50  # Average across RGB channels

        # Count non-background pixels
        nonzero_count = np.sum(mask)

        if nonzero_count < 100:  # Too few pixels — no detection
            return DetectionResult(
                detected=False,
                confidence=0.0,
                model_name="gdino",
            )

        # Find bounding box of non-background region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Simulate confidence (higher when object is larger and more distinct)
        area_ratio = nonzero_count / (height * width)
        confidence = min(0.95, 0.5 + area_ratio * 10)

        return DetectionResult(
            box=np.array([float(x_min), float(y_min), float(x_max), float(y_max)]),
            confidence=confidence,
            detected=True,
            model_name="gdino",
        )

    def _real_detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """
        Real Grounding DINO detection (requires GPU).
        Uncomment and use on RunPod.
        """
        # results = self.model(image, candidate_labels=[prompt])
        # if len(results) == 0:
        #     return DetectionResult(detected=False, model_name="gdino")
        # best = max(results, key=lambda x: x["score"])
        # box = best["box"]
        # return DetectionResult(
        #     box=np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]]),
        #     confidence=best["score"],
        #     detected=True,
        #     model_name="gdino",
        # )
        raise NotImplementedError("Real GDINO requires GPU. Set use_mock=True for local dev.")
