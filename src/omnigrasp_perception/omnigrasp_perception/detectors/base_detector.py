"""
base_detector.py — Abstract interface for all object detectors.

WHY AN ABSTRACT BASE CLASS?

This defines a CONTRACT that all detectors must follow. Both
GroundingDINODetector and OWLViTDetector implement this same
interface. This means:

1. You can swap models without changing any other code
2. The fusion module doesn't care WHICH detector produced a result
3. Adding a new model (e.g., PaliGemma) means writing ONE file
   that implements this interface — nothing else changes

This is the Strategy Pattern from software design. Interviewers
notice this because it shows you think about architecture, not
just making things work.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    """
    Standardized output from any detector.

    WHY A DATACLASS?
    A dataclass is a Python class that automatically generates
    __init__, __repr__, and __eq__ methods from the fields you
    define. It's cleaner than using a dictionary because:
    - Fields are typed (IDE autocomplete works)
    - You can't misspell a key (box vs bbox vs bounding_box)
    - It's self-documenting
    """

    box: Optional[np.ndarray] = None  # [x_min, y_min, x_max, y_max] in pixels
    confidence: float = 0.0  # 0.0 to 1.0
    detected: bool = False  # True if an object was found
    model_name: str = ""  # Which model produced this ("gdino" or "owlvit")


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    ABC = Abstract Base Class. Any class that inherits from this
    MUST implement the detect() method, or Python will raise
    TypeError when you try to instantiate it.

    This enforces the contract: every detector has a detect() method
    with the same signature.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, prompt: str) -> DetectionResult:
        """
        Detect an object in the image matching the text prompt.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8
            prompt: Text description of object to find, e.g. "red bolt"

        Returns:
            DetectionResult with bounding box and confidence
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights. Called once during initialization."""
        pass
