"""Image preprocessing and normalization constants."""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
