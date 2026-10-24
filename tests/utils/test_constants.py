"""Unit tests for spatialhub.utils.constants."""

from __future__ import annotations

import numpy as np

from spatialhub.utils.constants import IMAGENET_MEAN, IMAGENET_STD


class TestConstants:
    """Test suite for shared constants."""

    def test_imagenet_mean(self) -> None:
        assert isinstance(IMAGENET_MEAN, np.ndarray)
        assert IMAGENET_MEAN.shape == (3,)
        assert IMAGENET_MEAN.dtype == np.float32
        assert np.allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406])

    def test_imagenet_std(self) -> None:
        assert isinstance(IMAGENET_STD, np.ndarray)
        assert IMAGENET_STD.shape == (3,)
        assert IMAGENET_STD.dtype == np.float32
        assert np.allclose(IMAGENET_STD, [0.229, 0.224, 0.225])
