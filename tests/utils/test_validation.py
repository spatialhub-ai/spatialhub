"""Comprehensive unit tests for spatialhub.utils.validation module."""

from __future__ import annotations

import numpy as np
import pytest

from spatialhub.utils.validation import validate_array


class TestValidateArray:
    """Test suite for validate_array helper."""

    def test_valid_array(self) -> None:
        arr = np.zeros((2, 3, 3), dtype=np.float32)
        validate_array("test", arr, ndim=3, shape_suffix=(3, 3), dtype=np.float32)

    def test_none_handling(self) -> None:
        validate_array("test", None, optional=True)
        with pytest.raises(ValueError, match="test is required and cannot be None"):
            validate_array("test", None, optional=False)

    def test_non_numpy_type_raises(self) -> None:
        with pytest.raises(TypeError, match="test must be a numpy.ndarray, got list"):
            validate_array("test", [1, 2, 3])  # type: ignore[arg-type]

    def test_ndim_exact_match(self) -> None:
        arr = np.zeros((2, 4))
        validate_array("test", arr, ndim=2)
        with pytest.raises(ValueError, match=r"test expected ndim=3, got ndim=2 with shape \(2, 4\)"):
            validate_array("test", arr, ndim=3)

    def test_ndim_tuple_allowed(self) -> None:
        arr_2d = np.zeros((2, 4))
        arr_3d = np.zeros((1, 2, 4))
        validate_array("test", arr_2d, ndim=(2, 3))
        validate_array("test", arr_3d, ndim=(2, 3))

        arr_4d = np.zeros((1, 1, 2, 4))
        with pytest.raises(ValueError, match=r"test expected ndim=\(2, 3\)"):
            validate_array("test", arr_4d, ndim=(2, 3))

    def test_shape_suffix(self) -> None:
        arr = np.zeros((10, 4, 4))
        validate_array("test", arr, shape_suffix=(4, 4))

        with pytest.raises(ValueError, match=r"test trailing dimensions must be \(3, 3\), got shape \(10, 4, 4\)"):
            validate_array("test", arr, shape_suffix=(3, 3))

    def test_dtype_matching(self) -> None:
        arr_f = np.zeros((2, 3), dtype=np.float32)
        validate_array("test", arr_f, dtype=np.float32)
        validate_array("test", arr_f, dtype=np.floating)

        arr_i = np.zeros((2, 3), dtype=np.int64)
        validate_array("test", arr_i, dtype=np.integer)

        arr_b = np.ones((2, 3), dtype=bool)
        validate_array("test", arr_b, dtype=np.bool_)
        validate_array("test", arr_b, dtype=(np.bool_, np.integer))

    def test_dtype_mismatch_raises(self) -> None:
        arr_i = np.zeros((2, 3), dtype=np.int32)
        with pytest.raises(TypeError, match="test expected dtype in .*floating"):
            validate_array("test", arr_i, dtype=np.floating)

        arr_f = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="test expected dtype in"):
            validate_array("test", arr_f, dtype=(np.integer, np.bool_))

    def test_empty_array_handling(self) -> None:
        arr = np.zeros((0, 4), dtype=np.float32)
        validate_array("test", arr, ndim=2, shape_suffix=(4,))
