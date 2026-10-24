"""Array validation utilities for shape, dimension, and type assertions."""

from __future__ import annotations

import numpy as np


def validate_array(
    name: str,
    arr: np.ndarray | None,
    *,
    ndim: int | tuple[int, ...] | None = None,
    shape_suffix: tuple[int, ...] | None = None,
    dtype: type | np.dtype | tuple[type | np.dtype, ...] | None = None,
    optional: bool = False,
) -> None:
    """Validate array type, ndim, trailing dimensions, and data type.

    Args:
        name: Name of the attribute being validated.
        arr: Array instance or None.
        ndim: Expected number of dimensions or tuple of acceptable dimensions.
        shape_suffix: Expected trailing dimension tuple.
        dtype: Expected data type, generic type category, or tuple of acceptable types.
        optional: If True, allows None values without raising an error.

    Raises:
        TypeError: If arr is not an instance of np.ndarray, or if its dtype does not match.
        ValueError: If ndim or shape_suffix does not match the array shape.
    """
    if arr is None:
        if not optional:
            raise ValueError(f"{name} is required and cannot be None")
        return

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")

    if dtype is not None:
        expected_types = dtype if isinstance(dtype, tuple) else (dtype,)
        if not any(np.issubdtype(arr.dtype, t) for t in expected_types):
            type_names = [getattr(t, "__name__", str(t)) for t in expected_types]
            types_str = ", ".join(type_names)
            raise TypeError(f"{name} expected dtype in ({types_str}), got {arr.dtype}")

    if ndim is not None:
        valid_ndim = (arr.ndim in ndim) if isinstance(ndim, tuple) else (arr.ndim == ndim)
        if not valid_ndim:
            raise ValueError(f"{name} expected ndim={ndim}, got ndim={arr.ndim} with shape {arr.shape}")

    if shape_suffix is not None:
        actual_suffix = arr.shape[-len(shape_suffix):]
        if actual_suffix != shape_suffix:
            raise ValueError(
                f"{name} trailing dimensions must be {shape_suffix}, got shape {arr.shape}"
            )

