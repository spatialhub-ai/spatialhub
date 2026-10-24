"""Rigid 3D spatial transformations and point operations."""

from __future__ import annotations

import numpy as np

from .validation import validate_array


def invert_transform(
    transform: np.ndarray,
) -> np.ndarray:
    """Compute the inverse of a 4x4 rigid homogeneous transformation.

    Args:
        transform: 4x4 homogeneous transformation matrix of shape (4, 4) or (..., 4, 4).

    Returns:
        Inverted 4x4 transformation matrix matching input shape in float32.

    Raises:
        TypeError: If transform is not a numpy.ndarray.
        ValueError: If transform does not have shape (4, 4) or (..., 4, 4).
    """
    t_arr = np.asarray(transform, dtype=np.float32)
    validate_array("transform", t_arr, shape_suffix=(4, 4))

    rot = t_arr[..., :3, :3]
    t = t_arr[..., :3, 3:4]
    rot_inv = rot.swapaxes(-1, -2)
    t_inv = -rot_inv @ t

    output = np.zeros_like(t_arr)
    output[..., :3, :3] = rot_inv
    output[..., :3, 3:4] = t_inv
    output[..., 3, 3] = 1.0
    return output


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 rigid homogeneous transformation matrix to 3D points.

    Supports single (N, 3) point arrays with a (4, 4) transformation matrix,
    as well as batched (..., N, 3) arrays with (..., 4, 4) or broadcasted matrices.

    Args:
        points: 3D point coordinates of shape (N, 3) or (..., N, 3).
        transform: 4x4 homogeneous transformation matrix of shape (4, 4) or (..., 4, 4).

    Returns:
        Transformed 3D point coordinates matching broadcasted input shape in float32.

    Raises:
        TypeError: If points or transform is not a numpy.ndarray.
        ValueError: If points or transform dimensions do not conform to expected shapes.
    """
    pts = np.asarray(points, dtype=np.float32)
    t = np.asarray(transform, dtype=np.float32)

    validate_array("points", pts, shape_suffix=(3,))
    validate_array("transform", t, shape_suffix=(4, 4))

    rot = t[..., :3, :3]
    trans = t[..., :3, 3]
    return np.matmul(pts, rot.swapaxes(-1, -2)) + trans[..., None, :]


def extents_to_corners(
    extents: np.ndarray | list[float]
) -> np.ndarray:
    """Compute the 8 canonical 3D bounding box corners centered at origin.

    Corners follow standard parallel-face cyclic topology:
    - Corners 0..3 form the first quadrilateral face in cyclic order.
    - Corners 4..7 form the opposing parallel face in matching cyclic order.

    Args:
        extents: 3D bounding box dimensions (dx, dy, dz) of shape (3,).

    Returns:
        3D corner coordinates of shape (8, 3) in float32 centered at (0, 0, 0).

    Raises:
        TypeError: If extents is not array-like.
        ValueError: If extents does not have shape (3,).
    """
    ext = np.asarray(extents, dtype=np.float32)
    validate_array("extents", ext, ndim=1, shape_suffix=(3,))

    hx, hy, hz = ext / 2.0
    return np.array(
        [
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ],
        dtype=np.float32,
    )


def opencv_to_opengl(
    transform: np.ndarray,
) -> np.ndarray:
    """Convert a 4x4 rigid transformation between right-down-forward (OpenCV) and right-up-backward (OpenGL) coordinate frames.

    Applies the reflection involution matrix diag(1, -1, -1, 1), inverting the Y and Z axes:
    - Right-down-forward frame: +X right, +Y down, +Z forward along the optical axis.
    - Right-up-backward frame: +X right, +Y up, -Z forward along the optical axis.

    Because the reflection matrix is an involution, this operation is symmetric and
    inverts itself when applied a second time.

    Args:
        transform: 4x4 homogeneous transformation matrix of shape (4, 4) or (..., 4, 4).

    Returns:
        Converted 4x4 transformation matrix matching input shape in float32.

    Raises:
        ValueError: If transform does not have shape (4, 4) or (..., 4, 4).
    """
    t_arr = np.asarray(transform, dtype=np.float32)
    validate_array("transform", t_arr, shape_suffix=(4, 4))

    reflection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return reflection @ t_arr


__all__ = [
    "invert_transform",
    "transform_points",
    "extents_to_corners",
    "opencv_to_opengl",
]

