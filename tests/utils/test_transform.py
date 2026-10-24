"""Comprehensive unit tests for spatialhub.utils.transform module."""

from __future__ import annotations

import numpy as np
import pytest

from spatialhub.utils.transform import (
    extents_to_corners,
    invert_transform,
    opencv_to_opengl,
    transform_points,
)


class TestInvertTransform:
    """Test suite for 4x4 SE(3) transform inversion."""

    def test_identity_inversion(self) -> None:
        eye = np.eye(4, dtype=np.float32)
        inv = invert_transform(eye)
        assert np.allclose(inv, eye)

    def test_translation_inversion(self) -> None:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = [1.0, 2.0, 3.0]
        inv = invert_transform(mat)

        assert np.allclose(inv[:3, :3], np.eye(3))
        assert np.allclose(inv[:3, 3], [-1.0, -2.0, -3.0])
        assert np.allclose(inv[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_rotation_inversion(self) -> None:
        # 90 degree rotation about Z axis
        mat = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float32)
        inv = invert_transform(mat)

        assert np.allclose(mat @ inv, np.eye(4), atol=1e-6)
        assert np.allclose(inv @ mat, np.eye(4), atol=1e-6)

    def test_combined_rigid_inversion_roundtrip(self) -> None:
        # Random SE(3) matrix
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = q.astype(np.float32)
        mat[:3, 3] = rng.uniform(-10.0, 10.0, size=(3,)).astype(np.float32)

        inv = invert_transform(mat)
        assert np.allclose(mat @ inv, np.eye(4), atol=1e-5)
        assert np.allclose(invert_transform(inv), mat, atol=1e-5)

    def test_batched_inversion(self) -> None:
        batch_mat = np.zeros((5, 4, 4), dtype=np.float32)
        for i in range(5):
            batch_mat[i] = np.eye(4)
            batch_mat[i, 0, 3] = float(i + 1)

        inv_batch = invert_transform(batch_mat)
        assert inv_batch.shape == (5, 4, 4)
        for i in range(5):
            assert np.allclose(inv_batch[i, 0, 3], -float(i + 1))

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(4, 4\)"):
            invert_transform(np.eye(3, dtype=np.float32))


class TestTransformPoints:
    """Test suite for transform_points."""

    def test_identity_preserves_points(self) -> None:
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = transform_points(pts, np.eye(4, dtype=np.float32))
        assert np.allclose(out, pts)

    def test_pure_translation(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = [5.0, -2.0, 10.0]

        out = transform_points(pts, mat)
        expected = np.array([[5.0, -2.0, 10.0], [6.0, -1.0, 11.0]], dtype=np.float32)
        assert np.allclose(out, expected)

    def test_pure_rotation(self) -> None:
        # 90 degrees around Z axis: (1, 0, 0) -> (0, 1, 0)
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mat = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float32)

        out = transform_points(pts, mat)
        assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_batched_points_and_transforms(self) -> None:
        pts_batch = np.zeros((3, 10, 3), dtype=np.float32)
        mat_batch = np.zeros((3, 4, 4), dtype=np.float32)
        for i in range(3):
            mat_batch[i] = np.eye(4)
            mat_batch[i, 2, 3] = float(i + 1)

        out = transform_points(pts_batch, mat_batch)
        assert out.shape == (3, 10, 3)
        for i in range(3):
            assert np.allclose(out[i, :, 2], float(i + 1))


class TestExtentsToCorners:
    """Test suite for extents_to_corners."""

    def test_canonical_corners(self) -> None:
        extents = [2.0, 4.0, 6.0]
        corners = extents_to_corners(extents)

        assert corners.shape == (8, 3)
        assert corners.dtype == np.float32

        # Min and max coordinates along each axis should be +/- extents / 2
        assert np.allclose(np.min(corners, axis=0), [-1.0, -2.0, -3.0])
        assert np.allclose(np.max(corners, axis=0), [ 1.0,  2.0,  3.0])

        # Centroid of canonical corners must be origin
        assert np.allclose(np.mean(corners, axis=0), [0.0, 0.0, 0.0])

    def test_accepts_numpy_array(self) -> None:
        extents = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        corners = extents_to_corners(extents)
        assert corners.shape == (8, 3)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(3,\)"):
            extents_to_corners([1.0, 2.0])  # type: ignore[arg-type]


class TestOpenCVToOpenGL:
    """Test suite for opencv_to_opengl coordinate conversion."""

    def test_reflection_involution(self) -> None:
        # Transforming identity matrix inverts Y and Z axes
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = [1.0, 2.0, 3.0]

        gl_mat = opencv_to_opengl(mat)
        assert np.isclose(gl_mat[0, 3], 1.0)
        assert np.isclose(gl_mat[1, 3], -2.0)
        assert np.isclose(gl_mat[2, 3], -3.0)

        # Involution: applying twice restores original matrix
        restored = opencv_to_opengl(gl_mat)
        assert np.allclose(restored, mat)

    def test_batched_conversion(self) -> None:
        batch_mat = np.zeros((4, 4, 4), dtype=np.float32)
        for i in range(4):
            batch_mat[i] = np.eye(4)
            batch_mat[i, 1, 3] = float(i)

        out = opencv_to_opengl(batch_mat)
        assert out.shape == (4, 4, 4)
        for i in range(4):
            assert np.isclose(out[i, 1, 3], -float(i))
