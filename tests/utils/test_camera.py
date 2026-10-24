"""Comprehensive unit tests for spatialhub.utils.camera module."""

from __future__ import annotations

import numpy as np
import pytest

from spatialhub.utils.camera import (
    intrinsics_to_projection_matrix,
    project_points_to_image,
    reproject_depth_to_3d,
    sample_icosphere_poses,
    sample_sphere_points,
    sample_sphere_poses,
    scale_intrinsics,
)


class TestScaleIntrinsics:
    """Test suite for camera intrinsic scaling."""

    @pytest.fixture
    def base_k(self) -> np.ndarray:
        return np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_scale_intrinsics_isotropic(self, base_k: np.ndarray) -> None:
        # Downscale 2x: (640, 480) -> (320, 240)
        k_scaled = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(320, 240))
        assert np.isclose(k_scaled[0, 0], 250.0)
        assert np.isclose(k_scaled[1, 1], 250.0)
        assert np.isclose(k_scaled[0, 2], 160.0)
        assert np.isclose(k_scaled[1, 2], 120.0)
        assert k_scaled[2, 2] == 1.0

    def test_scale_intrinsics_anisotropic(self, base_k: np.ndarray) -> None:
        # Scale width by 2x, height by 0.5x
        k_scaled = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(1280, 240))
        assert np.isclose(k_scaled[0, 0], 1000.0)
        assert np.isclose(k_scaled[1, 1], 250.0)
        assert np.isclose(k_scaled[0, 2], 640.0)
        assert np.isclose(k_scaled[1, 2], 120.0)

    def test_scale_intrinsics_invertibility(self, base_k: np.ndarray) -> None:
        k_up = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(1280, 960))
        k_restored = scale_intrinsics(k_up, orig_size=(1280, 960), new_size=(640, 480))
        assert np.allclose(base_k, k_restored, atol=1e-5)

    def test_scale_intrinsics_batched(self, base_k: np.ndarray) -> None:
        k_batch = np.stack([base_k, base_k], axis=0)
        k_scaled = scale_intrinsics(k_batch, orig_size=(640, 480), new_size=(320, 240))
        assert k_scaled.shape == (2, 3, 3)
        assert np.isclose(k_scaled[0, 0, 0], 250.0)


class TestReprojectDepthTo3D:
    """Test suite for 3D depth reprojection."""

    @pytest.fixture
    def camera_intrinsics(self) -> np.ndarray:
        return np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_reproject_principal_point(self, camera_intrinsics: np.ndarray) -> None:
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[240, 320] = 2.0  # At optical center (cx=320, cy=240), Z=2.0 -> X=0, Y=0, Z=2.0

        pts = reproject_depth_to_3d(depth, camera_intrinsics)
        assert pts.shape == (480, 640, 3)
        center_pt = pts[240, 320]
        assert np.allclose(center_pt, [0.0, 0.0, 2.0], atol=1e-4)

    def test_reproject_depth_thresholding(self, camera_intrinsics: np.ndarray) -> None:
        depth = np.full((10, 10), 5.0, dtype=np.float32)
        depth[0, 0] = 0.0001  # below z_min
        depth[0, 1] = 100.0   # above z_max

        pts = reproject_depth_to_3d(depth, camera_intrinsics, z_min=0.1, z_max=10.0)
        assert np.allclose(pts[0, 0], [0.0, 0.0, 0.0])
        assert np.allclose(pts[0, 1], [0.0, 0.0, 0.0])
        assert not np.allclose(pts[5, 5], [0.0, 0.0, 0.0])

    def test_reproject_batched_depth(self, camera_intrinsics: np.ndarray) -> None:
        depth_batch = np.ones((2, 100, 100), dtype=np.float32) * 3.0
        pts_batch = reproject_depth_to_3d(depth_batch, camera_intrinsics)
        assert pts_batch.shape == (2, 100, 100, 3)
        assert np.isclose(pts_batch[0, 50, 50, 2], 3.0)


class TestProjectPointsToImage:
    """Test suite for project_points_to_image."""

    @pytest.fixture
    def k_matrix(self) -> np.ndarray:
        return np.array([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_project_points_without_pose(self, k_matrix: np.ndarray) -> None:
        # Point on optical axis at Z=2: X=0, Y=0, Z=2 -> u=50, v=50
        points = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 2.0],
        ], dtype=np.float32)

        pts_2d, valid = project_points_to_image(points, k_matrix)
        assert pts_2d.shape == (2, 2)
        assert np.all(valid)
        assert np.allclose(pts_2d[0], [50.0, 50.0])
        assert np.allclose(pts_2d[1], [100.0, 150.0])

    def test_project_points_behind_camera(self, k_matrix: np.ndarray) -> None:
        points = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0],
        ], dtype=np.float32)

        pts_2d, valid = project_points_to_image(points, k_matrix)
        assert not valid[0]
        assert valid[1]

    def test_project_points_with_pose(self, k_matrix: np.ndarray) -> None:
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        # Translation by Z=+5: transforms origin to (0, 0, 5)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [0.0, 0.0, 5.0]

        pts_2d, valid = project_points_to_image(points, k_matrix, pose=pose)
        assert np.all(valid)
        assert np.allclose(pts_2d[0], [50.0, 50.0])


class TestIntrinsicsToProjectionMatrix:
    """Test suite for intrinsics_to_projection_matrix."""

    def test_projection_matrix_structure(self) -> None:
        k = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        proj_down = intrinsics_to_projection_matrix(k, height=480, width=640, znear=0.1, zfar=10.0, window_coords="y_down")
        assert proj_down.shape == (4, 4)
        assert proj_down[3, 2] == -1.0

        proj_up = intrinsics_to_projection_matrix(k, height=480, width=640, znear=0.1, zfar=10.0, window_coords="y_up")
        assert proj_up.shape == (4, 4)

    def test_invalid_window_coords_raises(self) -> None:
        k = np.eye(3, dtype=np.float32)
        with pytest.raises(ValueError, match="window_coords must be either 'y_down' or 'y_up'"):
            intrinsics_to_projection_matrix(k, height=100, width=100, window_coords="invalid")


class TestSphereSampling:
    """Test suite for spherical point and pose sampling."""

    def test_sample_sphere_points_single(self) -> None:
        pts = sample_sphere_points(num_points=1, radius=2.5)
        assert pts.shape == (1, 3)
        assert np.allclose(pts, [[0.0, 0.0, 2.5]])

    def test_sample_sphere_points_multiple(self) -> None:
        pts = sample_sphere_points(num_points=50, radius=2.0)
        assert pts.shape == (50, 3)
        radii = np.linalg.norm(pts, axis=1)
        assert np.allclose(radii, 2.0, atol=1e-5)

    def test_sample_sphere_points_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="num_points must be greater than 0"):
            sample_sphere_points(num_points=0)

    def test_sample_sphere_poses(self) -> None:
        poses = sample_sphere_poses(num_viewpoints=10, radius=3.0)
        assert poses.shape == (10, 4, 4)
        for i in range(10):
            r = poses[i, :3, :3]
            t = poses[i, :3, 3]
            # Rotation must be orthonormal
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-4)
            assert np.isclose(np.linalg.det(r), 1.0, atol=1e-4)
            # Distance from origin must equal radius
            assert np.isclose(np.linalg.norm(t), 3.0, atol=1e-4)
            # Optical axis (+Z in camera frame) must point toward origin (0, 0, 0)
            forward = poses[i, :3, 2]
            to_origin = -t / np.linalg.norm(t)
            assert np.allclose(forward, to_origin, atol=1e-3)

    def test_sample_icosphere_poses(self) -> None:
        pytest.importorskip("trimesh")
        poses = sample_icosphere_poses(subdivisions=0, radius=1.5)
        assert poses.shape == (12, 4, 4)

        poses_min = sample_icosphere_poses(min_n_views=40, radius=1.0)
        assert len(poses_min) >= 40
