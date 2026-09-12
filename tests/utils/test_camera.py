"""Comprehensive unit tests for spatialhub.utils.camera module."""

import numpy as np
import pytest

from spatialhub.utils.camera import (
    create_projection_matrix,
    opencv_to_opengl_pose,
    reproject_depth_to_3d,
    reproject_depth_to_3d_batch,
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

    def test_scale_intrinsics_isotropic(self, base_k: np.ndarray):
        # Downscale 2x: (640, 480) -> (320, 240)
        k_scaled = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(320, 240))
        assert np.isclose(k_scaled[0, 0], 250.0)
        assert np.isclose(k_scaled[1, 1], 250.0)
        assert np.isclose(k_scaled[0, 2], 160.0)
        assert np.isclose(k_scaled[1, 2], 120.0)
        assert k_scaled[2, 2] == 1.0

    def test_scale_intrinsics_anisotropic(self, base_k: np.ndarray):
        # Scale width by 2x, height by 0.5x
        k_scaled = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(1280, 240))
        assert np.isclose(k_scaled[0, 0], 1000.0)
        assert np.isclose(k_scaled[1, 1], 250.0)
        assert np.isclose(k_scaled[0, 2], 640.0)
        assert np.isclose(k_scaled[1, 2], 120.0)

    def test_scale_intrinsics_invertibility(self, base_k: np.ndarray):
        # Scale up then scale back down -> should equal original base_k
        k_up = scale_intrinsics(base_k, orig_size=(640, 480), new_size=(1280, 960))
        k_restored = scale_intrinsics(k_up, orig_size=(1280, 960), new_size=(640, 480))
        assert np.allclose(base_k, k_restored, atol=1e-5)


class TestReprojectDepthTo3D:
    """Test suite for 3D depth reprojection."""

    @pytest.fixture
    def camera_intrinsics(self) -> np.ndarray:
        return np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_reproject_depth_to_3d_principal_point(self, camera_intrinsics: np.ndarray):
        # Constant 2.0 meters depth
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        xyz_map = reproject_depth_to_3d(depth, camera_intrinsics)

        assert xyz_map.shape == (480, 640, 3)
        assert xyz_map.dtype == np.float32
        # Principal point pixel (u=320, v=240) in (row=240, col=320) must be (0, 0, 2.0)
        assert np.allclose(xyz_map[240, 320], [0.0, 0.0, 2.0], atol=1e-4)

    def test_reproject_depth_to_3d_threshold_masking(self, camera_intrinsics: np.ndarray):
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        # Set invalid depths outside [0.5, 3.0]
        depth[100, 100] = 0.1  # below z_min
        depth[200, 200] = 5.0  # above z_max

        xyz_map = reproject_depth_to_3d(depth, camera_intrinsics, z_min=0.5, z_max=3.0)
        assert np.allclose(xyz_map[100, 100], [0.0, 0.0, 0.0])
        assert np.allclose(xyz_map[200, 200], [0.0, 0.0, 0.0])

    def test_reproject_depth_to_3d_custom_uvs(self, camera_intrinsics: np.ndarray):
        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        uvs = np.array([[320, 240], [0, 0]], dtype=np.float32)
        xyz_map = reproject_depth_to_3d(depth, camera_intrinsics, uvs=uvs)

        # Selected uvs must be populated
        assert np.allclose(xyz_map[240, 320], [0.0, 0.0, 1.5], atol=1e-4)
        assert np.any(xyz_map[0, 0] != 0.0)

    def test_reproject_depth_to_3d_batch(self, camera_intrinsics: np.ndarray):
        depths_batch = np.ones((3, 480, 640), dtype=np.float32) * 2.5
        ks_batch = np.stack([camera_intrinsics] * 3, axis=0)

        xyz_batch = reproject_depth_to_3d_batch(depths_batch, ks_batch)
        assert xyz_batch.shape == (3, 480, 640, 3)
        assert xyz_batch.dtype == np.float32

        for i in range(3):
            assert np.allclose(xyz_batch[i, 240, 320], [0.0, 0.0, 2.5], atol=1e-4)


class TestProjectionMatrix:
    """Test suite for OpenGL perspective projection matrix generation."""

    @pytest.fixture
    def sample_k(self) -> np.ndarray:
        return np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_create_projection_matrix_ydown(self, sample_k: np.ndarray):
        proj = create_projection_matrix(sample_k, height=480, width=640, znear=0.1, zfar=100.0, window_coords="y_down")
        assert proj.shape == (4, 4)
        assert proj.dtype == np.float32
        assert proj[3, 2] == -1.0
        assert proj[3, 3] == 0.0

    def test_create_projection_matrix_yup(self, sample_k: np.ndarray):
        proj = create_projection_matrix(sample_k, height=480, width=640, znear=0.1, zfar=100.0, window_coords="y_up")
        assert proj.shape == (4, 4)
        assert proj[3, 2] == -1.0

    def test_create_projection_matrix_invalid_window_coords(self, sample_k: np.ndarray):
        with pytest.raises(ValueError, match="window_coords must be either"):
            create_projection_matrix(sample_k, height=480, width=640, window_coords="invalid")


class TestPoseCoordinateConversion:
    """Test suite for OpenCV <-> OpenGL pose coordinate conversion."""

    def test_opencv_to_opengl_pose_single(self):
        pose_cv = np.eye(4, dtype=np.float32)
        pose_gl = opencv_to_opengl_pose(pose_cv)
        assert pose_gl.shape == (4, 4)
        assert np.allclose(pose_gl[:3, :3], np.diag([1.0, -1.0, -1.0]))

    def test_opencv_to_opengl_pose_batch(self):
        poses_cv = np.tile(np.eye(4, dtype=np.float32)[None], (4, 1, 1))
        poses_gl = opencv_to_opengl_pose(poses_cv)
        assert poses_gl.shape == (4, 4, 4)
        for i in range(4):
            assert np.allclose(poses_gl[i, :3, :3], np.diag([1.0, -1.0, -1.0]))

    def test_pose_conversion_involutory_property(self):
        # f(f(P)) == P (applying the transform twice returns the original pose)
        random_pose = np.eye(4, dtype=np.float32)
        random_pose[:3, 3] = [1.2, -0.5, 3.4]

        gl_pose = opencv_to_opengl_pose(random_pose)
        cv_restored = opencv_to_opengl_pose(gl_pose)
        assert np.allclose(random_pose, cv_restored)

    def test_pose_conversion_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected pose shape"):
            opencv_to_opengl_pose(np.zeros((3, 3), dtype=np.float32))
