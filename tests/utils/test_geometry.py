"""Comprehensive unit tests for spatialhub.utils.geometry module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spatialhub.utils.geometry import (
    estimate_relative_pose_2d,
    solve_pnp_ransac,
)


class TestEstimateRelativePose2D:
    """Test suite for 2D-to-2D relative camera pose estimation."""

    @pytest.fixture
    def intrinsics(self) -> np.ndarray:
        return np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    @pytest.fixture
    def synthetic_scene(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 60 3D points placed in front of the camera (Z between 1.5m and 4.5m)
        rng = np.random.default_rng(seed=42)
        x = rng.uniform(-0.8, 0.8, size=(60, 1))
        y = rng.uniform(-0.6, 0.6, size=(60, 1))
        z = rng.uniform(1.5, 4.5, size=(60, 1))
        pts_3d = np.hstack([x, y, z]).astype(np.float32)

        # Ground truth rotation (pitch and yaw: ~5 deg yaw, -3 deg pitch)
        rvec = np.array([0.05, -0.08, 0.02], dtype=np.float32)
        r_gt, _ = cv2.Rodrigues(rvec)

        # Ground truth translation
        t_gt = np.array([0.20, -0.05, 0.40], dtype=np.float32)

        return pts_3d, r_gt, t_gt

    def test_estimate_relative_pose_clean(
        self,
        intrinsics: np.ndarray,
        synthetic_scene: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        pts_3d, r_gt, t_gt = synthetic_scene
        k = intrinsics

        # Project to view A
        pts_a_2d = (k @ pts_3d.T).T
        pts_a = (pts_a_2d[:, :2] / pts_a_2d[:, 2:3]).astype(np.float32)

        # Transform to view B: P_B = R @ P_A + t
        pts_b_3d = (r_gt @ pts_3d.T).T + t_gt
        pts_b_2d = (k @ pts_b_3d.T).T
        pts_b = (pts_b_2d[:, :2] / pts_b_2d[:, 2:3]).astype(np.float32)

        transform, inliers = estimate_relative_pose_2d(
            points_a=pts_a,
            points_b=pts_b,
            intrinsics_a=k,
        )

        assert transform.shape == (4, 4)
        assert transform.dtype == np.float32
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])

        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        # Rotation error check: R_est @ R_gt.T should be near identity
        rot_diff = r_est @ r_gt.T
        assert np.allclose(rot_diff, np.eye(3), atol=1.5e-2)

        # Translation direction check: cosine similarity with unit ground truth
        t_gt_unit = t_gt / np.linalg.norm(t_gt)
        cosine_sim = float(np.dot(t_est, t_gt_unit))
        assert cosine_sim > 0.99
        assert np.isclose(np.linalg.norm(t_est), 1.0, atol=1e-5)

        # Most points should be valid inliers
        assert np.count_nonzero(inliers) >= 50

    def test_estimate_relative_pose_with_outliers(
        self,
        intrinsics: np.ndarray,
        synthetic_scene: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        pts_3d, r_gt, t_gt = synthetic_scene
        k = intrinsics

        pts_a_2d = (k @ pts_3d.T).T
        pts_a = (pts_a_2d[:, :2] / pts_a_2d[:, 2:3]).astype(np.float32)

        pts_b_3d = (r_gt @ pts_3d.T).T + t_gt
        pts_b_2d = (k @ pts_b_3d.T).T
        pts_b = (pts_b_2d[:, :2] / pts_b_2d[:, 2:3]).astype(np.float32)

        # Corrupt 12 correspondences (20% outliers)
        rng = np.random.default_rng(seed=123)
        pts_b[:12] += rng.uniform(50.0, 150.0, size=(12, 2)).astype(np.float32)

        transform, inliers = estimate_relative_pose_2d(
            points_a=pts_a,
            points_b=pts_b,
            intrinsics_a=k,
            threshold_px=1.5,
        )

        assert transform.shape == (4, 4)
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])

        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        rot_diff = r_est @ r_gt.T
        assert np.allclose(rot_diff, np.eye(3), atol=3e-2)

        t_gt_unit = t_gt / np.linalg.norm(t_gt)
        cosine_sim = float(np.dot(t_est, t_gt_unit))
        assert cosine_sim > 0.95

        # Inlier mask should reject corrupted points
        assert not np.all(inliers[:12])
        assert np.count_nonzero(inliers[12:]) >= 40

    def test_estimate_relative_pose_differing_intrinsics(
        self,
        synthetic_scene: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        pts_3d, r_gt, t_gt = synthetic_scene

        k_a = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        k_b = np.array([
            [700.0, 0.0, 350.0],
            [0.0, 700.0, 260.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        pts_a_2d = (k_a @ pts_3d.T).T
        pts_a = (pts_a_2d[:, :2] / pts_a_2d[:, 2:3]).astype(np.float32)

        pts_b_3d = (r_gt @ pts_3d.T).T + t_gt
        pts_b_2d = (k_b @ pts_b_3d.T).T
        pts_b = (pts_b_2d[:, :2] / pts_b_2d[:, 2:3]).astype(np.float32)

        transform, inliers = estimate_relative_pose_2d(
            points_a=pts_a,
            points_b=pts_b,
            intrinsics_a=k_a,
            intrinsics_b=k_b,
        )

        assert transform.shape == (4, 4)
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])

        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        rot_diff = r_est @ r_gt.T
        assert np.allclose(rot_diff, np.eye(3), atol=2e-2)

        t_gt_unit = t_gt / np.linalg.norm(t_gt)
        assert np.dot(t_est, t_gt_unit) > 0.98

    def test_estimate_relative_pose_minimal_five_points(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        # 5 points trigger OpenCV 5-point algorithm returning stacked candidate solutions
        k = intrinsics
        pts_3d = np.array([
            [-0.3, -0.2, 2.0],
            [0.4, -0.1, 2.5],
            [0.1, 0.3, 1.8],
            [-0.2, 0.4, 3.0],
            [0.3, 0.2, 2.2],
        ], dtype=np.float32)

        r_gt, _ = cv2.Rodrigues(np.array([0.05, -0.08, 0.02], dtype=np.float32))
        t_gt = np.array([0.2, -0.05, 0.4], dtype=np.float32)

        pts_a_2d = (k @ pts_3d.T).T
        pts_a = (pts_a_2d[:, :2] / pts_a_2d[:, 2:3]).astype(np.float32)

        pts_b_3d = (r_gt @ pts_3d.T).T + t_gt
        pts_b_2d = (k @ pts_b_3d.T).T
        pts_b = (pts_b_2d[:, :2] / pts_b_2d[:, 2:3]).astype(np.float32)

        transform, inliers = estimate_relative_pose_2d(
            points_a=pts_a,
            points_b=pts_b,
            intrinsics_a=k,
        )

        assert transform.shape == (4, 4)
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])
        assert inliers.shape == (5,)
        assert np.count_nonzero(inliers) >= 3

    def test_estimate_relative_pose_insufficient_points_raises(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        pts = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="At least 5 point correspondences required"):
            estimate_relative_pose_2d(pts, pts, intrinsics)

    def test_estimate_relative_pose_length_mismatch_raises(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        pts_a = np.ones((6, 2), dtype=np.float32)
        pts_b = np.ones((7, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="matching lengths"):
            estimate_relative_pose_2d(pts_a, pts_b, intrinsics)

    def test_estimate_relative_pose_invalid_type_raises(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        pts = [[100.0, 150.0], [200.0, 180.0], [300.0, 220.0], [150.0, 350.0], [400.0, 300.0]]
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            estimate_relative_pose_2d(pts, np.array(pts), intrinsics)  # type: ignore[arg-type]

    def test_estimate_relative_pose_invalid_ndim_raises(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        pts = np.ones((1, 10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="expected ndim=2"):
            estimate_relative_pose_2d(pts, pts, intrinsics)

    def test_estimate_relative_pose_collinear_degenerate(
        self,
        intrinsics: np.ndarray,
    ) -> None:
        # Collinear points cause cv2 solvers to fail gracefully
        pts = np.zeros((10, 2), dtype=np.float32)
        transform, inliers = estimate_relative_pose_2d(pts, pts, intrinsics)
        assert np.allclose(transform, np.eye(4))
        assert not np.any(inliers)


class TestSolvePnPRansac:
    """Test suite for 3D-to-2D Perspective-n-Point pose estimation."""

    @pytest.fixture
    def intrinsics(self) -> np.ndarray:
        return np.array([
            [550.0, 0.0, 320.0],
            [0.0, 550.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_solve_pnp_clean(self, intrinsics: np.ndarray) -> None:
        rng = np.random.default_rng(seed=100)
        pts_3d = rng.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
        pts_3d[:, 2] += 3.0  # Center around Z=3.0

        rvec_gt = np.array([0.1, -0.15, 0.05], dtype=np.float32)
        r_gt, _ = cv2.Rodrigues(rvec_gt)
        t_gt = np.array([0.25, -0.10, 0.50], dtype=np.float32)

        # Project points to 2D
        pts_cam = (r_gt @ pts_3d.T).T + t_gt
        proj_2d = (intrinsics @ pts_cam.T).T
        pts_2d = (proj_2d[:, :2] / proj_2d[:, 2:3]).astype(np.float32)

        transform, inliers = solve_pnp_ransac(
            points_3d=pts_3d,
            points_2d=pts_2d,
            intrinsics=intrinsics,
        )

        assert transform.shape == (4, 4)
        assert transform.dtype == np.float32
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])

        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        assert np.allclose(r_est @ r_gt.T, np.eye(3), atol=1e-3)
        assert np.allclose(t_est, t_gt, atol=1e-3)
        assert np.all(inliers)

    def test_solve_pnp_with_outliers(self, intrinsics: np.ndarray) -> None:
        rng = np.random.default_rng(seed=200)
        pts_3d = rng.uniform(-1.0, 1.0, size=(60, 3)).astype(np.float32)
        pts_3d[:, 2] += 3.0

        rvec_gt = np.array([0.08, 0.05, -0.02], dtype=np.float32)
        r_gt, _ = cv2.Rodrigues(rvec_gt)
        t_gt = np.array([0.10, 0.20, -0.15], dtype=np.float32)

        pts_cam = (r_gt @ pts_3d.T).T + t_gt
        proj_2d = (intrinsics @ pts_cam.T).T
        pts_2d = (proj_2d[:, :2] / proj_2d[:, 2:3]).astype(np.float32)

        # Corrupt 10 points
        pts_2d[:10] += rng.uniform(40.0, 100.0, size=(10, 2)).astype(np.float32)

        transform, inliers = solve_pnp_ransac(
            points_3d=pts_3d,
            points_2d=pts_2d,
            intrinsics=intrinsics,
        )

        assert transform.shape == (4, 4)
        assert np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])

        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        assert np.allclose(r_est @ r_gt.T, np.eye(3), atol=5e-3)
        assert np.allclose(t_est, t_gt, atol=5e-3)
        assert not np.all(inliers[:10])
        assert np.count_nonzero(inliers[10:]) >= 45

    def test_solve_pnp_with_distortion(self, intrinsics: np.ndarray) -> None:
        rng = np.random.default_rng(seed=300)
        pts_3d = rng.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
        pts_3d[:, 2] += 3.0

        rvec_gt = np.array([0.05, -0.10, 0.08], dtype=np.float32)
        r_gt, _ = cv2.Rodrigues(rvec_gt)
        t_gt = np.array([0.15, -0.20, 0.40], dtype=np.float32)
        dist_gt = np.array([-0.08, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)

        pts_2d_proj, _ = cv2.projectPoints(pts_3d, rvec_gt, t_gt, intrinsics, dist_gt)
        pts_2d = pts_2d_proj.squeeze(axis=1).astype(np.float32)

        transform, inliers = solve_pnp_ransac(
            points_3d=pts_3d,
            points_2d=pts_2d,
            intrinsics=intrinsics,
            dist_coeffs=dist_gt,
        )

        assert transform.shape == (4, 4)
        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        assert np.allclose(r_est @ r_gt.T, np.eye(3), atol=1e-3)
        assert np.allclose(t_est, t_gt, atol=1e-3)
        assert np.all(inliers)

    def test_solve_pnp_with_custom_flags(self, intrinsics: np.ndarray) -> None:
        rng = np.random.default_rng(seed=400)
        pts_3d = rng.uniform(-1.0, 1.0, size=(40, 3)).astype(np.float32)
        pts_3d[:, 2] += 3.0

        rvec_gt = np.array([0.02, 0.04, -0.01], dtype=np.float32)
        r_gt, _ = cv2.Rodrigues(rvec_gt)
        t_gt = np.array([0.05, 0.10, 0.20], dtype=np.float32)

        pts_cam = (r_gt @ pts_3d.T).T + t_gt
        proj_2d = (intrinsics @ pts_cam.T).T
        pts_2d = (proj_2d[:, :2] / proj_2d[:, 2:3]).astype(np.float32)

        transform, inliers = solve_pnp_ransac(
            points_3d=pts_3d,
            points_2d=pts_2d,
            intrinsics=intrinsics,
            max_iterations=150,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        assert transform.shape == (4, 4)
        r_est = transform[:3, :3]
        t_est = transform[:3, 3]

        assert np.allclose(r_est @ r_gt.T, np.eye(3), atol=1e-3)
        assert np.allclose(t_est, t_gt, atol=1e-3)
        assert np.all(inliers)

    def test_solve_pnp_insufficient_points_raises(self, intrinsics: np.ndarray) -> None:
        pts_3d = np.ones((3, 3), dtype=np.float32)
        pts_2d = np.ones((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="At least 4 point correspondences required"):
            solve_pnp_ransac(pts_3d, pts_2d, intrinsics)

    def test_solve_pnp_invalid_type_raises(self, intrinsics: np.ndarray) -> None:
        pts_3d = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        pts_2d = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            solve_pnp_ransac(pts_3d, pts_2d, intrinsics)  # type: ignore[arg-type]

    def test_solve_pnp_invalid_ndim_raises(self, intrinsics: np.ndarray) -> None:
        pts_3d = np.ones((1, 10, 3), dtype=np.float32)
        pts_2d = np.ones((1, 10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="expected ndim=2"):
            solve_pnp_ransac(pts_3d, pts_2d, intrinsics)

    def test_solve_pnp_degenerate_collinear(self, intrinsics: np.ndarray) -> None:
        pts_3d = np.zeros((10, 3), dtype=np.float32)
        pts_2d = np.zeros((10, 2), dtype=np.float32)
        transform, inliers = solve_pnp_ransac(pts_3d, pts_2d, intrinsics)
        assert np.allclose(transform, np.eye(4))
        assert not np.any(inliers)
