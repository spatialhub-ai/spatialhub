"""Geometric estimation solvers and RANSAC routines."""

from __future__ import annotations

import cv2
import numpy as np

from .validation import validate_array


def estimate_relative_pose_2d(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray | None = None,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate relative camera pose between two views from 2D point correspondences.

    Estimates the Essential matrix using 5-point RANSAC and decomposes it into
    rotation and unit translation. Triangulated depth is verified to be positive
    in both camera frames.

    Coordinates map from camera frame A to camera frame B:
        P_B = transform[:3, :3] @ P_A + transform[:3, 3]

    Args:
        points_a: 2D pixel coordinates in image A of shape (N, 2).
        points_b: 2D pixel coordinates in image B of shape (N, 2).
        intrinsics_a: Camera A intrinsic matrix of shape (3, 3).
        intrinsics_b: Camera B intrinsic matrix of shape (3, 3). Defaults to intrinsics_a.
        threshold_px: Reprojection error threshold in pixels for RANSAC inliers.
        confidence: RANSAC confidence level.

    Returns:
        Tuple containing:
            - transform: (4, 4) transformation matrix in SE(3) as float32
              mapping points from camera A to camera B, with unit translation.
            - inlier_mask: Boolean array of shape (N,) indicating inliers.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If input shapes are invalid or fewer than 5 correspondences are provided.
    """
    validate_array("points_a", points_a, ndim=2, shape_suffix=(2,))
    validate_array("points_b", points_b, ndim=2, shape_suffix=(2,))
    validate_array("intrinsics_a", intrinsics_a, ndim=2, shape_suffix=(3, 3))
    validate_array("intrinsics_b", intrinsics_b, ndim=2, shape_suffix=(3, 3), optional=True)

    pts_a = np.ascontiguousarray(points_a, dtype=np.float64)
    pts_b = np.ascontiguousarray(points_b, dtype=np.float64)
    k_a = np.ascontiguousarray(intrinsics_a, dtype=np.float64)
    k_b = k_a if intrinsics_b is None else np.ascontiguousarray(intrinsics_b, dtype=np.float64)

    if pts_a.shape[0] != pts_b.shape[0]:
        raise ValueError(
            f"points_a and points_b must have matching lengths, got {pts_a.shape[0]} and {pts_b.shape[0]}"
        )
    if pts_a.shape[0] < 5:
        raise ValueError(
            f"At least 5 point correspondences required for essential matrix estimation, got {pts_a.shape[0]}"
        )

    # Normalize to canonical camera plane (z = 1) to handle fx != fy and K_a != K_b uniformly
    norm_a = np.empty_like(pts_a)
    norm_a[:, 0] = (pts_a[:, 0] - k_a[0, 2]) / k_a[0, 0]
    norm_a[:, 1] = (pts_a[:, 1] - k_a[1, 2]) / k_a[1, 1]

    norm_b = np.empty_like(pts_b)
    norm_b[:, 0] = (pts_b[:, 0] - k_b[0, 2]) / k_b[0, 0]
    norm_b[:, 1] = (pts_b[:, 1] - k_b[1, 2]) / k_b[1, 1]

    f_avg = float((k_a[0, 0] + k_a[1, 1] + k_b[0, 0] + k_b[1, 1]) / 4.0)
    norm_threshold = float(threshold_px / max(f_avg, 1e-6))

    try:
        essential_mat, r_mask = cv2.findEssentialMat(
            points1=norm_a,
            points2=norm_b,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=float(confidence),
            threshold=norm_threshold,
        )
    except cv2.error:
        return np.eye(4, dtype=np.float32), np.zeros(len(pts_a), dtype=bool)

    if (
        r_mask is None
        or essential_mat is None
        or essential_mat.ndim != 2
        or essential_mat.shape[1] != 3
        or essential_mat.shape[0] % 3 != 0
    ):
        return np.eye(4, dtype=np.float32), np.zeros(len(pts_a), dtype=bool)

    # 5-point solver can return multiple candidate solutions stacked vertically (3*k, 3)
    num_candidates = essential_mat.shape[0] // 3
    best_inliers = -1
    best_rot = np.eye(3, dtype=np.float32)
    best_trans = np.zeros(3, dtype=np.float32)
    best_mask = np.zeros(len(pts_a), dtype=bool)

    for i in range(num_candidates):
        e_cand = np.ascontiguousarray(essential_mat[i * 3 : (i + 1) * 3, :])
        cand_mask = r_mask.copy()
        try:
            num_inliers, rot, trans, c_mask = cv2.recoverPose(
                E=e_cand,
                points1=norm_a,
                points2=norm_b,
                focal=1.0,
                pp=(0.0, 0.0),
                mask=cand_mask,
            )
        except cv2.error:
            continue

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_rot = rot.astype(np.float32)
            best_trans = trans.ravel().astype(np.float32)
            best_mask = (c_mask.ravel() > 0) if c_mask is not None else np.zeros(len(pts_a), dtype=bool)

    if best_inliers <= 0:
        return np.eye(4, dtype=np.float32), np.zeros(len(pts_a), dtype=bool)

    norm_val = np.linalg.norm(best_trans)
    if norm_val > 1e-7:
        best_trans = best_trans / norm_val

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = best_rot
    transform[:3, 3] = best_trans

    return transform, best_mask


def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    reprojection_error_px: float = 3.0,
    confidence: float = 0.999,
    max_iterations: int = 100,
    flags: int = cv2.SOLVEPNP_EPNP,
    refine_lm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera pose from 3D-to-2D point correspondences using PnP RANSAC.

    Coordinates map from source reference frame to camera frame:
        P_cam = transform[:3, :3] @ P_source + transform[:3, 3]

    Args:
        points_3d: 3D point coordinates of shape (N, 3) in source frame.
        points_2d: Corresponding 2D pixel coordinates of shape (N, 2) in camera frame.
        intrinsics: Camera intrinsic calibration matrix of shape (3, 3).
        dist_coeffs: Distortion coefficients array. Defaults to None.
        reprojection_error_px: Maximum reprojection error in pixels for RANSAC inliers.
        confidence: RANSAC confidence level.
        max_iterations: Maximum RANSAC iterations.
        flags: PnP solver flag (e.g. cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE).
        refine_lm: If True, refines the pose using Levenberg-Marquardt optimization on inliers.

    Returns:
        Tuple containing:
            - transform: (4, 4) transformation matrix in SE(3) as float32
              mapping points from source frame to camera frame.
            - inlier_mask: Boolean array of shape (N,) indicating inliers.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If input shapes are invalid or fewer than 4 correspondences are provided.
    """
    validate_array("points_3d", points_3d, ndim=2, shape_suffix=(3,))
    validate_array("points_2d", points_2d, ndim=2, shape_suffix=(2,))
    validate_array("intrinsics", intrinsics, ndim=2, shape_suffix=(3, 3))
    validate_array("dist_coeffs", dist_coeffs, ndim=(1, 2), optional=True)

    pts_3d = np.ascontiguousarray(points_3d, dtype=np.float64)
    pts_2d = np.ascontiguousarray(points_2d, dtype=np.float64)
    k = np.ascontiguousarray(intrinsics, dtype=np.float64)
    dist = None if dist_coeffs is None else np.ascontiguousarray(dist_coeffs, dtype=np.float64)

    if pts_3d.shape[0] != pts_2d.shape[0]:
        raise ValueError(
            f"points_3d and points_2d must have matching lengths, got {pts_3d.shape[0]} and {pts_2d.shape[0]}"
        )
    if pts_3d.shape[0] < 4:
        raise ValueError(
            f"At least 4 point correspondences required for PnP, got {pts_3d.shape[0]}"
        )

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=k,
            distCoeffs=dist,
            iterationsCount=int(max_iterations),
            reprojectionError=float(reprojection_error_px),
            confidence=float(confidence),
            flags=int(flags),
        )
    except cv2.error:
        return np.eye(4, dtype=np.float32), np.zeros(len(pts_3d), dtype=bool)

    if not success or rvec is None or tvec is None:
        return np.eye(4, dtype=np.float32), np.zeros(len(pts_3d), dtype=bool)

    inlier_mask = np.zeros(len(pts_3d), dtype=bool)
    if inliers is not None and len(inliers) > 0:
        inlier_indices = inliers.ravel()
        inlier_mask[inlier_indices] = True

        if refine_lm and len(inlier_indices) >= 4:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    objectPoints=pts_3d[inlier_indices],
                    imagePoints=pts_2d[inlier_indices],
                    cameraMatrix=k,
                    distCoeffs=dist,
                    rvec=rvec,
                    tvec=tvec,
                )
            except cv2.error:
                pass

    rot, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot.astype(np.float32)
    transform[:3, 3] = tvec.ravel().astype(np.float32)

    return transform, inlier_mask
