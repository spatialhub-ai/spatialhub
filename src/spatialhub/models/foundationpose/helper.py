"""
Geometric transformation, viewpoint sampling, and mesh buffer utilities for FoundationPose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from spatialhub.utils.mesh import sample_sphere_poses

logger = logging.getLogger(__name__)

# Delayed import for optional rendering dependency
_trimesh_installed = True
try:
    import trimesh
except ImportError:
    _trimesh_installed = False

from spatialhub.utils.mesh import MeshArrays, prepare_mesh_arrays

random_seed = 0
np.random.seed(random_seed)


def _check_trimesh() -> None:

    """
    Verify that trimesh is installed prior to executing mesh generation functions.

    Raises:
        ImportError: If trimesh is not installed in the environment.
    """
    if not _trimesh_installed:
        raise ImportError(
            "FoundationPose helper functions require optional rendering dependencies ('trimesh'). "
            "Install them with: uv sync --extra render (or pip install 'spatialhub[render]')"
        )


def cluster_poses(
    angle_diff: float,
    dist_diff: float,
    poses_in: np.ndarray,
    symmetry_tfs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cluster candidate poses based on translation distance and rotational angular distance.

    Args:
        angle_diff: Angular threshold in degrees.
        dist_diff: Distance threshold in meters.
        poses_in: Array of candidate poses of shape (N, 4, 4).
        symmetry_tfs: Optional array of object symmetry matrices of shape (M, 4, 4).

    Returns:
        Filtered cluster center poses of shape (K, 4, 4) in float32.
    """
    if len(poses_in) == 0:
        return np.empty((0, 4, 4), dtype=np.float32)

    poses_in = np.asarray(poses_in, dtype=np.float32)
    if symmetry_tfs is None:
        symmetry_tfs = np.eye(4, dtype=np.float32)[None]
    else:
        symmetry_tfs = np.asarray(symmetry_tfs, dtype=np.float32)
        if symmetry_tfs.ndim == 2:
            symmetry_tfs = symmetry_tfs[None, ...]

    poses_out = [poses_in[0]]
    radian_thres = float(angle_diff / 180.0 * np.pi)

    for i in range(1, len(poses_in)):
        isnew = True
        cur_pose = poses_in[i]
        t1 = cur_pose[:3, 3]

        for cluster in poses_out:
            t0 = cluster[:3, 3]
            if np.linalg.norm(t0 - t1) >= dist_diff:
                continue

            R_cluster = cluster[:3, :3]

            for tf in symmetry_tfs:
                cur_pose_tmp = cur_pose @ tf
                R_tmp = cur_pose_tmp[:3, :3]
                trace = np.trace(R_tmp @ R_cluster.T)
                cos_val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                ang = np.arccos(cos_val)
                if ang < radian_thres:
                    isnew = False
                    break
            if not isnew:
                break

        if isnew:
            poses_out.append(cur_pose)

    return np.array(poses_out, dtype=np.float32)


def make_rotation_grid(
    min_n_views: int = 40,
    inplane_step: float = 60.0,
    symmetry_tfs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate candidate 3D rotation grid sampled over an icosphere with in-plane rotations.

    Args:
        min_n_views: Minimum spherical viewpoints (default: 40).
        inplane_step: In-plane roll rotation step angle in degrees (default: 60.0).
        symmetry_tfs: Optional symmetry transformation matrices of shape (M, 4, 4).

    Returns:
        Candidate pose array of shape (K, 4, 4) in float32.
    """
    cam_in_obs = sample_sphere_poses(num_viewpoints=min_n_views, radius=1.0, sampling_method="icosphere", subdivisions=None, pose_type="camera_pose")
    rot_grid = []

    for cam_in_ob in cam_in_obs:
        for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
            R_inplane = np.eye(4, dtype=np.float32)
            cos_a, sin_a = np.cos(inplane_rot), np.sin(inplane_rot)
            R_inplane[0, 0] = cos_a
            R_inplane[0, 1] = -sin_a
            R_inplane[1, 0] = sin_a
            R_inplane[1, 1] = cos_a

            cam_ob_rot = cam_in_ob @ R_inplane
            ob_in_cam = np.linalg.inv(cam_ob_rot)
            rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid, dtype=np.float32)
    rot_grid = cluster_poses(30.0, 99999.0, rot_grid, symmetry_tfs)
    return rot_grid


def guess_translation(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Estimate 3D centroid translation from 2D binary mask and metric depth map.

    Args:
        depth: Metric depth map array of shape (H, W).
        mask: Binary object mask array of shape (H, W).
        K: 3x3 camera intrinsic matrix.

    Returns:
        Translation vector [X, Y, Z] in meters as float32 array of shape (3,).
    """
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return np.zeros(3, dtype=np.float32)

    uc = (us.min() + us.max()) / 2.0
    vc = (vs.min() + vs.max()) / 2.0

    valid = (mask > 0) & (depth >= 0.001)
    if not valid.any():
        return np.zeros(3, dtype=np.float32)

    zc = float(np.median(depth[valid]))
    K_inv = np.linalg.inv(K)
    center = (K_inv @ np.array([uc, vc, 1.0], dtype=np.float32).reshape(3, 1)) * zc
    return center.reshape(3).astype(np.float32)


def egocentric_delta_pose_to_pose(
    A_in_cam: np.ndarray,
    trans_delta: np.ndarray,
    rot_mat_delta: np.ndarray,
) -> np.ndarray:
    """
    Apply translation and rotation deltas to 4x4 poses in egocentric camera space.

    Args:
        A_in_cam: Input pose matrix of shape (4, 4) or array of shape (B, 4, 4).
        trans_delta: Translation delta of shape (3,) or (B, 3).
        rot_mat_delta: Rotation matrix delta of shape (3, 3) or (B, 3, 3).

    Returns:
        Updated pose matrix matching input shape of A_in_cam in float32.
    """
    A_in_cam = np.asarray(A_in_cam, dtype=np.float32)
    trans_delta = np.asarray(trans_delta, dtype=np.float32)
    rot_mat_delta = np.asarray(rot_mat_delta, dtype=np.float32)

    if A_in_cam.ndim == 2:
        B_in_cam = np.eye(4, dtype=np.float32)
        B_in_cam[:3, 3] = A_in_cam[:3, 3] + trans_delta
        B_in_cam[:3, :3] = rot_mat_delta @ A_in_cam[:3, :3]
        return B_in_cam
    else:
        B = len(A_in_cam)
        B_in_cam = np.tile(np.eye(4, dtype=np.float32), (B, 1, 1))
        B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
        B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
        return B_in_cam


def compute_crop_window_tf_batch(
    pts: np.ndarray | None = None,
    H: int = 480,
    W: int = 640,
    poses: np.ndarray = None,
    K: np.ndarray = None,
    crop_ratio: float = 1.2,
    out_size: tuple[int, int] = (160, 160),
    mesh_diameter: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D perspective affine crop transformation matrices for candidate poses.

    Args:
        pts: Optional 3D vertex array (unused if mesh_diameter is provided).
        H: Source image height in pixels.
        W: Source image width in pixels.
        poses: Array of candidate 4x4 poses of shape (B, 4, 4).
        K: 3x3 camera intrinsic matrix.
        crop_ratio: Spatial expansion factor around candidate bounding box.
        out_size: Output spatial crop resolution as (width, height).
        mesh_diameter: 3D bounding mesh diameter in meters.

    Returns:
        Tuple of (tfs, bbox2d_ori) where tfs is (B, 3, 3) float32 transformations
        and bbox2d_ori is (B, 4) bounding box bounds [left, top, right, bottom].
    """
    poses = np.asarray(poses, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    B = len(poses)

    radius = float(mesh_diameter * crop_ratio / 2.0)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
        ],
        dtype=np.float32,
    )

    pts_3d = poses[:, :3, 3][:, None, :] + offsets[None, :, :]
    pts_reshaped = pts_3d.reshape(-1, 3)

    projected = (K @ pts_reshaped.T).T
    uvs = projected[:, :2] / projected[:, 2:3]
    uvs = uvs.reshape(B, 5, 2)

    center = uvs[:, 0]
    radius_2d = np.abs(uvs - center[:, None, :]).reshape(B, -1).max(axis=-1)

    left = np.round(center[:, 0] - radius_2d)
    right = np.round(center[:, 0] + radius_2d)
    top = np.round(center[:, 1] - radius_2d)
    bottom = np.round(center[:, 1] + radius_2d)

    bbox2d_ori = np.stack([left, top, right, bottom], axis=-1).astype(np.float32)

    tf = np.tile(np.eye(3, dtype=np.float32)[None], (B, 1, 1))
    tf[:, 0, 2] = -left
    tf[:, 1, 2] = -top

    new_tf = np.tile(np.eye(3, dtype=np.float32)[None], (B, 1, 1))
    dx = np.maximum(right - left, 1.0)
    dy = np.maximum(bottom - top, 1.0)
    new_tf[:, 0, 0] = float(out_size[0]) / dx
    new_tf[:, 1, 1] = float(out_size[1]) / dy

    tfs = new_tf @ tf
    return tfs.astype(np.float32), bbox2d_ori


def transform_batch(
    rgbAs: np.ndarray,
    rgbBs: np.ndarray,
    xyz_mapAs: np.ndarray,
    xyz_mapBs: np.ndarray,
    poseA: np.ndarray,
    mesh_diameter: float,
    normalize_xyz: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Format and normalize RGB and coordinate crop tensors for model input.

    Args:
        rgbAs: Rendered synthetic RGB crops of shape (B, 3, H, W).
        rgbBs: Observed real RGB crops of shape (B, 3, H, W).
        xyz_mapAs: Rendered synthetic coordinate map crops of shape (B, 3, H, W).
        xyz_mapBs: Observed real coordinate map crops of shape (B, 3, H, W).
        poseA: Candidate pose array of shape (B, 4, 4).
        mesh_diameter: 3D bounding mesh diameter in meters.
        normalize_xyz: Whether to scale 3D coordinates by mesh radius.

    Returns:
        Tuple of (A_tensor, B_tensor) of shape (B, 6, H, W) in float32.
    """
    bs = len(rgbAs)
    rgbAs = np.asarray(rgbAs, dtype=np.float32) / 255.0 if rgbAs.max() > 1.0 else np.asarray(rgbAs, dtype=np.float32)
    rgbBs = np.asarray(rgbBs, dtype=np.float32) / 255.0 if rgbBs.max() > 1.0 else np.asarray(rgbBs, dtype=np.float32)

    xyz_mapAs = np.asarray(xyz_mapAs, dtype=np.float32)
    xyz_mapBs = np.asarray(xyz_mapBs, dtype=np.float32)
    poseA = np.asarray(poseA, dtype=np.float32)

    if normalize_xyz:
        invalidA = xyz_mapAs[:, 2:3] < 0.001
        invalidB = xyz_mapBs[:, 2:3] < 0.001

    center = poseA[:, :3, 3].reshape(bs, 3, 1, 1)
    xyz_mapAs = xyz_mapAs - center
    xyz_mapBs = xyz_mapBs - center

    if normalize_xyz:
        radius = float(mesh_diameter / 2.0)
        xyz_mapAs /= radius
        xyz_mapBs /= radius

        invalidA = invalidA | (np.abs(xyz_mapAs) >= 2)
        xyz_mapAs[invalidA] = 0

        invalidB = invalidB | (np.abs(xyz_mapBs) >= 2)
        xyz_mapBs[invalidB] = 0

    A = np.concatenate([rgbAs, xyz_mapAs], axis=1).astype(np.float32)
    B = np.concatenate([rgbBs, xyz_mapBs], axis=1).astype(np.float32)

    return A, B


def rotvec_to_rotmat_np(rotvec: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert 3D rotation vectors to 3x3 rotation matrices using Rodrigues' formula.

    Args:
        rotvec: Rotation vector array of shape (B, 3).
        epsilon: Small value to prevent division by zero.

    Returns:
        Rotation matrix array of shape (B, 3, 3) in float32.
    """
    rotvec = np.asarray(rotvec, dtype=np.float32)
    batch_shape = rotvec.shape[:-1]

    rotvec_flat = rotvec.reshape(-1, 3)

    theta = np.linalg.norm(rotvec_flat, axis=-1)
    is_angle_small = theta < epsilon

    axis = rotvec_flat / np.maximum(theta[..., np.newaxis], epsilon)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    one_minus_cos_theta = 1.0 - cos_theta

    xs_rod = kx * sin_theta
    ys_rod = ky * sin_theta
    zs_rod = kz * sin_theta

    xyc = kx * ky * one_minus_cos_theta
    xzc = kx * kz * one_minus_cos_theta
    yzc = ky * kz * one_minus_cos_theta
    xxc = kx**2 * one_minus_cos_theta
    yyc = ky**2 * one_minus_cos_theta
    zzc = kz**2 * one_minus_cos_theta

    R_rodrigues = np.stack(
        [
            1.0 - yyc - zzc,
            xyc - zs_rod,
            xzc + ys_rod,
            xyc + zs_rod,
            1.0 - xxc - zzc,
            -xs_rod + yzc,
            xzc - ys_rod,
            xs_rod + yzc,
            1.0 - xxc - yyc,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)

    xs_first, ys_first, zs_first = rotvec_flat[:, 0], rotvec_flat[:, 1], rotvec_flat[:, 2]
    one = np.ones_like(xs_first)

    R_first_order = np.stack(
        [
            one,
            -zs_first,
            ys_first,
            zs_first,
            one,
            -xs_first,
            -ys_first,
            xs_first,
            one,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)

    R = np.where(is_angle_small[:, np.newaxis, np.newaxis], R_first_order, R_rodrigues)
    return R.reshape(*batch_shape, 3, 3)


def special_gramschmidt_np(rot_6d: np.ndarray) -> np.ndarray:
    """
    Orthogonalize 6D rotation representations into 3x3 rotation matrices via Gram-Schmidt process.

    Args:
        rot_6d: 6D rotation vector array of shape (B, 6).

    Returns:
        Rotation matrix array of shape (B, 3, 3) in float32.
    """
    rot_6d = np.asarray(rot_6d, dtype=np.float32)
    v1 = rot_6d[:, :3]
    v2 = rot_6d[:, 3:]

    u1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    u2 = v2 - np.sum(u1 * v2, axis=-1, keepdims=True) * u1
    u2 = u2 / np.linalg.norm(u2, axis=-1, keepdims=True)
    u3 = np.cross(u1, u2)

    return np.stack([u1, u2, u3], axis=-1)


def prepare_mesh_for_rendering(
    mesh: trimesh.Trimesh,
    max_tex_size: int | None = None,
    flip_uv: bool = True,
) -> MeshArrays:
    """
    Extract contiguous vertex, normal, UV, and texture buffer arrays from a mesh.

    Args:
        mesh: Input mesh instance.
        max_tex_size: Optional maximum texture dimension limit in pixels.
        flip_uv: Whether to invert vertical UV coordinates (1 - V).

    Returns:
        MeshArrays container containing formatted float32 and int32 arrays.
    """
    return prepare_mesh_arrays(mesh=mesh, max_tex_size=max_tex_size, flip_uv=flip_uv)

