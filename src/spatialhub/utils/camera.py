"""
Camera geometry utilities for intrinsic scaling, 3D depth reprojection, and OpenGL projection matrices.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def scale_camera_intrinsics(
    K: np.ndarray,
    orig_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """
    Scale a 3x3 camera intrinsic matrix based on image dimension scaling.

    When an image is resized from `orig_size=(W_orig, H_orig)` to `new_size=(W_new, H_new)`,
    the focal lengths (fx, fy) and principal points (cx, cy) must be scaled proportionally.

    Args:
        K: Camera intrinsic matrix of shape (3, 3):
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        orig_size: Original image dimensions as (width, height) or (W_orig, H_orig).
        new_size: New image dimensions as (width, height) or (W_new, H_new).

    Returns:
        Scaled 3x3 camera intrinsic matrix as a float32 NumPy array.
    """
    K_scaled = np.array(K, dtype=np.float32).copy()
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = float(new_w) / float(orig_w)
    scale_y = float(new_h) / float(orig_h)

    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled


def reproject_depth_to_3d(
    depth: np.ndarray,
    K: np.ndarray,
    uvs: np.ndarray | None = None,
    z_min: float = 0.001,
    z_max: float = float("inf"),
) -> np.ndarray:
    """
    Reproject a 2D depth map into 3D camera-space coordinates.

    OpenCV Camera Coordinate System:
        - +X axis points to the RIGHT of the camera image.
        - +Y axis points DOWNWARDS towards the bottom of the camera image.
        - +Z axis points FORWARD along the optical axis into the scene.

    Mathematical Reprojection:
        Given pixel coordinates (u, v) and depth Z = depth[v, u]:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[v, u]

    Args:
        depth: 2D depth map array of shape (H, W), with depth values measured in meters.
        K: 3x3 camera intrinsic projection matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        uvs: Optional pixel coordinates array of shape (N, 2) [u, v]. If None, evaluates across all pixels.
        z_min: Minimum valid depth threshold in meters (values < z_min are masked out to 0).
        z_max: Maximum valid depth threshold in meters (values > z_max are masked out to 0).

    Returns:
        3D XYZ coordinate map array of shape (H, W, 3) in float32. Invalid/masked points are (0, 0, 0).
    """
    H, W = depth.shape[:2]
    invalid_mask = (depth < z_min) | (depth > z_max)

    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H, dtype=np.float32), np.arange(0, W, dtype=np.float32), indexing="ij")
    else:
        uvs_int = np.asarray(uvs).round().astype(int)
        us = uvs_int[:, 0].astype(np.float32)
        vs = uvs_int[:, 1].astype(np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if uvs is None:
        zs = depth
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        xyz_map = np.stack((xs, ys, zs), axis=-1).astype(np.float32)
        xyz_map[invalid_mask] = 0.0
        return xyz_map
    else:
        vs_int = vs.astype(int)
        us_int = us.astype(int)
        zs = depth[vs_int, us_int]
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        pts = np.stack((xs, ys, zs), axis=1).astype(np.float32)

        xyz_map = np.zeros((H, W, 3), dtype=np.float32)
        xyz_map[vs_int, us_int] = pts
        xyz_map[invalid_mask] = 0.0
        return xyz_map


def reproject_depth_to_3d_batch(
    depths: np.ndarray,
    Ks: np.ndarray,
    z_min: float = 0.001,
    z_max: float = float("inf"),
) -> np.ndarray:
    """
    Re-projection of 2D depth maps into 3D camera-space coordinates.

    OpenCV Camera Coordinate System:
        - +X right, +Y down, +Z forward into scene.

    Args:
        depths: NumPy array of shape (B, H, W) containing depth maps in meters.
        Ks: NumPy array of shape (B, 3, 3) containing 3x3 camera intrinsic matrices.
        z_min: Minimum valid depth threshold in meters.
        z_max: Maximum valid depth threshold in meters.

    Returns:
        Float32 NumPy array of shape (B, H, W, 3) representing 3D XYZ coordinate maps.
    """
    depths = np.asarray(depths, dtype=np.float32)
    Ks = np.asarray(Ks, dtype=np.float32)

    bs, H, W = depths.shape
    invalid_mask = (depths < z_min) | (depths > z_max)

    vs, us = np.meshgrid(
        np.arange(0, H, dtype=np.float32),
        np.arange(0, W, dtype=np.float32),
        indexing="ij",
    )
    # Broadcast (H, W) meshgrid to (B, H, W)
    us_batch = np.broadcast_to(us, (bs, H, W))
    vs_batch = np.broadcast_to(vs, (bs, H, W))

    fx = Ks[:, 0, 0][:, None, None]  # (B, 1, 1)
    fy = Ks[:, 1, 1][:, None, None]
    cx = Ks[:, 0, 2][:, None, None]
    cy = Ks[:, 1, 2][:, None, None]

    zs = depths
    xs = (us_batch - cx) * zs / fx
    ys = (vs_batch - cy) * zs / fy

    xyz_maps = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    xyz_maps[invalid_mask] = 0.0

    return xyz_maps


def create_perspective_projection_matrix(
    K: np.ndarray,
    height: int,
    width: int,
    znear: float = 0.001,
    zfar: float = 100.0,
    window_coords: str = "y_down",
) -> np.ndarray:
    """
    Convert a 3x3 OpenCV camera intrinsic matrix K into a 4x4 OpenGL clip-space projection matrix.

    Coordinate Conventions & Mapping:
        OpenCV intrinsic matrix K projects 3D points in (+X right, +Y down, +Z forward) into 2D window space.
        OpenGL rendering pipeline expects clip-space coordinates in (-1 to +1) NDC with (+X right, +Y up, -Z forward).

    Args:
        K: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        height: Image height H in pixels.
        width: Image width W in pixels.
        znear: Distance to the near clipping plane in meters.
        zfar: Distance to the far clipping plane in meters.
        window_coords: 'y_down' (standard OpenCV pixel origin top-left) or 'y_up' (OpenGL bottom-left).

    Returns:
        4x4 OpenGL projection matrix as a float32 NumPy array.
    """
    w = float(width)
    h = float(height)
    nc = float(znear)
    fc = float(zfar)
    x0 = 0.0
    y0 = 0.0

    depth = fc - nc
    q = -(fc + nc) / depth
    qn = -2.0 * (fc * nc) / depth

    if window_coords == "y_up":
        proj = np.array(
            [
                [2.0 * K[0, 0] / w, -2.0 * K[0, 1] / w, (-2.0 * K[0, 2] + w + 2.0 * x0) / w, 0.0],
                [0.0, -2.0 * K[1, 1] / h, (-2.0 * K[1, 2] + h + 2.0 * y0) / h, 0.0],
                [0.0, 0.0, q, qn],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
    elif window_coords == "y_down":
        proj = np.array(
            [
                [2.0 * K[0, 0] / w, -2.0 * K[0, 1] / w, (-2.0 * K[0, 2] + w + 2.0 * x0) / w, 0.0],
                [0.0, 2.0 * K[1, 1] / h, (2.0 * K[1, 2] - h + 2.0 * y0) / h, 0.0],
                [0.0, 0.0, q, qn],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("window_coords must be either 'y_down' or 'y_up'.")

    return proj


def convert_opencv_to_opengl_pose(pose_cv: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 rigid homogeneous camera pose transform between OpenCV and OpenGL camera frames using.

    Coordinate Space Transformation:
        - OpenCV: +X Right, +Y Down, +Z Forward into scene.
        - OpenGL: +X Right, +Y Up,   -Z Forward (Camera looks down -Z).

    Transformation Matrix T_gl_cv:
        [[ 1,  0,  0, 0],
         [ 0, -1,  0, 0],
         [ 0,  0, -1, 0],
         [ 0,  0,  0, 1]]

    Args:
        pose_cv: 4x4 homogeneous transformation matrix in OpenCV camera coordinates.

    Returns:
        4x4 homogeneous transformation matrix in OpenGL camera coordinates.
    """
    T_gl_cv = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    pose = np.array(pose_cv, dtype=np.float32)
    if pose.ndim == 2 and pose.shape == (4, 4):
        return T_gl_cv @ pose
    elif pose.ndim == 3 and pose.shape[1:] == (4, 4):
        return T_gl_cv[None] @ pose
    else:
        raise ValueError(f"Expected pose shape (4, 4) or (B, 4, 4), got {pose.shape}")

