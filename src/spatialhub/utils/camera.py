"""Camera geometry and projection utilities."""

from __future__ import annotations

import logging

import numpy as np

from .transform import transform_points
from .validation import validate_array

logger = logging.getLogger(__name__)


def scale_intrinsics(
    K: np.ndarray,
    orig_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """Scale camera intrinsic parameters proportionally to image dimension changes.

    Adjusts focal lengths (fx, fy) and principal point (cx, cy) based on the
    ratio of new dimensions to original dimensions.

    Args:
        K: Camera intrinsic matrix of shape (3, 3) or (..., 3, 3) in float32.
        orig_size: Original image dimensions as (width, height).
        new_size: Target image dimensions as (width, height).

    Returns:
        Scaled camera intrinsic matrix of shape (3, 3) or (..., 3, 3) in float32.
    """
    K_arr = np.asarray(K, dtype=np.float32)
    validate_array("K", K_arr, shape_suffix=(3, 3))

    K_scaled = K_arr.copy()
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = float(new_w) / float(orig_w)
    scale_y = float(new_h) / float(orig_h)

    K_scaled[..., 0, 0] *= scale_x
    K_scaled[..., 1, 1] *= scale_y
    K_scaled[..., 0, 2] *= scale_x
    K_scaled[..., 1, 2] *= scale_y

    return K_scaled


def reproject_depth_to_3d(
    depth: np.ndarray,
    K: np.ndarray,
    z_min: float = 0.001,
    z_max: float = float("inf"),
) -> np.ndarray:
    """Reproject 2D depth map(s) into 3D camera-space coordinates.

    Mathematical reprojection:
        Given pixel coordinates (u, v) and depth Z = depth[..., v, u]:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[..., v, u]

    Coordinate frame convention:
        - +X axis points right.
        - +Y axis points downward.
        - +Z axis points forward along the optical axis into the scene.

    Args:
        depth: 2D depth map of shape (H, W) or batched depth maps of shape (..., H, W) in meters.
        K: Camera intrinsic matrix of shape (3, 3) or batched matrices of shape (..., 3, 3).
        z_min: Minimum valid depth threshold in meters (values < z_min are masked to 0).
        z_max: Maximum valid depth threshold in meters (values > z_max are masked to 0).

    Returns:
        3D coordinate array of shape (H, W, 3) or (..., H, W, 3) in float32. Invalid/masked points are (0, 0, 0).

    Raises:
        ValueError: If depth or K have incompatible shapes or dimensions.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    k_arr = np.asarray(K, dtype=np.float32)

    validate_array("K", k_arr, shape_suffix=(3, 3))
    if depth_arr.ndim < 2:
        raise ValueError(f"depth must have shape (H, W) or (..., H, W), got {depth_arr.shape}")

    h, w = depth_arr.shape[-2:]
    vs, us = np.meshgrid(
        np.arange(0, h, dtype=np.float32),
        np.arange(0, w, dtype=np.float32),
        indexing="ij",
    )

    fx = k_arr[..., 0:1, 0:1]
    fy = k_arr[..., 1:2, 1:2]
    cx = k_arr[..., 0:1, 2:3]
    cy = k_arr[..., 1:2, 2:3]

    zs = depth_arr
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy

    xyz = np.stack((xs, ys, zs), axis=-1).astype(np.float32)
    invalid_mask = (zs < z_min) | (zs > z_max)
    xyz[invalid_mask] = 0.0

    return xyz


def project_points_to_image(
    points: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D points in object or camera space onto the 2D image plane.

    Args:
        points: 3D point coordinates of shape (N, 3) or (..., N, 3) in meters.
            If pose is None, points must already be in camera coordinate frame (X-right, Y-down, Z-forward).
            If pose is provided, points are in an object/source frame and will be transformed to camera space.
        intrinsics: Camera intrinsic calibration matrix (3, 3) or batched (..., 3, 3).
        pose: Optional rigid transformation matrix T_c<-o of shape (4, 4) or (..., 4, 4).
            Transforms points from object/source frame to camera frame: p_cam = R @ p_obj + t.

    Returns:
        Tuple of (points_2d, valid_depth_mask):
            - points_2d: Projected pixel coordinates (x, y) of shape (N, 2) or (..., N, 2) as float32.
            - valid_depth_mask: Boolean mask of shape (N,) or (..., N) indicating points strictly in front of the camera (Z > 0).

    Raises:
        ValueError: If array dimensions or shapes do not conform to expected geometric formats.
    """
    pts = np.asarray(points, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    validate_array("points", pts, shape_suffix=(3,))
    validate_array("intrinsics", k, shape_suffix=(3, 3))

    if pose is not None:
        t_arr = np.asarray(pose, dtype=np.float32)
        validate_array("pose", t_arr, shape_suffix=(4, 4))
        pts_cam = transform_points(pts, t_arr)
    else:
        pts_cam = pts

    z = pts_cam[..., 2:3]
    valid_depth = (z[..., 0] > 0) & np.isfinite(z[..., 0])

    z_safe = np.where(z != 0, z, 1e-6)
    pts_proj = np.matmul(pts_cam, k.swapaxes(-1, -2))
    pts_2d = pts_proj[..., :2] / z_safe

    if valid_depth.shape != pts_2d.shape[:-1]:
        valid_depth = np.broadcast_to(valid_depth, pts_2d.shape[:-1])

    return pts_2d, valid_depth


def intrinsics_to_projection_matrix(
    K: np.ndarray,
    height: int,
    width: int,
    znear: float = 0.001,
    zfar: float = 100.0,
    window_coords: str = "y_down",
) -> np.ndarray:
    """Construct a 4x4 perspective projection matrix from pinhole camera intrinsics.

    Maps 3D camera-space coordinates into the canonical clip-space view volume
    ([-1, 1] normalized device coordinates). Supports vertical image axis conventions
    with origin at top-left ('y_down') or bottom-left ('y_up').

    Args:
        K: Camera intrinsic matrix of shape (3, 3) or (..., 3, 3) in float32.
        height: Viewport height in pixels.
        width: Viewport width in pixels.
        znear: Near clipping plane distance along the optical axis in meters.
        zfar: Far clipping plane distance along the optical axis in meters.
        window_coords: Vertical axis orientation in image space ('y_down' for
            top-left origin, 'y_up' for bottom-left origin).

    Returns:
        4x4 projection matrix of shape (4, 4) or (..., 4, 4) in float32.

    Raises:
        ValueError: If window_coords is not 'y_down' or 'y_up'.
    """
    k_arr = np.asarray(K, dtype=np.float32)
    validate_array("K", k_arr, shape_suffix=(3, 3))

    w = float(width)
    h = float(height)
    nc = float(znear)
    fc = float(zfar)
    x0 = 0.0
    y0 = 0.0

    depth = fc - nc
    q = -(fc + nc) / depth
    qn = -2.0 * (fc * nc) / depth

    proj = np.zeros((*k_arr.shape[:-2], 4, 4), dtype=np.float32)
    proj[..., 0, 0] = 2.0 * k_arr[..., 0, 0] / w
    proj[..., 0, 1] = -2.0 * k_arr[..., 0, 1] / w
    proj[..., 0, 2] = (-2.0 * k_arr[..., 0, 2] + w + 2.0 * x0) / w

    if window_coords == "y_up":
        proj[..., 1, 1] = -2.0 * k_arr[..., 1, 1] / h
        proj[..., 1, 2] = (-2.0 * k_arr[..., 1, 2] + h + 2.0 * y0) / h
    elif window_coords == "y_down":
        proj[..., 1, 1] = 2.0 * k_arr[..., 1, 1] / h
        proj[..., 1, 2] = (2.0 * k_arr[..., 1, 2] - h + 2.0 * y0) / h
    else:
        raise ValueError("window_coords must be either 'y_down' or 'y_up'.")

    proj[..., 2, 2] = q
    proj[..., 2, 3] = qn
    proj[..., 3, 2] = -1.0

    return proj


def _camera_pose_from_target(
    camera_location: np.ndarray | list[float],
    target_point: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Compute a 4x4 camera-to-world transformation matrix directed toward a target position.

    Constructs an orthonormal camera basis where the optical axis (+Z) points
    from camera_location toward target_point, with right (+X) and down (+Y) axes
    aligned orthogonally.

    Args:
        camera_location: 3D camera origin coordinates of shape (3,) in float32.
        target_point: 3D target coordinates of shape (3,) in float32. Defaults to (0, 0, 0).

    Returns:
        4x4 camera-to-world rigid transformation matrix of shape (4, 4) in float32.
    """
    if target_point is None:
        target_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_loc = np.asarray(camera_location, dtype=np.float32)
    tgt = np.asarray(target_point, dtype=np.float32)
    validate_array("camera_location", cam_loc, ndim=1, shape_suffix=(3,))
    validate_array("target_point", tgt, ndim=1, shape_suffix=(3,))

    forward = tgt - cam_loc
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm

    tmp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # avoid degenerate cross product when forward is nearly parallel to tmp
    if np.abs(np.dot(forward, tmp)) > 0.999:
        tmp = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    right = np.cross(tmp, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm

    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = up / up_norm

    mat = np.stack((right, up, forward, cam_loc), axis=-1)
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat.astype(np.float32)


def sample_sphere_points(
    num_points: int,
    radius: float = 1.0,
) -> np.ndarray:
    """Sample 3D Cartesian coordinates on a spherical surface via Fibonacci spiral.

    Distributes points across a sphere of specified radius using a spherical Fibonacci
    lattice.

    Args:
        num_points: Number of points to sample. Must be greater than 0.
        radius: Sphere radius in meters. Defaults to 1.0.

    Returns:
        Array of Cartesian coordinates of shape (num_points, 3) in float32.

    Raises:
        ValueError: If num_points <= 0.
    """
    if num_points <= 0:
        raise ValueError("num_points must be greater than 0.")

    if num_points == 1:
        return np.array([[0.0, 0.0, float(radius)]], dtype=np.float32)

    indices = np.arange(num_points, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    y = 1.0 - (indices / float(num_points - 1)) * 2.0
    r_at_y = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices

    x = np.cos(theta) * r_at_y
    z = np.sin(theta) * r_at_y

    points = np.stack([x, y, z], axis=1) * float(radius)
    return points.astype(np.float32)


def sample_sphere_poses(
    num_viewpoints: int = 42,
    radius: float = 1.0,
) -> np.ndarray:
    """Sample camera-to-world poses on a spherical surface via Fibonacci spiral.

    Positions cameras on a sphere using a spherical Fibonacci lattice and orients
    each camera coordinate frame so its optical axis (+Z) points toward the origin
    (0, 0, 0).

    Args:
        num_viewpoints: Number of viewpoints to sample. Must be greater than 0. Defaults to 42.
        radius: Sphere radius in meters. Defaults to 1.0.

    Returns:
        Array of 4x4 camera-to-world transformation matrices of shape (num_viewpoints, 4, 4) in float32.

    Raises:
        ValueError: If num_viewpoints <= 0.
    """
    if num_viewpoints <= 0:
        raise ValueError("num_viewpoints must be greater than 0.")

    points = sample_sphere_points(num_viewpoints, radius=float(radius))
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    poses = [_camera_pose_from_target(pt, target) for pt in points]
    return np.array(poses, dtype=np.float32)


def sample_icosphere_poses(
    subdivisions: int | None = None,
    min_n_views: int | None = None,
    radius: float = 1.0,
) -> np.ndarray:
    """Sample camera-to-world poses from geodesic icosphere vertices.

    Positions cameras at the vertices of a regular subdivided icosahedron and orients
    each camera coordinate frame so its optical axis (+Z) points toward the origin
    (0, 0, 0).

    Subdivision vertex counts follow V = 10 * 4^s + 2:
        - subdivisions=0: 12 viewpoints
        - subdivisions=1: 42 viewpoints
        - subdivisions=2: 162 viewpoints
        - subdivisions=3: 642 viewpoints

    Args:
        subdivisions: Geodesic subdivision level (0, 1, 2, ...).
        min_n_views: Minimum viewpoint count. Resolves the lowest subdivision level
            satisfying this threshold.
        radius: Sphere radius in meters. Defaults to 1.0.

    Returns:
        Array of 4x4 camera-to-world transformation matrices of shape (N, 4, 4) in float32.

    Raises:
        ValueError: If subdivisions < 0 or min_n_views <= 0.
        ImportError: If trimesh is not installed.
    """
    if subdivisions is not None:
        if subdivisions < 0:
            raise ValueError("subdivisions must be greater than or equal to 0.")
        subdivision_level = int(subdivisions)
    elif min_n_views is not None:
        if min_n_views <= 0:
            raise ValueError("min_n_views must be greater than 0.")
        subdivision_level = 0
        while True:
            v_count = 10 * (4 ** subdivision_level) + 2
            if v_count >= min_n_views:
                break
            subdivision_level += 1
    else:
        subdivision_level = 1

    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "sample_icosphere_poses requires 'trimesh'. "
            "Install with: uv sync --extra render (or pip install 'spatialhub[render]')"
        ) from None

    mesh = trimesh.creation.icosphere(subdivisions=subdivision_level, radius=float(radius))
    points = mesh.vertices.astype(np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    poses = [_camera_pose_from_target(pt, target) for pt in points]
    return np.array(poses, dtype=np.float32)


__all__ = [
    "sample_sphere_points",
    "intrinsics_to_projection_matrix",
    "project_points_to_image",
    "reproject_depth_to_3d",
    "sample_icosphere_poses",
    "sample_sphere_poses",
    "scale_intrinsics",
]


