"""
Utilities for reading, scaling, centering, and analyzing 3D CAD meshes.

Provides modular functions (`read_mesh`, `scale_mesh`, `center_mesh`) alongside a
combined high-level composite pipeline (`load_mesh`) and auxiliary tools (`to_single_mesh`, `compute_mesh_diameter`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Delayed/lazy imports for trimesh to avoid bloating base library imports
_trimesh_installed = True
try:
    import trimesh
except ImportError:
    _trimesh_installed = False


@dataclass
class MeshArrays:
    """
    Data container holding vertex position, normal, index, and texture render buffers.

    Attributes:
        pos: Vertex coordinate array of shape (V, 3) in float32.
        faces: Triangle element index array of shape (F, 3) in int32.
        vnormals: Vertex unit normal array of shape (V, 3) in float32.
        tex: Optional normalized RGB texture map array of shape (1, H, W, 3) in float32.
        uv: Optional normalized UV coordinate array of shape (V, 2) in float32.
        vertex_color: Optional normalized vertex color array of shape (V, 3) in float32.
    """

    pos: np.ndarray
    faces: np.ndarray
    vnormals: np.ndarray
    tex: np.ndarray | None = None
    uv: np.ndarray | None = None
    vertex_color: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Convert container attributes to a dictionary, omitting None values.

        Returns:
            Dictionary mapping buffer names to non-null arrays.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}



def _check_trimesh():
    """Checks that the optional trimesh dependency is installed."""
    if not _trimesh_installed:
        raise ImportError(
            "Mesh processing functionality requires optional rendering dependencies ('trimesh'). "
            "Install them with: uv sync --extra render (or pip install 'spatialhub[render]')"
        )


def to_single_mesh(scene_or_mesh: Any) -> trimesh.Trimesh:
    """
    Convert a Trimesh scene or mesh into a single concatenated Trimesh object.

    Args:
        scene_or_mesh: Input trimesh.Scene or trimesh.Trimesh instance.

    Returns:
        Concatenated single trimesh.Trimesh object.

    Raises:
        ValueError: If the loaded trimesh.Scene is empty.
        TypeError: If input is not a trimesh.Trimesh or trimesh.Scene.
    """
    _check_trimesh()

    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError("The loaded trimesh Scene is empty.")
        # Concatenate geometries
        mesh = scene_or_mesh.to_geometry()
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
        return mesh

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh

    raise TypeError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(scene_or_mesh)}")


def read_mesh(
    mesh_input: str | Path | trimesh.Trimesh | trimesh.Scene,
) -> trimesh.Trimesh:
    """
    Read and parse arbitrary 3D mesh inputs into a single Trimesh instance.

    Args:
        mesh_input: File path (str or Path), pre-loaded trimesh.Trimesh, or trimesh.Scene.

    Returns:
        A standalone trimesh.Trimesh instance.
    """
    _check_trimesh()

    if isinstance(mesh_input, (str, Path)):
        loaded = trimesh.load(str(mesh_input))
        return to_single_mesh(loaded)

    return to_single_mesh(mesh_input).copy()


def scale_mesh(
    mesh: trimesh.Trimesh,
    model_unit: str | float = "m",
) -> trimesh.Trimesh:
    """
    Scale a 3D mesh from a specified unit system or numeric multiplier into meters.

    Supported units:
        - "m": Meters (scale factor = 1.0)
        - "cm": Centimeters (scale factor = 0.01)
        - "mm": Millimeters (scale factor = 0.001)
        - numeric (float/int): Custom scale factor multiplied directly into vertex coordinates.

    Args:
        mesh: Input trimesh.Trimesh object.
        model_unit: Target unit identifier ('m', 'cm', 'mm') or custom numeric scale multiplier.

    Returns:
        The scaled trimesh.Trimesh object in meter units.
    """
    _check_trimesh()
    scaled_mesh = mesh.copy()

    if isinstance(model_unit, (int, float)):
        scale_factor = float(model_unit)
    elif isinstance(model_unit, str):
        unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
        scale_factor = unit_map.get(model_unit.lower().strip(), 1.0)
    else:
        raise TypeError("model_unit must be a string identifier ('m', 'cm', 'mm') or numeric scale factor.")

    if scale_factor != 1.0:
        scaled_mesh.apply_scale(scale_factor)
        logger.info(f"Scaled CAD mesh by factor {scale_factor} to convert to meters.")

    return scaled_mesh


def center_mesh(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    Translate the mesh so its bounding box centroid sits at origin (0, 0, 0).

    Args:
        mesh: Input trimesh.Trimesh instance.

    Returns:
        Tuple of (centered_mesh, translation_offset_applied).
    """
    _check_trimesh()
    centered_mesh = mesh.copy()
    bbox_center = centered_mesh.bounding_box.centroid
    offset = -bbox_center
    centered_mesh.apply_translation(offset)
    logger.info(f"Centered mesh at origin (offset applied: {offset}).")
    return centered_mesh, offset


def compute_mesh_diameter(mesh: trimesh.Trimesh, n_sample: int = 10000) -> float:
    """
    Auxiliary Operation: Compute 3D bounding diameter of a mesh (maximum Euclidean distance between points).

    Args:
        mesh: Input trimesh.Trimesh object.
        n_sample: Maximum surface points to sample if vertex count is very large.

    Returns:
        Bounding diameter in meters as a float.
    """
    _check_trimesh()
    pts = mesh.vertices
    if len(pts) > n_sample:
        pts = trimesh.sample.sample_surface(mesh, n_sample)[0]

    # Convex hull points provide exact maximum pairwise Euclidean distance
    hull_pts = mesh.convex_hull.vertices if hasattr(mesh, "convex_hull") else pts
    dists = np.linalg.norm(hull_pts[:, None, :] - hull_pts[None, :, :], axis=-1)
    return float(dists.max())


def load_mesh(
    mesh_input: str | Path | trimesh.Trimesh | trimesh.Scene,
    model_unit: str | float = "m",
    center: bool = True,
) -> trimesh.Trimesh:
    """
    Read, scale to meters, and optionally center a 3D mesh at (0, 0, 0).

    Args:
        mesh_input: File path string, Path object, or pre-loaded trimesh.Trimesh / trimesh.Scene.
        model_unit: Unit system ('m', 'cm', 'mm') or custom numeric scale factor.
        center: Whether to center the mesh bounding box at origin (0, 0, 0) (default: True).

    Returns:
        Standardized trimesh.Trimesh object in meter units.
    """
    _check_trimesh()

    # Read
    mesh = read_mesh(mesh_input)

    # Scale
    mesh = scale_mesh(mesh, model_unit=model_unit)

    # Center
    if center:
        mesh, _ = center_mesh(mesh)

    return mesh


def compute_obb(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the oriented bounding box (OBB) corners, dimensions, and transformation matrix.

    Transforms the canonical OBB corners back into the input mesh's coordinate frame, allowing tight wireframe projection without requiring post-hoc pose modification.

    Args:
        mesh: Input trimesh.Trimesh instance.

    Returns:
        corners_in_mesh: 3D corner coordinates of shape (8, 3) aligned to the object's principal axes.
        extents: 3D dimensions (width, height, depth) along principal axes of shape (3,).
        to_origin: 4x4 transformation matrix mapping from mesh space to canonical centered OBB space.
    """
    _check_trimesh()

    # Compute Oriented Bounding Box
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    to_origin = to_origin.astype(np.float32)
    extents = extents.astype(np.float32)

    # Canonical box centered at (0, 0, 0)
    bbox_canonical = np.stack([-extents / 2.0, extents / 2.0], axis=0)  # (2, 3)
    corners_canonical = trimesh.bounds.corners(bbox_canonical)          # (8, 3)

    # Transform corners from canonical OBB frame back into the mesh coordinate frame
    inv_to_origin = np.linalg.inv(to_origin)
    corners_in_mesh = (inv_to_origin[:3, :3] @ corners_canonical.T + inv_to_origin[:3, 3:4]).T.astype(np.float32)

    return corners_in_mesh, extents, to_origin


def look_at(cam_location: np.ndarray, target_point: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate camera-to-world 4x4 transform pointing camera at target_point from cam_location.

    Uses OpenCV camera coordinate convention (+X right, +Y down, +Z forward into scene).

    Args:
        cam_location: 3D camera location coordinate array of shape (3,).
        target_point: 3D look-at target point coordinate array of shape (3,). Defaults to [0.0, 0.0, 0.0].

    Returns:
        4x4 homogeneous transformation matrix in float32.
    """
    if target_point is None:
        target_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_location = np.asarray(cam_location, dtype=np.float32)
    target_point = np.asarray(target_point, dtype=np.float32)

    forward = target_point - cam_location
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm

    tmp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # Check if the forward vector is nearly parallel to standard up/down direction
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

    mat = np.stack((right, up, forward, cam_location), axis=-1)
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat.astype(np.float32)


def invert_transform(trans: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 rigid homogeneous transformation.

    Args:
        trans: 4x4 homogeneous transformation matrix of shape (4, 4) or (B, 4, 4).

    Returns:
        Inverted 4x4 transformation matrix matching input shape in float32.
    """
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 2 and trans.shape == (4, 4):
        rot = trans[:3, :3]
        t = trans[:3, 3]
        rot_inv = rot.T
        t_inv = -rot_inv @ t

        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = rot_inv
        output[:3, 3] = t_inv
        return output
    elif trans.ndim == 3 and trans.shape[1:] == (4, 4):
        rot = trans[:, :3, :3]
        t = trans[:, :3, 3:]
        rot_inv = rot.transpose(0, 2, 1)
        t_inv = -rot_inv @ t

        output = np.tile(np.eye(4, dtype=np.float32)[None], (len(trans), 1, 1))
        output[:, :3, :3] = rot_inv
        output[:, :3, 3:] = t_inv
        return output
    else:
        raise ValueError(f"Expected transformation matrix of shape (4, 4) or (B, 4, 4), got {trans.shape}")


def sample_sphere_poses(
    num_viewpoints: int = 42,
    radius: float = 1.0,
    sampling_method: str = "fibonacci",
    subdivisions: int | None = None,
    pose_type: str = "object_pose",
) -> np.ndarray:
    """
    Generate evenly distributed camera or object poses around a sphere.

    Supports both Fibonacci spiral sphere sampling and subdivided icosphere sampling.

    Args:
        num_viewpoints: Target count of viewpoints to generate (default: 42).
        radius: Sphere radius / distance from model centroid in meters (default: 1.0).
        sampling_method: Spherical distribution method: ``"fibonacci"`` or ``"icosphere"``.
        subdivisions: Optional fixed icosphere subdivision level (only used if ``sampling_method="icosphere"``).
        pose_type: Coordinate frame interpretation:
            - ``"object_pose"``: Object-to-camera / world-to-camera transform (w2c).
            - ``"camera_pose"``: Camera-to-world / camera-in-object transform (c2w).

    Returns:
        NumPy array of shape (N, 4, 4) containing homogeneous transformation matrices in float32.
    """
    if sampling_method == "fibonacci":
        if num_viewpoints <= 1:
            raise ValueError("num_viewpoints must be greater than 1.")

        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

        for i in range(num_viewpoints):
            y = 1.0 - (i / float(num_viewpoints - 1)) * 2.0  # y goes from 1 to -1
            r_at_y = np.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            x = np.cos(theta) * r_at_y
            z = np.sin(theta) * r_at_y
            points.append([x, y, z])

        points = np.array(points, dtype=np.float32) * float(radius)

    elif sampling_method == "icosphere":
        _check_trimesh()
        if subdivisions is not None:
            mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
        else:
            subdivision = 1
            while True:
                mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=float(radius))
                if mesh.vertices.shape[0] >= num_viewpoints:
                    break
                subdivision += 1
        points = mesh.vertices.astype(np.float32)

    else:
        raise ValueError(f"Unsupported sampling_method: '{sampling_method}'. Must be 'fibonacci' or 'icosphere'.")

    poses = []
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for pt in points:
        c2w = look_at(pt, target)
        if pose_type == "object_pose":
            w2c = invert_transform(c2w)
            poses.append(w2c)
        elif pose_type == "camera_pose":
            poses.append(c2w)
        else:
            raise ValueError(f"Unsupported pose_type: '{pose_type}'. Must be 'object_pose' or 'camera_pose'.")

    return np.array(poses, dtype=np.float32)


def prepare_mesh_arrays(
    mesh: trimesh.Trimesh,
    max_tex_size: int | None = None,
    flip_uv: bool = True,
) -> MeshArrays:
    """
    Extract contiguous vertex, normal, UV, and texture buffer arrays from a CAD mesh.

    Args:
        mesh: Input trimesh.Trimesh instance.
        max_tex_size: Optional maximum texture dimension limit in pixels.
        flip_uv: Whether to invert vertical UV coordinates (1 - V) for OpenGL convention (default: True).

    Returns:
        MeshArrays container containing formatted float32 and int32 NumPy arrays.
    """
    _check_trimesh()

    pos = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    vnormals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    tex: np.ndarray | None = None
    uv: np.ndarray | None = None
    vertex_color: np.ndarray | None = None

    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.material.image is not None:
        img = np.array(mesh.visual.material.image.convert("RGB"), dtype=np.float32)
        img = img[..., :3]

        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = float(max_tex_size) / float(max_size)
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)

        tex = np.ascontiguousarray(img[np.newaxis, ...] / 255.0, dtype=np.float32)

        raw_uv = np.array(mesh.visual.uv, dtype=np.float32).copy()
        if flip_uv:
            raw_uv[:, 1] = 1.0 - raw_uv[:, 1]
        uv = np.ascontiguousarray(raw_uv, dtype=np.float32)
    else:
        raw_colors = getattr(mesh.visual, "vertex_colors", None)
        if raw_colors is None or len(raw_colors) == 0:
            logger.debug("Mesh lacks vertex colors; defaulting to neutral gray (128, 128, 128).")
            raw_colors = np.full((len(mesh.vertices), 3), 128, dtype=np.uint8)

        vertex_color = np.ascontiguousarray(
            raw_colors[..., :3].astype(np.float32) / 255.0,
            dtype=np.float32,
        )

    return MeshArrays(
        pos=pos,
        faces=faces,
        vnormals=vnormals,
        tex=tex,
        uv=uv,
        vertex_color=vertex_color,
    )


