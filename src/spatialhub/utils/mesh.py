"""Utilities for loading, transforming, analyzing, and preparing 3D CAD meshes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .camera import sample_sphere_points
from .transform import extents_to_corners, invert_transform, transform_points

logger = logging.getLogger(__name__)


@dataclass
class MeshBuffers:
    """Data container holding vertex position, normal, index, and texture render buffers.

    Attributes:
        vertices: Vertex coordinate array of shape (V, 3) in float32.
        faces: Triangle element index array of shape (F, 3) in int32.
        normals: Vertex unit normal array of shape (V, 3) in float32.
        texture: Optional normalized RGB texture map array of shape (1, H, W, 3) in float32.
        uvs: Optional normalized UV coordinate array of shape (V, 2) in float32.
        vertex_colors: Optional normalized vertex color array of shape (V, 3) in float32.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    texture: np.ndarray | None = None
    uvs: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Convert container attributes to a dictionary, omitting None values.

        Returns:
            Dictionary mapping buffer names to non-null arrays.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OrientedBoundingBox:
    """Oriented bounding box (OBB) representation for a 3D mesh.

    Attributes:
        corners: 3D corner coordinates of shape (8, 3) in float32 in the mesh coordinate frame.
        extents: 3D full dimensions (dx, dy, dz) of shape (3,) in float32 along principal axes.
        to_origin: 4x4 transformation matrix of shape (4, 4) in float32 mapping points from mesh coordinate frame to canonical origin-centered box frame.
    """

    corners: np.ndarray
    extents: np.ndarray
    to_origin: np.ndarray | None = None

    def __iter__(self):
        """Yield corners, extents, and to_origin for tuple unpacking."""
        yield self.corners
        yield self.extents
        yield self.to_origin

    def as_dict(self) -> dict[str, np.ndarray]:
        """Convert OBB attributes to a dictionary, omitting None values.

        Returns:
            Dictionary mapping attribute names to non-null arrays.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


def load_mesh(
    path: str | Path
) -> trimesh.Trimesh:
    """Load a 3D mesh geometry file from disk.

    Args:
        path: Filesystem path to the 3D mesh file.

    Returns:
        Loaded mesh instance.

    Raises:
        TypeError: If path is not a string or Path.
        FileNotFoundError: If path does not point to an existing file.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")

    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    return trimesh.load_mesh(str(mesh_path))


def scale_mesh(
    mesh: trimesh.Trimesh,
    model_unit: str | float = "m",
) -> trimesh.Trimesh:
    """Scale a 3D mesh in-place from a specified unit system into meters.

    Supported units:
        - "m": Meters (scale factor = 1.0)
        - "cm": Centimeters (scale factor = 0.01)
        - "mm": Millimeters (scale factor = 0.001)
        - numeric (float/int): Custom numeric scale multiplier.

    Args:
        mesh: Input trimesh.Trimesh instance, modified in-place.
        model_unit: Target unit identifier ('m', 'cm', 'mm') or custom numeric scale multiplier.

    Returns:
        The scaled trimesh.Trimesh instance.

    Raises:
        TypeError: If model_unit is neither a string nor numeric.
        ValueError: If model_unit string is not a supported unit.
    """
    if isinstance(model_unit, (int, float)):
        scale_factor = float(model_unit)
    elif isinstance(model_unit, str):
        unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
        unit_key = model_unit.lower().strip()
        if unit_key not in unit_map:
            raise ValueError(
                f"Unsupported model_unit '{model_unit}'. Supported units are 'm', 'cm', 'mm', or a numeric scale factor."
            )
        scale_factor = unit_map[unit_key]
    else:
        raise TypeError(
            f"model_unit must be a string identifier ('m', 'cm', 'mm') or numeric scale factor, got {type(model_unit).__name__}"
        )

    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)
        logger.info(f"Scaled CAD mesh in-place by factor {scale_factor} to convert to meters.")

    return mesh


def center_mesh(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Translate the mesh in-place so its bounding box centroid sits at origin (0, 0, 0).

    Args:
        mesh: Input trimesh.Trimesh instance, modified in-place.

    Returns:
        Tuple of (centered_mesh, translation_offset_applied).
    """
    offset = np.asarray(-mesh.bounding_box.centroid, dtype=np.float32)
    mesh.apply_translation(offset)
    logger.info(f"Centered mesh in-place at origin (offset applied: {offset}).")
    return mesh, offset


def compute_mesh_diameter(
    mesh: trimesh.Trimesh, 
    num_dirs: int = 256
) -> float:
    """Compute the 3D bounding diameter of a mesh.

    Finds extreme boundary vertices by projecting onto uniformly sampled
    directions on a unit sphere, then calculates maximum pairwise distance.

    Args:
        mesh: Input trimesh.Trimesh instance.
        num_dirs: Number of directional vectors to sample on the sphere. Defaults to 256.

    Returns:
        Bounding diameter as a float.

    Raises:
        ValueError: If mesh contains no vertices.
    """
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    n_pts = len(pts)

    if n_pts == 0:
        raise ValueError("Cannot compute diameter of an empty mesh.")
    if n_pts == 1:
        return 0.0
    if n_pts <= num_dirs:
        diffs = pts[:, None, :] - pts[None, :, :]
        return float(np.sqrt(np.max(np.sum(diffs**2, axis=-1))))

    dirs = sample_sphere_points(num_dirs, radius=1.0)
    projs = pts @ dirs.T
    cand_indices = np.unique(np.concatenate([np.argmax(projs, axis=0), np.argmin(projs, axis=0)]))
    cand_pts = pts[cand_indices]

    diffs = cand_pts[:, None, :] - cand_pts[None, :, :]
    return float(np.sqrt(np.max(np.sum(diffs**2, axis=-1))))


def compute_obb(
    mesh: trimesh.Trimesh
) -> OrientedBoundingBox:
    """Compute the minimum-volume oriented bounding box (OBB) for a mesh.

    Calculates the canonical box extents, the rigid transformation matrix mapping
    from mesh coordinates to canonical origin-centered box coordinates, and the 8
    corner vertices in the original mesh coordinate frame.

    Args:
        mesh: Input trimesh.Trimesh instance.

    Returns:
        OrientedBoundingBox containing canonical corners, extents, and origin transform.
    """
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    to_origin = to_origin.astype(np.float32)
    extents = extents.astype(np.float32)

    corners_canonical = extents_to_corners(extents)
    mesh_from_origin = invert_transform(to_origin)
    corners_in_mesh = transform_points(corners_canonical, mesh_from_origin)

    return OrientedBoundingBox(corners_in_mesh, extents, to_origin)


def prepare_mesh_buffers(
    mesh: trimesh.Trimesh,
    max_tex_size: int | None = None,
    flip_uv: bool = True,
) -> MeshBuffers:
    """Extract contiguous vertex, normal, UV, and texture buffer arrays from a CAD mesh.

    Args:
        mesh: Input trimesh.Trimesh instance.
        max_tex_size: Optional maximum texture dimension limit in pixels.
        flip_uv: Whether to invert vertical UV coordinates (1 - V) for OpenGL convention (default: True).

    Returns:
        MeshBuffers container holding formatted buffer arrays.
    """
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    texture: np.ndarray | None = None
    uvs: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None

    has_valid_texture = (
        isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        and hasattr(mesh.visual, "material")
        and hasattr(mesh.visual.material, "image")
        and mesh.visual.material.image is not None
        and not isinstance(mesh.visual.material.image, trimesh.exceptions.ExceptionWrapper)
        and getattr(mesh.visual, "uv", None) is not None
        and len(mesh.visual.uv) > 0
    )

    if has_valid_texture:
        raw_img = mesh.visual.material.image
        if hasattr(raw_img, "convert"):
            img = np.asarray(raw_img.convert("RGB"), dtype=np.float32)
        else:
            img = np.asarray(raw_img, dtype=np.float32)
            if img.ndim == 2:
                img = np.repeat(img[..., np.newaxis], 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., :3]

        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = float(max_tex_size) / float(max_size)
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)

        texture = np.ascontiguousarray(img[np.newaxis, ...] / 255.0, dtype=np.float32)

        raw_uv = np.array(mesh.visual.uv, dtype=np.float32).copy()
        if flip_uv:
            raw_uv[:, 1] = 1.0 - raw_uv[:, 1]
        uvs = np.ascontiguousarray(raw_uv, dtype=np.float32)
    else:
        raw_colors = getattr(mesh.visual, "vertex_colors", None)
        if raw_colors is None or len(raw_colors) == 0:
            logger.debug("Mesh lacks vertex colors; defaulting to neutral gray (128, 128, 128).")
            raw_colors = np.full((len(mesh.vertices), 3), 128, dtype=np.uint8)

        vertex_colors = np.ascontiguousarray(
            raw_colors[..., :3].astype(np.float32) / 255.0,
            dtype=np.float32,
        )

    return MeshBuffers(
        vertices=vertices,
        faces=faces,
        normals=normals,
        texture=texture,
        uvs=uvs,
        vertex_colors=vertex_colors,
    )


__all__ = [
    "MeshBuffers",
    "OrientedBoundingBox",
    "center_mesh",
    "compute_mesh_diameter",
    "compute_obb",
    "load_mesh",
    "prepare_mesh_buffers",
    "scale_mesh",
]

