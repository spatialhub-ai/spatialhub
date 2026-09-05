"""
Utilities for reading, scaling, centering, and analyzing 3D CAD meshes.

Provides modular functions (`read_mesh`, `scale_mesh`, `center_mesh`) alongside a
combined high-level composite pipeline (`load_mesh`) and auxiliary tools (`to_single_mesh`, `compute_mesh_diameter`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Delayed/lazy imports for trimesh to avoid bloating base library imports
_trimesh_installed = True
try:
    import trimesh
except ImportError:
    _trimesh_installed = False


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
        mesh = scene_or_mesh.dump(concatenate=True)
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


def compute_oriented_bounding_box(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


