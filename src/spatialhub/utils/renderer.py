"""
Utilities for rendering 3D meshes with Trimesh and Pyrender.
"""

from __future__ import annotations

import logging
import ctypes
import os
import cv2
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List

import numpy as np

# Configure headless EGL platform for Linux before OpenGL is loaded
if sys.platform != "win32":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# -------------------------------------------------------------------------
# PyOpenGL ctypes CArgObject Patch for Python 3.10+ / Windows
# -------------------------------------------------------------------------
try:
    import OpenGL

    OpenGL.ERROR_CHECKING = False
    from OpenGL.arrays import arraydatatype

    CArgObject = type(ctypes.byref(ctypes.c_int(0)))

    # Patch ArrayDatatype.from_param to pass pre-converted CArgObjects through
    _orig_from_param = arraydatatype.ArrayDatatype.from_param

    @classmethod
    def _patched_from_param(cls, value, typeCast=None):
        if isinstance(value, CArgObject):
            return value
        return _orig_from_param(value, typeCast)

    arraydatatype.ArrayDatatype.from_param = _patched_from_param
except Exception:
    pass

logger = logging.getLogger(__name__)

# Delayed/lazy imports for trimesh and pyrender to avoid bloating base library imports
_trimesh_installed = True
try:
    import trimesh
except ImportError:
    _trimesh_installed = False

_pyrender_installed = True
try:
    import pyrender
except ImportError:
    _pyrender_installed = False

def _check_dependencies():
    """Checks that the optional rendering dependencies are installed."""
    if not _trimesh_installed or not _pyrender_installed:
        raise ImportError(
            "Rendering functionality requires optional rendering dependencies. "
            "Install them with: uv sync --extra render (or pip install 'spatialhub[render]')"
        )

def to_single_mesh(scene_or_mesh: Any) -> trimesh.Trimesh:
    """Convert a Trimesh scene or mesh into a single mesh."""

    _check_dependencies()

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

def look_at(cam_location: np.ndarray, target_point: np.ndarray) -> np.ndarray:
    """Calculates camera-to-world 4x4 transform pointing camera at target_point from cam_location."""
    forward = target_point - cam_location
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / forward_norm

    tmp = np.array([0.0, 0.0, -1.0])
    # Check if camera location is nearly parallel to standard up/down direction
    norm = min(
        np.linalg.norm(cam_location - tmp),
        np.linalg.norm(cam_location + tmp),
    )
    if norm < 1e-3:
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm

    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = up / up_norm

    mat = np.stack((right, up, forward, cam_location), axis=-1)
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def inverse_transform(trans: np.ndarray) -> np.ndarray:
    """Computes the inverse of a 4x4 rigid homogeneous transformation."""
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot_inv = np.transpose(rot)
    t_inv = -np.matmul(rot_inv, t)
    
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1.0
    output[:3, :3] = rot_inv
    output[:3, 3] = t_inv
    return output

def generate_fibonacci_sphere_poses(num_viewpoints: int = 42, radius: float = 0.4, pose_type: str = "object_pose",) -> np.ndarray:
    """
    Generate evenly distributed camera or object poses around a sphere.

    Args:
        num_viewpoints: Number of poses to generate.
        radius: Distance from each viewpoint to the origin, in meters.
        pose_type: If 'object_pose', returns world-to-camera transforms
            for an object at the origin. Otherwise, returns camera-to-world
            transforms for a camera positioned around an object at the origin.

    Returns:
        Array of shape (num_viewpoints, 4, 4) containing homogeneous
        transformation matrices.
    """
    if num_viewpoints <= 1:
        raise ValueError("num_viewpoints must be greater than 1.")

    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

    for i in range(0, num_viewpoints):
        y = 1.0 - (i / float(num_viewpoints - 1)) * 2.0  # y goes from 1 to -1
        r_at_y = np.sqrt(max(0.0, 1.0 - y * y))  # radius at y
        theta = phi * i  # Golden angle increment
        x = np.cos(theta) * r_at_y
        z = np.sin(theta) * r_at_y
        points.append([x, y, z])
        
    points = np.array(points) * radius
    
    poses = []
    for pt in points:
        c2w = look_at(pt, np.array([0.0, 0.0, 0.0]))
        if pose_type == "object_pose":
            w2c = inverse_transform(c2w)
            poses.append(w2c)
        else:
            poses.append(c2w)
            
    return np.array(poses, dtype=np.float32)

# Convert OpenCV camera coordinates to OpenGL/Pyrender coordinates.
T_gl_cv = np.array([
                [1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0]
            ], dtype=np.float32)

class TemplateRenderer:
    """
    Render RGB and depth templates from a CAD model.
    """

    def __init__(
        self,
        model_path: str,
        model_unit: Union[str, float] = "m",
        ambient_light: Optional[Tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0),
        light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity: float = 1.0,
        bg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ):
        """
        Initialize the renderer and its lighting settings.

        Args:
            model_path:
                Path to the CAD model to render.
            model_unit:
                Unit or scale factor used by the CAD model. Supported string values are ``"m"``, ``"cm"``, and ``"mm"``.
                A numeric value is interpreted as a custom scale factor that is multiplied directly into the mesh coordinates.
        """

        self.model_path = str(model_path)
        self.model_unit = model_unit
        
        self.ambient_light = np.array(ambient_light, dtype=np.float32) if ambient_light is not None else None
        self.light_color = np.array(light_color, dtype=np.float32)
        self.light_intensity = light_intensity
        self.bg_color = np.array(bg_color, dtype=np.float32)

    def load_mesh(self, mesh_input: str | Path | trimesh.Trimesh, model_unit: str | float = "m",) -> trimesh.Trimesh:
        """
        Load a mesh, convert it to meters, and center it at the origin.
        """

        _check_dependencies()
        
        # Load mesh if string or Path
        if isinstance(mesh_input, (str, Path)):
            loaded = trimesh.load_mesh(str(mesh_input))
            mesh = to_single_mesh(loaded)
        else:
            mesh = to_single_mesh(mesh_input).copy()

        # Handle unit conversion scale factor
        if isinstance(model_unit, (int, float)):
            scale_factor = float(model_unit)
        elif isinstance(model_unit, str):
            unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
            scale_factor = unit_map.get(model_unit.lower().strip(), 1.0)
        else:
            raise TypeError("mesh_unit must be a string or a numeric scale factor.")

        if scale_factor != 1.0:
            mesh.apply_scale(scale_factor)
            logger.info(f"Scaled CAD mesh by factor {scale_factor} to convert to meters.")

        # Center the mesh bounding box at (0, 0, 0)
        bbox_center = mesh.bounding_box.centroid
        mesh.apply_translation(-bbox_center)
        logger.info(f"Centered mesh at origin (offset applied: {-bbox_center}).")

        return mesh

    def render_templates(self, 
                         width: int,
                         height: int,
                         intrinsics: np.ndarray | List[float],
                         poses: str | Path | np.ndarray = None,
                         pose_unit: str | float = "mm",
                         pose_type: str = None,
                         num_viewpoints: int = 42,
                         radius: float = 0.4,
                         ) -> List[Dict[str, np.ndarray]]:
        """
        Render RGB and depth templates of the CAD model from multiple viewpoints.

        The camera projection is defined by the provided intrinsic parameters. Viewpoints can either be generated on a sphere around the model or loaded
        from a set of 4x4 transformation matrices.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            intrinsics: Camera intrinsics, provided as either a 3x3 matrix
                ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` or as
                ``[fx, fy, cx, cy]``.
            poses: Optional viewpoint poses as a ``(N, 4, 4)`` NumPy array or path to a ``.npy`` file. If omitted, poses are generated using a Fibonacci sphere. \
                When a pose file is provided and ``pose_type`` is not specified, filenames containing ``"cam_pose"`` or ``"obj_pose"`` are used to infer the pose type.
            pose_unit: Unit or scale factor for the translation component of loaded poses. Supported units are ``"m"``, ``"cm"``, and ``"mm"``. A numeric value is used as a custom scale factor.
            pose_type: Specifies whether the poses represent object motion or camera motion. Supported values are ``"object_pose"`` and ``"camera_pose"``. If omitted, ``"object_pose"`` is used.
            num_viewpoints: Number of viewpoints to generate when ``poses`` is not provided.
            radius: Distance from the model origin to generated viewpoints, in meters. For loaded poses, the translation is additionally scaled by this value.

        Returns:
            A list of dictionaries, one per viewpoint, containing:
            ``rgba``:
                RGBA image as a ``(height, width, 4)`` ``uint8`` array.
            ``depth``:
                Depth map as a ``(height, width)`` ``float32`` array, with depth values in meters.
        """
        _check_dependencies()

        # Parse Intrinsics
        if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
        elif len(intrinsics) == 4:
            fx, fy, cx, cy = map(float, intrinsics)
        else:
            raise ValueError("Intrinsics must be a 3x3 camera matrix or a sequence of [fx, fy, cx, cy]")

        # Load model
        mesh = self.load_mesh(self.model_path, self.model_unit)

        # Get view attributes
        view_poses, pose_type = self._get_view_attribute(poses, pose_unit, pose_type, radius, num_viewpoints)

        # Create offscreen renderer
        render_engine = pyrender.OffscreenRenderer(width, height)

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=self.ambient_light)

        # Create camera
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000)

        # Create light
        light = pyrender.SpotLight(
                                    color=np.ones(3),
                                    intensity=self.light_intensity,
                                    innerConeAngle=np.pi / 16.0,
                                    outerConeAngle=np.pi / 6.0,
                                )

        # Create pyrender mesh from trimesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = scene.add(pyrender_mesh, pose=np.eye(4), name="mesh_node")
        
        if pose_type == "object_pose":
            # Camera and Light are fixed, mesh moves
            camera_node = scene.add(camera, pose=T_gl_cv, name="camera_node")
            light_node = scene.add(light, pose=T_gl_cv, name="light_node")
        else:
            # Mesh is fixed at origin, Camera and Light move
            camera_node = scene.add(camera, pose=np.eye(4), name="camera_node")
            light_node = scene.add(light, pose=np.eye(4), name="light_node")

        templates = []
        try:
            for pose in view_poses:

                if pose_type == "object_pose":
                    # Update mesh pose
                    scene.set_pose(mesh_node, pose)
                else:
                    # Update camera & light pose
                    cam_pose = pose @ T_gl_cv
                    scene.set_pose(camera_node, cam_pose)
                    scene.set_pose(light_node, cam_pose)
                    
                # Render frame
                color, depth = render_engine.render(scene, flags=pyrender.constants.RenderFlags.RGBA)

                templates.append({
                    "rgba": color.copy(),                       # [H, W, 4] uint8 (32-bit RGBA)
                    "depth": depth.astype(np.float32),          # [H, W] float32
                })
        finally:
            # Release Pyrender OpenGL resources to prevent memory leaks
            render_engine.delete()

        return templates

    def save(self, results: List[Dict[str, np.ndarray]], output_dir: Union[str, Path], save_scene=True, save_depth=False) -> Tuple[List[Path], List[Path]]:
        """
        Save rendered RGBA images and/or depth maps to disk.

        Args:
            results: Rendered outputs returned by ``render_templates()``.
            output_dir: Directory where the outputs are saved.
            save_scene: Whether to save the rendered RGBA images.
            save_depth: Whether to save the depth maps.

        Returns:
            Paths to the saved RGBA images and depth maps.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_rgb_paths: List[Path] = []
        saved_depth_paths: List[Path] = []

        for idx, res in enumerate(results):

            if save_scene:
                rgba_filename = output_path / f"{idx:06d}_rgba.png"

                cv2.imwrite(str(rgba_filename), cv2.cvtColor(res["rgba"], cv2.COLOR_RGBA2BGRA))

                saved_rgb_paths.append(rgba_filename)

            if save_depth:
                depth_filename = output_path / f"{idx:06d}_depth.npy"

                np.save(str(depth_filename), res["depth"])

                saved_depth_paths.append(depth_filename)


        logger.info(f"Saved {len(results)} rendered frames to: {output_dir}")
        return saved_rgb_paths, saved_depth_paths

    def _get_view_attribute(self, poses, pose_unit, pose_type, radius, num_viewpoints):
        """Load or generate view poses and resolve the pose type."""

        if poses is not None:
            if isinstance(poses, (str, Path)):
                path = Path(poses)
                view_poses = np.load(str(path)).astype(np.float32)
                if pose_type is None:
                    stem = path.stem.lower()
                    if "cam_pose" in stem:
                        pose_type = "camera_pose"
                    elif "obj_pose" in stem:
                        pose_type = "object_pose"
            elif isinstance(poses, np.ndarray):
                view_poses = poses.astype(np.float32)
            else:
                raise TypeError(
                    "poses must be a numpy array or path to a .npy file."
                )

            if view_poses.ndim != 3 or view_poses.shape[1:] != (4, 4):
                raise ValueError(
                    f"Expected poses shape (N, 4, 4), got {view_poses.shape}"
                )

            if isinstance(pose_unit, (int, float)):
                pose_scale = float(pose_unit)
            elif isinstance(pose_unit, str):
                unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
                pose_scale = unit_map.get(pose_unit.lower().strip(), 1.0)
            else:
                pose_scale = 1.0

            view_poses[:, :3, 3] *= pose_scale * radius
            logger.info(f"Scaled pose translations by factor {pose_scale} * {radius} to convert to meters.")

            pose_type = pose_type or "object_pose"
        else:
            # Generate view poses
            pose_type = pose_type or "object_pose"
            view_poses = generate_fibonacci_sphere_poses(num_viewpoints=num_viewpoints, radius=radius, pose_type=pose_type,)

        return view_poses, pose_type, 


