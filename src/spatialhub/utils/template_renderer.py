"""
Utilities for rendering 3D CAD mesh templates and scene projections using ModernGL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import moderngl
import numpy as np

from .camera import create_projection_matrix, opencv_to_opengl_pose
from .mesh import invert_transform, load_mesh, prepare_mesh_arrays, sample_sphere_poses
from .moderngl.atlas_renderer import BatchedAtlasRenderer
from .moderngl.context import create_moderngl_context

logger = logging.getLogger(__name__)

# Delayed/lazy imports for trimesh to avoid bloating base library imports
_trimesh_installed = True
try:
    import trimesh
except ImportError:
    _trimesh_installed = False


def _check_dependencies() -> None:
    """Checks that the optional rendering dependencies are installed."""
    if not _trimesh_installed:
        raise ImportError(
            "Rendering functionality requires optional rendering dependencies ('trimesh'). "
            "Install them with: uv sync --extra render (or pip install 'spatialhub[render]')"
        )


DEFAULT_TEMPLATE_VS = """
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    in vec2 in_uv;

    in mat4 in_pose;       // Extrinsic pose matrix (OpenCV camera frame)
    in mat4 in_clip_mat;   // Combined projection + OpenGL pose matrix (MVP)

    uniform float C;
    uniform float R;

    out vec3 v_cam_pos;
    out vec3 v_cam_normal;
    out vec3 v_color;
    out vec2 v_uv;
    out vec2 v_ndc;

    void main() {
        vec4 pos_cam = in_pose * vec4(in_position, 1.0);
        v_cam_pos = pos_cam.xyz;

        v_cam_normal = mat3(in_pose) * in_normal;
        v_color = in_color;
        v_uv = in_uv;

        vec4 pos_clip = in_clip_mat * vec4(in_position, 1.0);
        v_ndc = pos_clip.xy / pos_clip.w;

        float sx = 1.0 / C;
        float sy = 1.0 / R;
        int gx = gl_InstanceID % int(C);
        int gy = gl_InstanceID / int(C);

        float tx = -1.0 + (2.0 * float(gx) + 1.0) * sx;
        float ty = -1.0 + (2.0 * float(gy) + 1.0) * sy;

        pos_clip.x = pos_clip.x * sx + pos_clip.w * tx;
        pos_clip.y = pos_clip.y * sy + pos_clip.w * ty;

        gl_Position = pos_clip;
    }
"""

DEFAULT_TEMPLATE_FS = """
    #version 330
    in vec3 v_cam_pos;
    in vec3 v_cam_normal;
    in vec3 v_color;
    in vec2 v_uv;
    in vec2 v_ndc;

    layout(location = 0) out vec4 f_color;
    layout(location = 1) out vec4 f_depth;

    uniform int has_tex;
    uniform sampler2D tex_sampler;
    uniform int use_light;
    uniform vec3 light_dir;
    uniform vec3 light_color;
    uniform float light_intensity;
    uniform vec4 ambient_light;

    void main() {
        if (abs(v_ndc.x) > 1.0 || abs(v_ndc.y) > 1.0) {
            discard;
        }

        vec3 n = normalize(v_cam_normal);
        vec3 base_color = v_color;

        if (has_tex == 1) {
            base_color = texture(tex_sampler, v_uv).rgb;
        }

        vec3 final_color = base_color;
        if (use_light == 1) {
            vec3 l_dir = normalize(-light_dir);
            float diff = clamp(dot(n, l_dir), 0.0, 1.0);
            vec3 ambient = ambient_light.rgb * base_color;
            vec3 diffuse = diff * light_color * light_intensity * base_color;
            final_color = ambient + diffuse;
        }

        f_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
        f_depth = vec4(v_cam_pos.z, v_cam_pos.z, v_cam_pos.z, 1.0);
    }
"""


class TemplateRenderer:
    """
    Render RGB and depth templates from a CAD model using ModernGL batched atlas rendering.
    """

    def __init__(
        self,
        model_path: str | Path | trimesh.Trimesh,
        model_unit: str | float = "m",
        ambient_light: tuple[float, float, float, float] | None = (1.0, 1.0, 1.0, 1.0),
        light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity: float = 1.0,
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        ctx: moderngl.Context | None = None,
        vertex_shader: str | None = None,
        fragment_shader: str | None = None,
        gbuffer_layout: list[tuple[int, str]] | None = None,
    ):
        """
        Initialize the template renderer, CAD geometry buffers, and lighting parameters.

        Args:
            model_path: Path to the CAD model file (.ply, .obj, .stl) or pre-loaded trimesh.Trimesh.
            model_unit: Unit or scale factor of raw CAD mesh vertices ('m', 'cm', 'mm', or float).
            ambient_light: Ambient RGBA light intensity vector (default: (1.0, 1.0, 1.0, 1.0)).
            light_color: Directional/spot light RGB color vector (default: (1.0, 1.0, 1.0)).
            light_intensity: Light intensity multiplier (default: 1.0).
            bg_color: Framebuffer background clear RGBA color (default: (0.0, 0.0, 0.0, 0.0)).
            ctx: Optional pre-existing ModernGL context instance. If None, creates a standalone context.
            vertex_shader: Optional custom vertex shader GLSL source string.
            fragment_shader: Optional custom fragment shader GLSL source string.
            gbuffer_layout: Optional custom G-Buffer attachment layout specifications.
        """
        _check_dependencies()

        self.model_path = model_path
        self.model_unit = model_unit

        self.ambient_light = np.array(ambient_light, dtype=np.float32) if ambient_light is not None else np.ones(4, dtype=np.float32)
        self.light_color = np.array(light_color, dtype=np.float32)
        self.light_intensity = float(light_intensity)
        self.bg_color = np.array(bg_color, dtype=np.float32)

        self.ctx = ctx if ctx is not None else create_moderngl_context()
        self.engine = BatchedAtlasRenderer(self.ctx)

        self.vs_source = vertex_shader or DEFAULT_TEMPLATE_VS
        self.fs_source = fragment_shader or DEFAULT_TEMPLATE_FS
        self.prog = self.ctx.program(vertex_shader=self.vs_source, fragment_shader=self.fs_source)

        # Default layout: Attachment 0 = RGBA (4 x float32), Attachment 1 = Metric Depth (4 x float32)
        self.gbuffer_layout = gbuffer_layout or [(4, "f4"), (4, "f4")]

        # Load and prepare static mesh geometry
        self.mesh = load_mesh(self.model_path, self.model_unit, center=True)
        self._init_mesh_buffers()


    def _init_mesh_buffers(self) -> None:
        """Allocate static GPU vertex, normal, UV, texture, and index buffers."""
        mesh_arrays = prepare_mesh_arrays(self.mesh)

        self.pos_buffer = self.ctx.buffer(mesh_arrays.pos.tobytes())
        self.norm_buffer = self.ctx.buffer(mesh_arrays.vnormals.tobytes())
        self.idx_buffer = self.ctx.buffer(mesh_arrays.faces.tobytes())

        self.has_tex = False
        self.tex_obj: moderngl.Texture | None = None

        if mesh_arrays.tex is not None:
            tex_data = mesh_arrays.tex
            if tex_data.ndim == 4:
                tex_data = tex_data[0]
            if tex_data.dtype != np.uint8:
                tex_data = np.clip(tex_data * 255.0, 0.0, 255.0).astype(np.uint8)

            self.uv_buffer = self.ctx.buffer(mesh_arrays.uv.tobytes())
            self.color_buffer = self.ctx.buffer(np.ones_like(mesh_arrays.pos, dtype=np.float32).tobytes())
            self.tex_obj = self.ctx.texture(
                (tex_data.shape[1], tex_data.shape[0]),
                3,
                np.ascontiguousarray(tex_data).tobytes(),
            )
            self.has_tex = True
        else:
            self.color_buffer = self.ctx.buffer(mesh_arrays.vertex_color.tobytes())
            self.uv_buffer = self.ctx.buffer(np.zeros((len(mesh_arrays.pos), 2), dtype=np.float32).tobytes())


    def render_templates(
        self,
        width: int,
        height: int,
        intrinsics: np.ndarray | list[float],
        poses: str | Path | np.ndarray | None = None,
        pose_unit: str | float = "mm",
        pose_type: str | None = None,
        num_viewpoints: int = 42,
        radius: float = 0.4,
        znear: float = 0.001,
        zfar: float = 100.0,
    ) -> list[dict[str, np.ndarray]]:
        """
        Render RGB and depth templates of the CAD model from multiple viewpoints.

        The camera projection is defined by the provided intrinsic parameters. Viewpoints can either
        be generated on a sphere around the model or loaded from a set of 4x4 transformation matrices.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            intrinsics: Camera intrinsics as a 3x3 matrix or 4-element sequence [fx, fy, cx, cy].
            poses: Optional viewpoint poses as an (N, 4, 4) NumPy array or path to a .npy file.
                If omitted, poses are generated using a Fibonacci sphere.
            pose_unit: Unit or scale factor for the translation component of loaded poses ('m', 'cm', 'mm').
            pose_type: Specifies whether poses represent 'object_pose' (w2c) or 'camera_pose' (c2w).
            num_viewpoints: Number of viewpoints to generate when poses is omitted.
            radius: Distance from model origin to generated viewpoints, in meters.
            znear: Near clipping plane distance in meters.
            zfar: Far clipping plane distance in meters.

        Returns:
            A list of dictionaries, one per viewpoint, containing:
                "rgba": RGBA image as a (height, width, 4) uint8 array.
                "depth": Metric depth map as a (height, width) float32 array in meters.
        """
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

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Resolve viewpoint poses and pose type
        view_poses, resolved_pose_type = self._get_view_attribute(poses, pose_unit, pose_type, radius, num_viewpoints)

        N = len(view_poses)
        C = int(np.ceil(np.sqrt(N)))
        R = int(np.ceil(N / C))

        # Convert view poses to object-in-camera (OpenCV frame)
        if resolved_pose_type == "object_pose":
            ob_in_cams = view_poses
        else:
            ob_in_cams = invert_transform(view_poses)

        # Transform to OpenGL coordinate frame and compute MVP clip matrices
        ob_in_glcams = opencv_to_opengl_pose(ob_in_cams)
        proj_mat = create_projection_matrix(K, height=height, width=width, znear=znear, zfar=zfar)
        clip_matrices = proj_mat @ ob_in_glcams

        # Set program uniforms
        if "C" in self.prog:
            self.prog["C"].value = float(C)
        if "R" in self.prog:
            self.prog["R"].value = float(R)
        if "has_tex" in self.prog:
            self.prog["has_tex"].value = 1 if self.has_tex else 0
        if "use_light" in self.prog:
            self.prog["use_light"].value = 1
        if "light_dir" in self.prog:
            self.prog["light_dir"].value = (0.0, 0.0, -1.0)
        if "light_color" in self.prog:
            self.prog["light_color"].value = tuple(self.light_color)
        if "light_intensity" in self.prog:
            self.prog["light_intensity"].value = self.light_intensity
        if "ambient_light" in self.prog:
            self.prog["ambient_light"].value = tuple(self.ambient_light)

        if self.has_tex and self.tex_obj is not None:
            self.tex_obj.use(0)
            if "tex_sampler" in self.prog:
                self.prog["tex_sampler"].value = 0

        # Create dynamic instance buffers
        pose_bytes = np.ascontiguousarray(ob_in_cams.transpose(0, 2, 1), dtype=np.float32).tobytes()
        clip_bytes = np.ascontiguousarray(clip_matrices.transpose(0, 2, 1), dtype=np.float32).tobytes()

        pose_buffer = self.ctx.buffer(pose_bytes, dynamic=True)
        clip_buffer = self.ctx.buffer(clip_bytes, dynamic=True)

        vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.pos_buffer, "3f", "in_position"),
                (self.norm_buffer, "3f", "in_normal"),
                (self.color_buffer, "3f", "in_color"),
                (self.uv_buffer, "2f", "in_uv"),
                (pose_buffer, "16f/i", "in_pose"),
                (clip_buffer, "16f/i", "in_clip_mat"),
            ],
            self.idx_buffer,
        )

        # Execute batched atlas render pass
        self.engine.render(vao, N, C, R, height, width, layout=self.gbuffer_layout)

        # Unpack color and metric depth attachments
        color_full = self.engine.unpack_attachment(0)  # (N, H, W, 4) float32
        depth_full = self.engine.unpack_attachment(1)  # (N, H, W, 4) float32

        # Convert OpenGL bottom-up coordinates to OpenCV top-down format
        color = np.flip(color_full, axis=1)
        depth = np.flip(depth_full[..., 0], axis=1)

        # Extract RGBA uint8 and apply background mask to depth
        alpha = color[..., 3:]
        rgba_uint8 = np.clip(color * 255.0, 0.0, 255.0).astype(np.uint8)
        depth = np.where(alpha[..., 0] > 0.0, depth, 0.0).astype(np.float32)

        # Cleanup per-frame GPU resources
        vao.release()
        pose_buffer.release()
        clip_buffer.release()

        templates = [
            {
                "rgba": rgba_uint8[i],
                "depth": depth[i],
            }
            for i in range(N)
        ]

        return templates


    def save(
        self,
        results: list[dict[str, np.ndarray]],
        output_dir: str | Path,
        save_scene: bool = True,
        save_depth: bool = False,
    ) -> tuple[list[Path], list[Path]]:
        """
        Save rendered RGBA images and/or depth maps to disk.

        Args:
            results: Rendered outputs returned by render_templates().
            output_dir: Directory where the outputs are saved.
            save_scene: Whether to save the rendered RGBA images.
            save_depth: Whether to save the depth maps.

        Returns:
            Tuple of lists containing paths to saved RGBA images and depth maps.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_rgb_paths: list[Path] = []
        saved_depth_paths: list[Path] = []

        for idx, res in enumerate(results):
            if save_scene:
                rgba_filename = output_path / f"{idx:06d}_rgba.png"
                cv2.imwrite(str(rgba_filename), cv2.cvtColor(res["rgba"], cv2.COLOR_RGBA2BGRA))
                saved_rgb_paths.append(rgba_filename)

            if save_depth:
                depth_filename = output_path / f"{idx:06d}_depth.npy"
                np.save(str(depth_filename), res["depth"])
                saved_depth_paths.append(depth_filename)

        logger.info("Saved %d rendered frames to: %s", len(results), output_dir)
        return saved_rgb_paths, saved_depth_paths


    def _get_view_attribute(
        self,
        poses: str | Path | np.ndarray | None,
        pose_unit: str | float,
        pose_type: str | None,
        radius: float,
        num_viewpoints: int,
    ) -> tuple[np.ndarray, str]:
        """Load or generate viewpoint poses and resolve pose reference type."""
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
                raise TypeError("poses must be a numpy array or path to a .npy file.")

            if view_poses.ndim != 3 or view_poses.shape[1:] != (4, 4):
                raise ValueError(f"Expected poses shape (N, 4, 4), got {view_poses.shape}")

            if isinstance(pose_unit, (int, float)):
                pose_scale = float(pose_unit)
            elif isinstance(pose_unit, str):
                unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
                pose_scale = unit_map.get(pose_unit.lower().strip(), 1.0)
            else:
                pose_scale = 1.0

            view_poses[:, :3, 3] *= pose_scale * radius
            logger.info("Scaled pose translations by factor %s * %s to convert to meters.", pose_scale, radius)

            resolved_pose_type = pose_type or "object_pose"
        else:
            resolved_pose_type = pose_type or "object_pose"
            view_poses = sample_sphere_poses(
                num_viewpoints=num_viewpoints,
                radius=radius,
                sampling_method="fibonacci",
                pose_type=resolved_pose_type,
            )

        return view_poses, resolved_pose_type


    def release(self) -> None:
        """Release allocated vertex buffers, textures, program, and framebuffer objects."""
        if hasattr(self, "prog"):
            self.prog.release()
        if hasattr(self, "pos_buffer"):
            self.pos_buffer.release()
        if hasattr(self, "norm_buffer"):
            self.norm_buffer.release()
        if hasattr(self, "idx_buffer"):
            self.idx_buffer.release()
        if hasattr(self, "color_buffer"):
            self.color_buffer.release()
        if hasattr(self, "uv_buffer"):
            self.uv_buffer.release()
        if self.tex_obj is not None:
            self.tex_obj.release()
            self.tex_obj = None
        self.engine.release_fbo()
