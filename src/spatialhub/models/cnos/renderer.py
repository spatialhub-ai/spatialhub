from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import moderngl
import numpy as np
import trimesh

from spatialhub.structures import CameraExtrinsics, CameraIntrinsics, ImageBuffers
from spatialhub.utils.camera import (
    intrinsics_to_projection_matrix,
    sample_sphere_poses,
)
from spatialhub.utils.image import save_image
from spatialhub.utils.mesh import (
    center_mesh,
    load_mesh,
    prepare_mesh_buffers,
    scale_mesh,
)
from spatialhub.utils.moderngl import (
    AtlasRenderer,
    create_moderngl_context,
    validate_moderngl_context,
)
from spatialhub.utils.transform import invert_transform, opencv_to_opengl
from spatialhub.utils.validation import validate_array

logger = logging.getLogger(__name__)

TEMPLATE_VS = """
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
    out vec4 v_clip_pos;

    void main() {
        vec4 pos_cam = in_pose * vec4(in_position, 1.0);
        v_cam_pos = pos_cam.xyz;

        v_cam_normal = mat3(in_pose) * in_normal;
        v_color = in_color;
        v_uv = in_uv;

        vec4 pos_clip = in_clip_mat * vec4(in_position, 1.0);
        v_clip_pos = pos_clip;

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

TEMPLATE_FS = """
    #version 330
    in vec3 v_cam_pos;
    in vec3 v_cam_normal;
    in vec3 v_color;
    in vec2 v_uv;
    in vec4 v_clip_pos;

    layout(location = 0) out vec4 f_color;
    layout(location = 1) out float f_depth;
    layout(location = 2) out vec4 f_normals;

    uniform int has_tex;
    uniform sampler2D tex_sampler;
    uniform int unshaded;
    uniform int use_light;
    uniform vec3 light_dir;
    uniform vec3 light_color;
    uniform float light_intensity;
    uniform vec4 ambient_light;

    void main() {
        if (v_clip_pos.w <= 0.0 || abs(v_clip_pos.x) > v_clip_pos.w || abs(v_clip_pos.y) > v_clip_pos.w) {
            discard;
        }

        vec3 n = normalize(v_cam_normal);
        vec3 base_color = v_color;

        if (has_tex == 1) {
            base_color = texture(tex_sampler, v_uv).rgb;
        }

        vec3 final_color = base_color;
        if (unshaded == 0 && use_light == 1) {
            vec3 l_dir = normalize(-light_dir);
            float diff = clamp(dot(n, l_dir), 0.0, 1.0);
            vec3 ambient = ambient_light.rgb * base_color;
            vec3 diffuse = diff * light_color * light_intensity * base_color;
            final_color = ambient + diffuse;
        }

        f_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
        f_depth = v_cam_pos.z;
        f_normals = vec4(n, 1.0);
    }
"""


def parse_poses(
    poses: str | Path | np.ndarray | CameraExtrinsics | list | tuple,
    pose_unit: str | float = "m",
) -> np.ndarray:
    """Load, validate, and standardize 4x4 camera-to-world viewpoint poses.

    Viewpoint poses represent rigid transformations from camera coordinates into
    world/object coordinates:
        X_world = R @ X_cam + t

    Args:
        poses: Camera-to-world (C2W) poses as CameraExtrinsics, (N, 4, 4) or (4, 4) array,
            sequence of pose matrices, or path to a .npy file.
        pose_unit: Unit or scale factor of translation coordinates ('m', 'cm', 'mm', or float multiplier).

    Returns:
        Metric camera-to-world pose array of shape (N, 4, 4) in float32.

    Raises:
        TypeError: If poses is not a valid CameraExtrinsics, NumPy array, list/tuple, or file path.
        ValueError: If poses array does not have shape (N, 4, 4) or (4, 4).
    """
    if isinstance(poses, CameraExtrinsics):
        view_poses = poses.matrix.astype(np.float32).copy()
    elif isinstance(poses, (str, Path)):
        view_poses = np.load(str(poses)).astype(np.float32)
    elif isinstance(poses, np.ndarray):
        view_poses = poses.astype(np.float32).copy()
    elif isinstance(poses, (list, tuple)):
        view_poses = np.asarray(poses, dtype=np.float32)
    else:
        raise TypeError("poses must be a CameraExtrinsics instance, numpy array, list/tuple, or path to a .npy file.")

    validate_array("poses", view_poses, ndim=(2, 3), shape_suffix=(4, 4), dtype=np.floating)
    if view_poses.ndim == 2:
        view_poses = view_poses[np.newaxis, ...]

    if isinstance(pose_unit, (int, float)):
        pose_scale = float(pose_unit)
    elif isinstance(pose_unit, str):
        unit_map = {"m": 1.0, "cm": 0.01, "mm": 0.001}
        pose_scale = unit_map.get(pose_unit.lower().strip(), 1.0)
    else:
        pose_scale = 1.0

    if pose_scale != 1.0:
        view_poses[:, :3, 3] *= pose_scale

    return view_poses


class TemplateRenderer:
    """Render multi-view RGB, depth, normal, and mask templates from a 3D CAD mesh using ModernGL."""

    def __init__(
        self,
        model: str | Path | trimesh.Trimesh,
        model_unit: str | float = "m",
        center: bool = True,
        width: int = 640,
        height: int = 480,
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | None = None,
        znear: float = 0.001,
        zfar: float = 100.0,
        unshaded: bool = False,
        ambient_light: tuple[float, float, float, float] | None = (1.0, 1.0, 1.0, 1.0),
        light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity: float = 1.0,
        max_chunk_size: int = 128,
        ctx: moderngl.Context | None = None,
    ) -> None:
        """Initialize the template renderer, CAD geometry buffers, and shader programs.

        Args:
            model: Path to the CAD mesh file (.ply, .obj, .stl) or pre-loaded trimesh.Trimesh.
            model_unit: Unit or scale factor of raw CAD mesh vertices ('m', 'cm', 'mm', or float).
            center: Whether to center the CAD mesh vertices at origin (default: True).
            width: Default template rendering width in pixels (default: 640).
            height: Default template rendering height in pixels (default: 480).
            intrinsics: Default camera intrinsics as CameraIntrinsics, 3x3 matrix, or 4-element sequence [fx, fy, cx, cy].
            znear: Near clipping plane distance in meters (default: 0.001).
            zfar: Far clipping plane distance in meters (default: 100.0).
            unshaded: Whether to disable directional shading and render flat base colors/textures.
            ambient_light: Ambient RGBA light intensity vector (default: (1.0, 1.0, 1.0, 1.0)).
            light_color: Directional/spot light RGB color vector (default: (1.0, 1.0, 1.0)).
            light_intensity: Light intensity multiplier (default: 1.0).
            max_chunk_size: Maximum number of viewpoints rendered per atlas sub-batch (default: 128).
            ctx: Optional pre-existing ModernGL context instance. If None, creates a standalone context.
        """
        if model is None:
            raise ValueError("model must be provided.")

        self.model = model
        self.model_unit = model_unit
        self.center = bool(center)
        self.width = int(width)
        self.height = int(height)
        self.intrinsics = intrinsics
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.unshaded = bool(unshaded)
        self.max_chunk_size = max(1, int(max_chunk_size))

        self.ambient_light = (
            np.array(ambient_light, dtype=np.float32)
            if ambient_light is not None
            else np.ones(4, dtype=np.float32)
        )
        self.light_color = np.array(light_color, dtype=np.float32)
        self.light_intensity = float(light_intensity)

        if ctx is not None:
            self.ctx = validate_moderngl_context(ctx)
            self._own_ctx = False
        else:
            self.ctx = create_moderngl_context()
            self._own_ctx = True

        self.engine = AtlasRenderer(self.ctx)

        self.prog = self.ctx.program(vertex_shader=TEMPLATE_VS, fragment_shader=TEMPLATE_FS)
        # Attachment 0 = RGBA (4 x f4), Attachment 1 = Metric Depth (1 x f4), Attachment 2 = Normals (4 x f4)
        self.gbuffer_layout = [(4, "f4"), (1, "f4"), (4, "f4")]

        if isinstance(self.model, (str, Path)):
            mesh = load_mesh(self.model)
        elif isinstance(self.model, trimesh.Trimesh):
            mesh = self.model
        else:
            raise TypeError("model must be a file path or trimesh.Trimesh instance.")
        mesh = scale_mesh(mesh, model_unit=self.model_unit)
        if self.center:
            mesh, _ = center_mesh(mesh)
        self.mesh = mesh
        self._init_mesh_buffers()

    def _init_mesh_buffers(self) -> None:
        """Allocate static GPU vertex, normal, UV, texture, and index buffers."""
        mesh_buffers = prepare_mesh_buffers(self.mesh)

        self.pos_buffer = self.ctx.buffer(mesh_buffers.vertices.tobytes())
        self.norm_buffer = self.ctx.buffer(mesh_buffers.normals.tobytes())
        self.idx_buffer = self.ctx.buffer(mesh_buffers.faces.tobytes())

        self.has_tex = False
        self.tex_obj: moderngl.Texture | None = None

        if mesh_buffers.texture is not None:
            tex_data = mesh_buffers.texture
            if tex_data.ndim == 4:
                tex_data = tex_data[0]
            if tex_data.dtype != np.uint8:
                tex_data = np.clip(tex_data * 255.0, 0.0, 255.0).astype(np.uint8)

            self.uv_buffer = self.ctx.buffer(mesh_buffers.uvs.tobytes())
            self.color_buffer = self.ctx.buffer(np.ones_like(mesh_buffers.vertices, dtype=np.float32).tobytes())
            self.tex_obj = self.ctx.texture(
                (tex_data.shape[1], tex_data.shape[0]),
                3,
                np.ascontiguousarray(tex_data).tobytes(),
            )
            self.has_tex = True
        else:
            self.color_buffer = self.ctx.buffer(mesh_buffers.vertex_colors.tobytes())
            self.uv_buffer = self.ctx.buffer(np.zeros((len(mesh_buffers.vertices), 2), dtype=np.float32).tobytes())

    def render_templates(
        self,
        width: int | None = None,
        height: int | None = None,
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | None = None,
        poses: str | Path | np.ndarray | CameraExtrinsics | list | tuple | None = None,
        pose_unit: str | float = "m",
        radius: float | None = None,
        num_viewpoints: int = 42,
        unshaded: bool | None = None,
        passes: Sequence[str] = ("rgba", "depth", "masks"),
    ) -> ImageBuffers:
        """Render multi-view templates returning a standardized ImageBuffers instance.

        Args:
            width: Output image width in pixels (defaults to self.width).
            height: Output image height in pixels (defaults to self.height).
            intrinsics: Camera intrinsics as CameraIntrinsics, 3x3 matrix, or 4-element sequence [fx, fy, cx, cy].
            poses: Camera-to-world (C2W) viewpoint poses as CameraExtrinsics, (N, 4, 4) array, sequence, or path to .npy.
            pose_unit: Unit or scale factor for the translation component of loaded poses ('m', 'cm', 'mm').
            radius: Distance from model origin in meters. When poses is None, sets the sampling sphere radius
                (defaults to 1.0). When poses is provided, scales viewpoint translation vectors if specified.
            num_viewpoints: Number of viewpoints generated when poses is None (default: 42).
            unshaded: Optional override for unshaded flat rendering (defaults to self.unshaded).
            passes: Sequence of render passes to generate: 'rgba', 'depth', 'normals', 'masks'.

        Returns:
            ImageBuffers containing batched output arrays.
        """
        render_w = int(width) if width is not None else self.width
        render_h = int(height) if height is not None else self.height
        k_intrinsics = intrinsics if intrinsics is not None else self.intrinsics
        if isinstance(k_intrinsics, CameraIntrinsics):
            k_mat = k_intrinsics.matrix[0].astype(np.float32)
        elif k_intrinsics is None:
            fx = float(render_w)
            fy = float(render_w)
            cx = render_w / 2.0
            cy = render_h / 2.0
            k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        elif isinstance(k_intrinsics, (list, tuple)) or (isinstance(k_intrinsics, np.ndarray) and k_intrinsics.ndim == 1):
            if len(k_intrinsics) != 4:
                raise ValueError("Intrinsics must be a 3x3 camera matrix, CameraIntrinsics, or 4-element sequence [fx, fy, cx, cy].")
            fx, fy, cx, cy = [float(v) for v in k_intrinsics]
            k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        else:
            k_mat = np.asarray(k_intrinsics, dtype=np.float32)
            if k_mat.ndim == 3 and k_mat.shape[1:] == (3, 3):
                k_mat = k_mat[0]
            if k_mat.shape != (3, 3):
                raise ValueError(
                    f"Intrinsics must be a 3x3 camera matrix or 4-element sequence [fx, fy, cx, cy], got shape {k_mat.shape}"
                )

        if poses is not None:
            view_poses = parse_poses(poses, pose_unit=pose_unit)
            if radius is not None:
                view_poses[:, :3, 3] *= radius
        else:
            view_poses = sample_sphere_poses(
                num_viewpoints=num_viewpoints,
                radius=1.0 if radius is None else float(radius),
            )

        N = len(view_poses)
        if N == 0:
            raise ValueError("No viewpoints to render.")

        # Convert camera-to-world poses to object-in-camera (OpenCV frame)
        ob_in_cams = invert_transform(view_poses)

        # Transform to right-up-backward frame and compute MVP clip matrices
        ob_in_glcams = opencv_to_opengl(ob_in_cams)
        proj_mat = intrinsics_to_projection_matrix(
            k_mat,
            height=render_h,
            width=render_w,
            znear=self.znear,
            zfar=self.zfar,
        )
        clip_matrices = proj_mat @ ob_in_glcams

        is_unshaded = self.unshaded if unshaded is None else bool(unshaded)
        if "unshaded" in self.prog:
            self.prog["unshaded"].value = 1 if is_unshaded else 0
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

        # Bound chunk size by gpu texture limits
        max_tex = 8192
        if hasattr(self.ctx, "info") and "GL_MAX_TEXTURE_SIZE" in self.ctx.info:
            max_tex = int(self.ctx.info["GL_MAX_TEXTURE_SIZE"])
        max_cols = max(1, max_tex // render_w)
        max_rows = max(1, max_tex // render_h)
        hw_limit = max(1, max_cols * max_rows)
        chunk_size = min(N, self.max_chunk_size, hw_limit)

        color_chunks: list[np.ndarray] = []
        depth_chunks: list[np.ndarray] = []
        normals_chunks: list[np.ndarray] = []

        passes_set = {p.lower().strip() for p in passes}
        need_normals = "normals" in passes_set

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(N, chunk_start + chunk_size)
            chunk_N = chunk_end - chunk_start

            C = int(np.ceil(np.sqrt(chunk_N)))
            R = int(np.ceil(chunk_N / C))

            if "C" in self.prog:
                self.prog["C"].value = float(C)
            if "R" in self.prog:
                self.prog["R"].value = float(R)

            chunk_ob = ob_in_cams[chunk_start:chunk_end]
            chunk_clip = clip_matrices[chunk_start:chunk_end]

            pose_bytes = np.ascontiguousarray(chunk_ob.transpose(0, 2, 1), dtype=np.float32).tobytes()
            clip_bytes = np.ascontiguousarray(chunk_clip.transpose(0, 2, 1), dtype=np.float32).tobytes()

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

            self.engine.render(
                vao,
                num_instances=chunk_N,
                grid_shape=(R, C),
                tile_shape=(render_h, render_w),
                layout=self.gbuffer_layout,
            )

            color_chunks.append(self.engine.read_tiles(0))
            depth_chunks.append(self.engine.read_tiles(1))
            if need_normals:
                normals_chunks.append(self.engine.read_tiles(2))

            vao.release()
            pose_buffer.release()
            clip_buffer.release()

        # Concatenate unpacked chunks across the batch dimension
        color_full = np.concatenate(color_chunks, axis=0)
        depth_full = np.concatenate(depth_chunks, axis=0)

        # Convert OpenGL bottom-up to OpenCV top-down coordinates
        color_flipped = np.flip(color_full, axis=1)
        depth_flipped = np.flip(depth_full[..., 0], axis=1)

        alpha = color_flipped[..., 3]
        foreground_mask = alpha > 0.0

        out_rgba: np.ndarray | None = None
        if "rgba" in passes_set:
            out_rgba = np.clip(color_flipped * 255.0, 0.0, 255.0).astype(np.uint8)

        out_depth: np.ndarray | None = None
        if "depth" in passes_set:
            out_depth = np.where(foreground_mask, depth_flipped, 0.0).astype(np.float32)

        out_masks: np.ndarray | None = None
        if "masks" in passes_set:
            out_masks = foreground_mask

        out_normals: np.ndarray | None = None
        if need_normals:
            normals_full = np.concatenate(normals_chunks, axis=0)
            normals_flipped = np.flip(normals_full[..., :3], axis=1)
            out_normals = np.where(foreground_mask[..., np.newaxis], normals_flipped, 0.0).astype(np.float32)

        return ImageBuffers(
            rgba=out_rgba,
            depth=out_depth,
            normal=out_normals,
            mask=out_masks,
        )

    @staticmethod
    def save_templates(
        buffers: ImageBuffers,
        output_dir: str | Path,
        passes: Sequence[str] = ("rgba", "depth", "masks"),
        prefix: str = "template",
    ) -> dict[str, list[Path]]:
        """Save rendered multi-view template buffers to disk.

        Iterates over the requested render passes present in the ImageBuffers instance
        and writes each viewpoint frame to disk in the target output directory.

        Args:
            buffers: ImageBuffers container returned by render_templates.
            output_dir: Destination folder path for saved image files.
            passes: Sequence of render passes to save ('rgba', 'depth', 'normals', 'masks').
            prefix: Filename prefix for the output image files (default: 'template').

        Returns:
            Dictionary mapping pass names to lists of saved file paths.

        Raises:
            ValueError: If an explicitly requested pass is absent from buffers.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        num_views = len(buffers)
        if num_views == 0:
            return {}

        saved_files: dict[str, list[Path]] = {}
        passes_set = {p.lower().strip() for p in passes}

        pass_mapping = {
            "rgba": ("rgba", "RGBA", "png"),
            "depth": ("depth", None, "npy"),
            "normals": ("normal", None, "npy"),
            "normal": ("normal", None, "npy"),
            "masks": ("mask", "GRAY", "png"),
            "mask": ("mask", "GRAY", "png"),
        }

        for pass_name in passes:
            pass_key = pass_name.lower().strip()
            if pass_key not in pass_mapping:
                continue

            attr_name, color_mode, ext = pass_mapping[pass_key]
            buf = getattr(buffers, attr_name, None)
            if buf is None:
                raise ValueError(
                    f"Requested pass '{pass_name}' is not present in the provided ImageBuffers."
                )

            pass_files: list[Path] = []
            pass_dir = out_path / pass_key
            pass_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(num_views):
                frame = buf[idx]
                file_path = pass_dir / f"{prefix}_{idx:04d}.{ext}"

                if ext == "npy":
                    np.save(str(file_path), frame)
                else:
                    save_image(frame, file_path, color_mode=color_mode)

                pass_files.append(file_path)

            saved_files[pass_key] = pass_files

        return saved_files

    def __enter__(self) -> TemplateRenderer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release allocated GPU resources and context."""
        self.release()

    def release(self) -> None:
        """Release allocated vertex buffers, textures, program, and framebuffer objects."""
        if hasattr(self, "prog") and self.prog is not None:
            self.prog.release()
            self.prog = None
        if hasattr(self, "pos_buffer") and self.pos_buffer is not None:
            self.pos_buffer.release()
            self.pos_buffer = None
        if hasattr(self, "norm_buffer") and self.norm_buffer is not None:
            self.norm_buffer.release()
            self.norm_buffer = None
        if hasattr(self, "idx_buffer") and self.idx_buffer is not None:
            self.idx_buffer.release()
            self.idx_buffer = None
        if hasattr(self, "color_buffer") and self.color_buffer is not None:
            self.color_buffer.release()
            self.color_buffer = None
        if hasattr(self, "uv_buffer") and self.uv_buffer is not None:
            self.uv_buffer.release()
            self.uv_buffer = None
        if hasattr(self, "tex_obj") and self.tex_obj is not None:
            self.tex_obj.release()
            self.tex_obj = None
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.release_fbo()
        if hasattr(self, "ctx") and self.ctx is not None and getattr(self, "_own_ctx", True):
            self.ctx.release()
        self.ctx = None
