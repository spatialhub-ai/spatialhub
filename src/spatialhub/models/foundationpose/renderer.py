"""
Batched atlas renderer for template generation and G-Buffer extraction.
"""

from __future__ import annotations

import moderngl
import numpy as np

from spatialhub.utils import BatchedAtlasRenderer
from spatialhub.utils.camera import convert_opencv_to_opengl_pose, create_perspective_projection_matrix

from .helper import MeshArrays

ATLAS_VS = """
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    in vec2 in_uv;
    
    in mat4 in_pose;       // Extrinsic pose matrix
    in mat4 in_clip_mat;   // Combined projection + crop matrix
    
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

ATLAS_FS = """
    #version 330
    in vec3 v_cam_pos;
    in vec3 v_cam_normal;
    in vec3 v_color;
    in vec2 v_uv;
    in vec2 v_ndc;

    layout(location = 0) out vec4 f_color;
    layout(location = 1) out vec4 f_xyz;
    layout(location = 2) out vec4 f_normal;

    uniform int use_light;
    uniform int use_dir_light;
    uniform int has_tex;
    uniform vec3 light_dir;
    uniform vec3 light_pos;
    uniform vec3 light_color;
    uniform float w_ambient;
    uniform float w_diffuse;
    uniform sampler2D tex_sampler;

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
            vec3 l_dir;
            if (use_dir_light == 1) {
                l_dir = normalize(-light_dir);
            } else {
                l_dir = normalize(light_pos - v_cam_pos);
            }
            float diff = clamp(dot(n, l_dir), 0.0, 1.0);
            final_color = base_color * w_ambient + diff * light_color * w_diffuse;
        }
        
        f_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
        f_xyz = vec4(v_cam_pos, 1.0);
        f_normal = vec4(n, 1.0);
    }
"""


class Renderer:
    """
    Batched atlas renderer generating synthetic RGBA, XYZ, and normal maps for pose candidate batches.
    """

    def __init__(self, glctx: moderngl.Context, mesh_arrays: MeshArrays):
        """
        Initialize shader programs and static vertex buffer objects.

        Args:
            glctx: Execution context instance.
            mesh_arrays: MeshArrays container holding vertex positions, normals, faces, and textures.
        """
        self.glctx = glctx
        self.engine = BatchedAtlasRenderer(self.glctx)

        self.prog = self.glctx.program(vertex_shader=ATLAS_VS, fragment_shader=ATLAS_FS)

        # Multi-target G-Buffer layout
        # Attachment 0: RGBA color (4 x float32)
        # Attachment 1: Camera-space XYZ + depth (4 x float32)
        # Attachment 2: Surface normals (4 x float32)
        self.gbuffer_layout = [(4, "f4"), (4, "f4"), (4, "f4")]

        def to_bytes(arr: np.ndarray, dtype: type) -> bytes:
            return np.ascontiguousarray(arr, dtype=dtype).tobytes()

        mesh_dict = mesh_arrays.as_dict()

        self.has_tex = "tex" in mesh_dict and mesh_dict["tex"] is not None

        # Allocate static geometry buffers
        self.pos_buffer = self.glctx.buffer(to_bytes(mesh_dict["pos"], np.float32))
        self.norm_buffer = self.glctx.buffer(to_bytes(mesh_dict["vnormals"], np.float32))
        self.idx_buffer = self.glctx.buffer(to_bytes(mesh_dict["faces"], np.int32))

        self.tex_obj = None
        if self.has_tex:
            self.uv_buffer = self.glctx.buffer(to_bytes(mesh_dict["uv"], np.float32))
            self.color_buffer = self.glctx.buffer(to_bytes(np.ones_like(mesh_dict["pos"]), np.float32))

            tex_data = np.ascontiguousarray(mesh_dict["tex"], dtype=np.uint8)
            if tex_data.ndim == 4:
                tex_data = tex_data[0]

            self.tex_obj = self.glctx.texture(
                (tex_data.shape[1], tex_data.shape[0]), tex_data.shape[2], tex_data.tobytes()
            )
        else:
            self.color_buffer = self.glctx.buffer(to_bytes(mesh_dict["vertex_color"], np.float32))
            self.uv_buffer = self.glctx.buffer(to_bytes(np.zeros((mesh_dict["pos"].shape[0], 2)), np.float32))

    def render(
        self,
        K: np.ndarray,
        H: int,
        W: int,
        ob_in_cams: np.ndarray,
        output_size: tuple[int, int],
        bbox2d: np.ndarray,
        get_normal: bool = False,
        use_light: bool = False,
        light_color: np.ndarray | None = None,
        light_dir: np.ndarray = np.array([0, 0, 1]),
        light_pos: np.ndarray = np.array([0, 0, 0]),
        w_ambient: float = 0.8,
        w_diffuse: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Render a grid atlas of synthetic RGB, depth, normal, and XYZ maps across N pose hypotheses.

        Args:
            K: 3x3 camera intrinsic matrix.
            H: Full image height in pixels.
            W: Full image width in pixels.
            ob_in_cams: Candidate pose hypotheses array of shape (N, 4, 4).
            output_size: Output tile crop resolution as (width, height).
            bbox2d: 2D bounding boxes array of shape (N, 4) or None.
            get_normal: Whether to unpack surface normal maps (default: False).
            use_light: Whether to compute directional/point lighting (default: False).
            light_color: Light source RGB color vector.
            light_dir: Light direction vector.
            light_pos: Light source position coordinates.
            w_ambient: Ambient lighting coefficient weight.
            w_diffuse: Diffuse lighting coefficient weight.

        Returns:
            Tuple of (color, depth, normal_map, xyz_map) arrays for all N candidate viewpoints.
        """
        N = len(ob_in_cams)
        out_H, out_W = int(output_size[0]), int(output_size[1])
        C = int(np.ceil(np.sqrt(N)))
        R = int(np.ceil(N / C))

        # Set program uniforms
        self.prog["C"].value = C
        self.prog["R"].value = R
        self.prog["has_tex"].value = 1 if self.has_tex else 0
        self.prog["use_light"].value = 1 if use_light else 0

        if self.has_tex:
            self.tex_obj.use(0)
            self.prog["tex_sampler"].value = 0

        if use_light:
            self.prog["w_ambient"].value = float(w_ambient)
            self.prog["w_diffuse"].value = float(w_diffuse)
            self.prog["use_dir_light"].value = 1 if light_dir is not None else 0
            self.prog["light_dir"].value = tuple(light_dir) if light_dir is not None else (0.0, 0.0, 1.0)
            self.prog["light_pos"].value = tuple(light_pos) if light_pos is not None else (0.0, 0.0, 0.0)
            self.prog["light_color"].value = tuple(light_color) if light_color is not None else (1.0, 1.0, 1.0)

        # Transform camera poses to rendering coordinate convention
        ob_in_glcams = convert_opencv_to_opengl_pose(ob_in_cams)

        projection_mat = create_perspective_projection_matrix(K, height=H, width=W, znear=0.001, zfar=100)
        mtx = projection_mat @ ob_in_glcams

        if bbox2d is not None:
            l, t, r, b = bbox2d[:, 0], H - bbox2d[:, 1], bbox2d[:, 2], H - bbox2d[:, 3]
            tf = np.zeros((N, 4, 4), dtype=np.float32)
            tf[:, 0, 0], tf[:, 1, 1] = W / (r - l), H / (t - b)
            tf[:, 3, 0], tf[:, 3, 1] = (W - r - l) / (r - l), (H - t - b) / (t - b)
            tf[:, 2, 2] = 1.0
            tf[:, 3, 3] = 1.0
            final_mtx = tf.transpose(0, 2, 1) @ mtx
        else:
            final_mtx = mtx

        pose_bytes = np.ascontiguousarray(ob_in_cams.transpose(0, 2, 1), dtype=np.float32).tobytes()
        clip_bytes = np.ascontiguousarray(final_mtx.transpose(0, 2, 1), dtype=np.float32).tobytes()

        # Dynamic instance buffer allocation
        pose_buffer = self.glctx.buffer(pose_bytes, dynamic=True)
        clip_buffer = self.glctx.buffer(clip_bytes, dynamic=True)

        vao = self.glctx.vertex_array(
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

        # Render atlas grid into framebuffer
        self.engine.render(vao, N, C, R, out_H, out_W, layout=self.gbuffer_layout)

        # Unpack color and coordinate attachments
        color_full = self.engine.unpack_attachment(0)
        xyz_full = self.engine.unpack_attachment(1)

        color, alpha = color_full[..., :3], color_full[..., 3:]
        xyz_map, depth = xyz_full[..., :3], xyz_full[..., 2]

        color = np.flip(color * np.clip(alpha, 0.0, 1.0), axis=1)
        depth = np.flip(depth, axis=1)
        xyz_map = np.flip(xyz_map, axis=1)

        normal_map = None
        if get_normal:
            raw_normals = self.engine.unpack_attachment(2)[..., :3]
            norms = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            normal_map = raw_normals / norms
            normal_map = np.flip(normal_map, axis=1)

        # Release dynamic frame resources
        vao.release()
        pose_buffer.release()
        clip_buffer.release()

        return color, depth, normal_map, xyz_map

    def release(self) -> None:
        """Release allocated buffers and framebuffer objects."""
        self.prog.release()
        self.pos_buffer.release()
        self.norm_buffer.release()
        self.idx_buffer.release()
        self.color_buffer.release()
        self.uv_buffer.release()
        if self.tex_obj:
            self.tex_obj.release()
        self.engine.release_fbo()
