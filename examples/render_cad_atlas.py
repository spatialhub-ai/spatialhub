"""Batched CAD mesh multi-view atlas rendering example.

Loads a 3D textured CAD model (.ply), samples camera viewpoints, executes an
instanced rasterization pass across an offscreen atlas grid using
AtlasRenderer, un-tiles the multi-target color and normal attachments,
and exports an image montage and an animated visualization.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import moderngl
import numpy as np

import trimesh
from spatialhub.utils.camera import (
    intrinsics_to_projection_matrix,
    sample_sphere_poses,
)
from spatialhub.utils.image import save_image
from spatialhub.utils.mesh import (
    MeshBuffers,
    center_mesh,
    compute_mesh_diameter,
    load_mesh,
    prepare_mesh_buffers,
    scale_mesh,
)
from spatialhub.utils.moderngl import AtlasRenderer, create_moderngl_context
from spatialhub.utils.transform import invert_transform

logger = logging.getLogger(__name__)


# --8<-- [start:load_mesh_func]
def load_centered_mesh(mesh_path: str | Path, unit: str = "mm") -> trimesh.Trimesh:
    """Load a 3D CAD mesh, convert units to meters, and center at coordinate origin.

    Args:
        mesh_path: Path to the 3D CAD mesh file.
        unit: Source unit of mesh geometry (default: 'mm').

    Returns:
        Centered mesh scaled to meters.
    """
    mesh = load_mesh(mesh_path)
    mesh = scale_mesh(mesh, unit)
    mesh, _ = center_mesh(mesh)
    return mesh
# --8<-- [end:load_mesh_func]


# --8<-- [start:upload_texture_func]
def upload_mesh_texture(ctx: moderngl.Context, texture_data: np.ndarray | None) -> moderngl.Texture:
    """Upload mesh diffuse texture map to GPU VRAM or create fallback neutral texture.

    Args:
        ctx: Headless OpenGL context.
        texture_data: Optional float32 texture array of shape (1, H, W, 3).

    Returns:
        ModernGL 2D texture instance.
    """
    if texture_data is not None:
        tex = texture_data[0]
        tex_h, tex_w = tex.shape[:2]
        gl_texture = ctx.texture((tex_w, tex_h), 3, data=tex.tobytes(), dtype="f4")
        gl_texture.build_mipmaps()
        return gl_texture
    return ctx.texture((1, 1), 3, data=np.array([[[1.0, 1.0, 1.0]]], dtype="f4").tobytes(), dtype="f4")
# --8<-- [end:upload_texture_func]


def build_preview_montage(views: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Arrange an array of un-tiled view crops into a single 2D montage image.

    Args:
        views: Array of shape (N, H, W, C) in float32 normalized to [0, 1].
        grid_shape: Number of (rows, cols) in the montage layout.

    Returns:
        Montage array of shape (rows * H, cols * W, 3) in uint8 [0, 255].
    """
    num_views, tile_h, tile_w, _ = views.shape
    rows, cols = grid_shape

    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx in range(num_views):
        r = idx // cols
        c = idx % cols
        tile_rgb = (np.clip(views[idx, :, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        montage[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile_rgb

    return montage


def export_animated_gif(views: np.ndarray, output_path: str | Path, duration_ms: int = 100) -> None:
    """Export an array of viewpoint crops to an animated GIF using OpenCV.

    Args:
        views: Array of shape (N, H, W, C) in float32 normalized to [0, 1].
        output_path: Destination filesystem path for the GIF file.
        duration_ms: Frame display duration in milliseconds (default: 100ms).
    """
    if len(views) == 0:
        return

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    frames_bgr = [
        cv2.cvtColor((np.clip(views[idx, :, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for idx in range(len(views))
    ]

    if hasattr(cv2, "imwriteanimation") and hasattr(cv2, "Animation"):
        anim = cv2.Animation()
        anim.frames = frames_bgr
        anim.durations = [duration_ms] * len(frames_bgr)
        anim.loop_count = 0
        success = cv2.imwriteanimation(str(out_file), anim)
        if not success:
            raise IOError(f"Failed to write animated GIF to {out_file}")
    else:
        from PIL import Image

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        pil_frames[0].save(
            str(out_file),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )


# --8<-- [start:instanced_pipeline]
def create_instanced_mesh_pipeline(
    ctx: moderngl.Context,
    buffers: MeshBuffers,
    mvp_matrices: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[moderngl.Program, moderngl.VertexArray, list[moderngl.Buffer]]:
    """Build instanced shader program, pack geometry buffers, and bind VAO.

    Args:
        ctx: Headless OpenGL context.
        buffers: Vertex, normal, face, and texture coordinate buffers.
        mvp_matrices: Instanced column-major Model-View-Projection matrices of shape (N, 16).
        grid_shape: (rows, cols) dimensions of the 2D atlas grid.

    Returns:
        Tuple containing compiled shader program, configured VAO, and allocated VBOs.
    """
    vertex_shader = """
    #version 330
    in vec3 in_pos;
    in vec3 in_normal;
    in vec2 in_uv;
    in mat4 in_mvp;

    uniform int C;
    uniform int R;

    out vec3 v_normal;
    out vec2 v_uv;

    void main() {
        int col = gl_InstanceID % C;
        int row = gl_InstanceID / C;

        vec4 clip_pos = in_mvp * vec4(in_pos, 1.0);
        vec3 ndc = clip_pos.xyz / clip_pos.w;

        vec2 cell_scale = vec2(1.0 / float(C), 1.0 / float(R));
        vec2 cell_offset = vec2(
            -1.0 + (float(col) + 0.5) * 2.0 * cell_scale.x,
            -1.0 + (float(row) + 0.5) * 2.0 * cell_scale.y
        );

        gl_Position = vec4(ndc.xy * cell_scale + cell_offset, ndc.z, 1.0);
        v_normal = in_normal;
        v_uv = in_uv;
    }
    """

    fragment_shader = """
    #version 330
    uniform sampler2D u_texture;
    in vec3 v_normal;
    in vec2 v_uv;

    layout(location = 0) out vec4 fragColor;
    layout(location = 1) out vec4 fragNormal;

    void main() {
        vec3 base_color = texture(u_texture, v_uv).rgb;
        vec3 light_dir = normalize(vec3(0.5, 1.0, 0.8));
        float diff = max(dot(normalize(v_normal), light_dir), 0.35);

        fragColor = vec4(base_color * diff, 1.0);
        fragNormal = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
    }
    """

    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    program["C"].value = grid_shape[1]
    program["R"].value = grid_shape[0]
    program["u_texture"].value = 0

    uv_data = buffers.uvs if buffers.uvs is not None else np.zeros((len(buffers.vertices), 2), dtype="f4")
    vertex_data = np.hstack([buffers.vertices, buffers.normals, uv_data]).astype("f4")

    vbo_geometry = ctx.buffer(vertex_data.tobytes())
    ibo_faces = ctx.buffer(buffers.faces.astype("i4").tobytes())
    vbo_mvp = ctx.buffer(mvp_matrices.tobytes())

    vao = ctx.vertex_array(
        program,
        [
            (vbo_geometry, "3f 3f 2f", "in_pos", "in_normal", "in_uv"),
            (vbo_mvp, "16f/i", "in_mvp"),
        ],
        index_buffer=ibo_faces,
    )

    return program, vao, [vbo_geometry, ibo_faces, vbo_mvp]
# --8<-- [end:instanced_pipeline]


# --8<-- [start:render_multiview_func]
def render_multiview_atlas(
    mesh_path: str | Path,
    num_views: int = 16,
    grid_shape: tuple[int, int] = (4, 4),
    tile_shape: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Render a CAD mesh from spherical camera viewpoints into multi-view arrays.

    Args:
        mesh_path: Path to the 3D CAD mesh asset (.ply, .obj, etc.).
        num_views: Number of viewpoints to sample on the sphere.
        grid_shape: (rows, cols) dimensions of the 2D atlas grid.
        tile_shape: (height, width) dimensions of each individual tile in pixels.

    Returns:
        Tuple of (color_views, normal_views), each an array of shape (num_views, H, W, 4).
    """
    # 1. Mesh preparation (scaled to meters, centered at origin)
    mesh = load_centered_mesh(mesh_path, unit="mm")
    buffers = prepare_mesh_buffers(mesh)

    # 2. Viewpoint sampling (Fibonacci sphere around object)
    diameter = compute_mesh_diameter(mesh)
    radius = diameter * 1.8
    camera_poses = sample_sphere_poses(num_views, radius=radius)

    # Convert OpenCV camera poses (+X right, +Y down, +Z forward) to OpenGL view matrices
    gl_from_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype("f4")
    view_matrices = np.array([gl_from_cv @ invert_transform(p) for p in camera_poses], dtype="f4")

    # Perspective projection matrix from pinhole camera intrinsics
    tile_w, tile_h = tile_shape
    focal_length = float(tile_w) / (2.0 * np.tan(np.radians(45.0) / 2.0))
    intrinsic_k = np.array([
        [focal_length, 0.0, tile_w / 2.0],
        [0.0, focal_length, tile_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype="f4")

    projection = intrinsics_to_projection_matrix(
        intrinsic_k,
        height=tile_h,
        width=tile_w,
        znear=0.01,
        zfar=2.0,
        window_coords="y_up",
    )
    mvp_matrices = np.array([(projection @ v).T for v in view_matrices], dtype="f4")

    # 3. Headless OpenGL context and instanced pipeline setup
    ctx = create_moderngl_context()
    gl_texture = upload_mesh_texture(ctx, buffers.texture)
    program, vao, gl_buffers = create_instanced_mesh_pipeline(ctx, buffers, mvp_matrices, grid_shape)

    # 4. Offscreen atlas rasterization and framebuffer readback
    # --8<-- [start:atlas_renderer_usage]
    with AtlasRenderer(ctx=ctx) as renderer:
        gl_texture.use(0)

        renderer.render(
            vao=vao,
            num_instances=num_views,
            grid_shape=grid_shape,
            tile_shape=tile_shape,
            layout=[(4, "f4"), (4, "f4")],
        )

        color_views = renderer.read_tiles(attachment_index=0)
        normal_views = renderer.read_tiles(attachment_index=1)
    # --8<-- [end:atlas_renderer_usage]

    # 5. Clean up allocated GPU resources
    vao.release()
    for buf in gl_buffers:
        buf.release()
    gl_texture.release()
    program.release()
    ctx.release()

    return color_views, normal_views
# --8<-- [end:render_multiview_func]


def main(
    mesh_path: str = "examples/assets/hope_000002.ply",
    num_views: int = 16,
    grid_rows: int = 4,
    grid_cols: int = 4,
    tile_size: int = 256,
    output_dir: str = "examples/output",
    update_docs: bool = False,
) -> None:
    target_dir = Path("docs/assets") if update_docs else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    grid_shape = (grid_rows, grid_cols)
    tile_shape = (tile_size, tile_size)

    color_views, normal_views = render_multiview_atlas(
        mesh_path=mesh_path,
        num_views=num_views,
        grid_shape=grid_shape,
        tile_shape=tile_shape,
    )

    print(f"Loaded CAD model:     {mesh_path}")
    print(f"Rendered views:       {num_views}")
    print(f"Color tensor shape:   {color_views.shape} ({color_views.dtype})")
    print(f"Normal tensor shape:  {normal_views.shape} ({normal_views.dtype})")

    # Build 2D preview montages for disk visualization
    color_montage = build_preview_montage(color_views, grid_shape=grid_shape)
    normal_montage = build_preview_montage(normal_views, grid_shape=grid_shape)

    # Export individual filtered arrays and animated visualization
    color_png_path = target_dir / "cad_atlas_color_views.png"
    normal_png_path = target_dir / "cad_atlas_normal_views.png"
    gif_path = target_dir / "cad_multiview.gif"

    save_image(color_montage, color_png_path, color_mode="RGB")
    save_image(normal_montage, normal_png_path, color_mode="RGB")
    export_animated_gif(color_views, gif_path, duration_ms=100)

    print(f"Saved color montage:  {color_png_path}")
    print(f"Saved normal montage: {normal_png_path}")
    print(f"Saved animated GIF:   {gif_path}")
    print("Resource cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched CAD Mesh Atlas Renderer")
    parser.add_argument(
        "--mesh-path",
        type=str,
        default="examples/assets/hope_000002.ply",
        help="Path to 3D CAD mesh file (default: examples/assets/hope_000002.ply)",
    )
    parser.add_argument("--num-views", type=int, default=16, help="Number of viewpoints to render (default: 16)")
    parser.add_argument("--grid-rows", type=int, default=4, help="Atlas grid rows (default: 4)")
    parser.add_argument("--grid-cols", type=int, default=4, help="Atlas grid columns (default: 4)")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile resolution in pixels (default: 256)")
    parser.add_argument("--output-dir", type=str, default="examples/output", help="Directory for saved artifacts")
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    main(
        mesh_path=args.mesh_path,
        num_views=args.num_views,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        tile_size=args.tile_size,
        output_dir=args.output_dir,
        update_docs=args.update_docs,
    )
