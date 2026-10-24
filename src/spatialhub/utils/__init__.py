import importlib
from typing import TYPE_CHECKING, Any

from .camera import (
    sample_sphere_points,
    intrinsics_to_projection_matrix,
    project_points_to_image,
    reproject_depth_to_3d,
    sample_icosphere_poses,
    sample_sphere_poses,
    scale_intrinsics,
)
from .colors import (
    to_color,
)
from .download import download_file
from .image import (
    load_image,
    mask_to_bbox,
    nms_boxes,
    normalize_image,
    pad_to_square,
    save_image,
    save_video,
    scale_depth,
)
from .transform import (
    extents_to_corners,
    invert_transform,
    opencv_to_opengl,
    transform_points,
)
from .visualization import (
    blend_masks,
    draw_3d_axis,
    draw_3d_box,
    draw_boxes_2d,
    visualize_dense_features,
    visualize_depth,
    visualize_global_features,
    visualize_matched_keypoints,
)

if TYPE_CHECKING:
    from .mesh import (
        MeshBuffers,
        OrientedBoundingBox,
        center_mesh,
        compute_mesh_diameter,
        compute_obb,
        load_mesh,
        prepare_mesh_buffers,
        scale_mesh,
    )
    from .moderngl import (
        AtlasRenderer,
        FullscreenRenderer,
        create_moderngl_context,
        validate_moderngl_context,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MeshBuffers": (".mesh", "MeshBuffers"),
    "OrientedBoundingBox": (".mesh", "OrientedBoundingBox"),
    "center_mesh": (".mesh", "center_mesh"),
    "compute_mesh_diameter": (".mesh", "compute_mesh_diameter"),
    "compute_obb": (".mesh", "compute_obb"),
    "load_mesh": (".mesh", "load_mesh"),
    "prepare_mesh_buffers": (".mesh", "prepare_mesh_buffers"),
    "scale_mesh": (".mesh", "scale_mesh"),
    "create_moderngl_context": (".moderngl.context", "create_moderngl_context"),
    "validate_moderngl_context": (".moderngl.context", "validate_moderngl_context"),
    "AtlasRenderer": (".moderngl.atlas_renderer", "AtlasRenderer"),
    "FullscreenRenderer": (".moderngl.fullscreen", "FullscreenRenderer"),
}

__all__ = [
    "to_color",
    "download_file",
    "load_image",
    "save_image",
    "save_video",
    "scale_depth",
    "mask_to_bbox",
    "pad_to_square",
    "normalize_image",
    "nms_boxes",
    "load_mesh",
    "scale_mesh",
    "center_mesh",
    "compute_mesh_diameter",
    "blend_masks",
    "draw_3d_axis",
    "draw_3d_box",
    "draw_boxes_2d",
    "visualize_dense_features",
    "visualize_depth",
    "visualize_global_features",
    "visualize_matched_keypoints",
    "extents_to_corners",
    "OrientedBoundingBox",
    "compute_obb",
    "sample_sphere_points",
    "invert_transform",
    "transform_points",
    "sample_sphere_poses",
    "sample_icosphere_poses",
    "MeshBuffers",
    "prepare_mesh_buffers",
    "scale_intrinsics",
    "reproject_depth_to_3d",
    "intrinsics_to_projection_matrix",
    "opencv_to_opengl",
    "project_points_to_image",
    "create_moderngl_context",
    "validate_moderngl_context",
    "AtlasRenderer",
    "FullscreenRenderer",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_path, package=__name__)
            val = getattr(module, attr_name)
            globals()[name] = val
            return val
        except ImportError as exc:
            err_msg = str(exc)
            if any(dep in err_msg for dep in ("trimesh", "moderngl", "scipy")):
                raise ImportError(
                    f"'{name}' requires optional rendering dependencies ('trimesh', 'moderngl'). "
                    f"Install with: uv sync --extra render (or pip install 'spatialhub[render]')"
                ) from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
