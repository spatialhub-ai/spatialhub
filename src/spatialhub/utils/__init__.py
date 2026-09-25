import importlib
from typing import TYPE_CHECKING, Any

from .camera import (
    create_projection_matrix,
    opencv_to_opengl_pose,
    reproject_depth_to_3d,
    reproject_depth_to_3d_batch,
    scale_intrinsics,
)
from .image import (
    extract_foreground_bbox,
    load_image,
    non_max_suppression,
    normalize_image,
    square_crop_and_resize,
)
from .mesh import (
    MeshArrays,
    center_mesh,
    compute_mesh_diameter,
    compute_obb,
    invert_transform,
    load_mesh,
    look_at,
    prepare_mesh_arrays,
    read_mesh,
    sample_sphere_poses,
    scale_mesh,
    to_single_mesh,
)
from .viz import (
    draw_3d_axis,
    draw_3d_box,
    visualize_masks,
    visualize_matches,
)

if TYPE_CHECKING:
    from .moderngl import (
        BatchedAtlasRenderer,
        FullscreenShader,
        create_moderngl_context,
    )
    from .template_renderer import TemplateRenderer

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TemplateRenderer": (".template_renderer", "TemplateRenderer"),
    "create_moderngl_context": (".moderngl.context", "create_moderngl_context"),
    "BatchedAtlasRenderer": (".moderngl.atlas_renderer", "BatchedAtlasRenderer"),
    "FullscreenShader": (".moderngl.fullscreen", "FullscreenShader"),
}

__all__ = [
    "visualize_matches",
    "visualize_masks",
    "draw_3d_box",
    "draw_3d_axis",
    "load_image",
    "extract_foreground_bbox",
    "square_crop_and_resize",
    "normalize_image",
    "non_max_suppression",
    "load_mesh",
    "read_mesh",
    "scale_mesh",
    "center_mesh",
    "to_single_mesh",
    "compute_mesh_diameter",
    "compute_obb",
    "look_at",
    "invert_transform",
    "sample_sphere_poses",
    "MeshArrays",
    "prepare_mesh_arrays",
    "scale_intrinsics",
    "reproject_depth_to_3d",
    "reproject_depth_to_3d_batch",
    "create_projection_matrix",
    "opencv_to_opengl_pose",
    "TemplateRenderer",
    "create_moderngl_context",
    "BatchedAtlasRenderer",
    "FullscreenShader",
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
            if "moderngl" in str(exc):
                raise ImportError(
                    f"'{name}' requires optional rendering dependencies ('moderngl'). "
                    f"Install with: pip install 'spatialhub[render]' or pip install 'spatialhub[gpu,render]'"
                ) from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
