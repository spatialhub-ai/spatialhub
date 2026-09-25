import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cnos import CNOSAdapter as CNOS
    from .depth_anything_3 import DepthAnything3Adapter as DepthAnything3
    from .dinov2 import DINOv2Adapter as DINOv2
    from .efficient_loftr import EfficientLoFTRAdapter as EfficientLoFTR
    from .fastsam import FastSAMAdapter as FastSAM
    from .foundationpose import FoundationPoseAdapter as FoundationPose
    from .sam import SAMAdapter as SAM

_MODEL_EXPORTS: dict[str, tuple[str, str]] = {
    "CNOS": (".cnos", "CNOSAdapter"),
    "DepthAnything3": (".depth_anything_3", "DepthAnything3Adapter"),
    "DINOv2": (".dinov2", "DINOv2Adapter"),
    "EfficientLoFTR": (".efficient_loftr", "EfficientLoFTRAdapter"),
    "FastSAM": (".fastsam", "FastSAMAdapter"),
    "FoundationPose": (".foundationpose", "FoundationPoseAdapter"),
    "SAM": (".sam", "SAMAdapter"),
}

__all__ = [
    "CNOS",
    "DINOv2",
    "DepthAnything3",
    "EfficientLoFTR",
    "FastSAM",
    "FoundationPose",
    "SAM",
]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        module_path, attr_name = _MODEL_EXPORTS[name]
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
