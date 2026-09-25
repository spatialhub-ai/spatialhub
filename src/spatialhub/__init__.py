import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        CNOS,
        DINOv2,
        DepthAnything3,
        EfficientLoFTR,
        FastSAM,
        FoundationPose,
        SAM,
    )

logging.getLogger(__name__).addHandler(logging.NullHandler())

_MODEL_EXPORTS: dict[str, tuple[str, str]] = {
    "CNOS": (".models.cnos", "CNOSAdapter"),
    "DepthAnything3": (".models.depth_anything_3", "DepthAnything3Adapter"),
    "DINOv2": (".models.dinov2", "DINOv2Adapter"),
    "EfficientLoFTR": (".models.efficient_loftr", "EfficientLoFTRAdapter"),
    "FastSAM": (".models.fastsam", "FastSAMAdapter"),
    "FoundationPose": (".models.foundationpose", "FoundationPoseAdapter"),
    "SAM": (".models.sam", "SAMAdapter"),
}

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
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
