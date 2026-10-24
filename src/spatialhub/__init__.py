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
    if name in __all__:
        models = importlib.import_module(".models", package=__name__)
        if hasattr(models, name):
            val = getattr(models, name)
            globals()[name] = val
            return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
