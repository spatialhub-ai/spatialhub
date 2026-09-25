"""Offscreen rendering and batched atlas generation backend."""
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .atlas_renderer import BatchedAtlasRenderer
    from .context import create_moderngl_context
    from .fullscreen import FullscreenShader

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "create_moderngl_context": (".context", "create_moderngl_context"),
    "BatchedAtlasRenderer": (".atlas_renderer", "BatchedAtlasRenderer"),
    "FullscreenShader": (".fullscreen", "FullscreenShader"),
}

__all__ = [
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
