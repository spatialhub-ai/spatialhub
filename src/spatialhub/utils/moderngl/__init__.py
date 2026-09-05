"""
Offscreen rendering and batched atlas generation backend.
"""

from .context import create_moderngl_context
from .fullscreen import FullscreenShader
from .renderer import BatchedAtlasRenderer

__all__ = [
    "create_moderngl_context",
    "BatchedAtlasRenderer",
    "FullscreenShader",
]