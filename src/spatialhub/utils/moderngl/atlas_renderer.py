"""Batched atlas mesh renderer and offscreen framebuffer manager."""

from __future__ import annotations

import logging

import moderngl
import numpy as np

from .context import create_moderngl_context, validate_moderngl_context

logger = logging.getLogger(__name__)


class AtlasRenderer:
    """Manages offscreen framebuffer attachments for batched 2D grid atlas rendering."""

    def __init__(self, ctx: moderngl.Context | None = None):
        """Initialize the renderer.

        Args:
            ctx: Optional externally created context. If None, creates a standalone context.
        """
        if ctx is not None:
            self.ctx = validate_moderngl_context(ctx)
            self._own_ctx = False
        else:
            self.ctx = create_moderngl_context()
            self._own_ctx = True

        self._color_attachments: list[moderngl.Texture] = []
        self._depth_rb: moderngl.Renderbuffer | None = None
        self._fbo: moderngl.Framebuffer | None = None

        self._current_size = (0, 0)
        self._current_layout: list[tuple[int, str]] = []

        # cached grid tiling state for un-tiling rendered atlas in read_tiles
        self._grid_state = (0, 0, 0, 0, 0)

        # texture format specifiers to numpy dtypes
        self._dtype_map = {
            "f4": np.float32,
            "f2": np.float16,
            "i4": np.int32,
            "u4": np.uint32,
            "i2": np.int16,
            "u2": np.uint16,
            "i1": np.int8,
            "u1": np.uint8,
        }

    def _prepare_resources(self, atlas_W: int, atlas_H: int, layout: list[tuple[int, str]]) -> None:
        """Allocate or resize multi-target color textures and depth renderbuffer attachments."""
        if self._current_size != (atlas_W, atlas_H) or self._current_layout != layout:
            self.release_fbo()

            self._current_size = (atlas_W, atlas_H)
            self._current_layout = layout

            self._color_attachments = [
                self.ctx.texture((atlas_W, atlas_H), components, dtype=dtype)
                for components, dtype in layout
            ]
            self._depth_rb = self.ctx.depth_renderbuffer((atlas_W, atlas_H))

            self._fbo = self.ctx.framebuffer(
                color_attachments=self._color_attachments,
                depth_attachment=self._depth_rb,
            )

    def render(
        self,
        vao: moderngl.VertexArray,
        num_instances: int,
        grid_shape: tuple[int, int],
        tile_shape: tuple[int, int],
        layout: list[tuple[int, str]],
    ) -> None:
        """Configure target framebuffer resolution and execute instanced draw calls.

        Args:
            vao: Instanced vertex array object containing geometry and instance attributes.
            num_instances: Total number of instanced viewpoints (N).
            grid_shape: Number of grid (rows, cols) in atlas layout.
            tile_shape: Spatial (height, width) of each individual tile crop in pixels.
            layout: List of tuples specifying attachment channel count and dtype string (e.g. [(4, 'f4')]).
        """
        n_rows, n_cols = grid_shape
        tile_h, tile_w = tile_shape
        self._grid_state = (num_instances, n_rows, n_cols, tile_h, tile_w)

        atlas_W, atlas_H = n_cols * tile_w, n_rows * tile_h
        self._prepare_resources(atlas_W, atlas_H, layout)

        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES, instances=num_instances)

    def read_tiles(self, attachment_index: int = 0) -> np.ndarray:
        """Read back a framebuffer color attachment texture and un-tile grid cells into a 4D array.

        Args:
            attachment_index: Index of the target color attachment in the layout (default: 0).

        Returns:
            Un-tiled array of shape (N, out_H, out_W, components) in corresponding array dtype.
        """
        if not self._color_attachments or attachment_index >= len(self._color_attachments):
            raise IndexError(f"Attachment index {attachment_index} is invalid for the current layout.")

        num_instances, n_rows, n_cols, tile_h, tile_w = self._grid_state

        tex = self._color_attachments[attachment_index]
        atlas_W, atlas_H = self._current_size

        components, type_str = self._current_layout[attachment_index]
        dtype = self._dtype_map.get(type_str, np.uint8)

        arr = np.frombuffer(tex.read(), dtype=dtype).reshape((atlas_H, atlas_W, components))

        # un-tile 2d grid atlas into instanced tiles
        arr = arr.reshape((n_rows, tile_h, n_cols, tile_w, components))
        arr = arr.transpose((0, 2, 1, 3, 4))

        return arr.reshape((n_rows * n_cols, tile_h, tile_w, components))[:num_instances]

    def release_fbo(self) -> None:
        """Release allocated color textures, depth renderbuffer, and framebuffer objects."""
        if self._fbo:
            for tex in self._color_attachments:
                tex.release()
            self._color_attachments.clear()
            self._depth_rb.release()
            self._fbo.release()
            self._depth_rb = None
            self._fbo = None
            self._current_size = (0, 0)
            self._current_layout = []

    def release(self) -> None:
        """Release allocated framebuffer objects and context if owned."""
        self.release_fbo()
        if hasattr(self, "ctx") and self.ctx is not None and getattr(self, "_own_ctx", True):
            self.ctx.release()
            self.ctx = None

    def close(self) -> None:
        """Alias for release."""
        self.release()

    def __enter__(self) -> AtlasRenderer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

