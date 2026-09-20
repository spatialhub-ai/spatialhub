"""
Batched atlas mesh renderer and offscreen framebuffer manager.
"""

from __future__ import annotations

import logging

import moderngl
import numpy as np

from .context import create_moderngl_context

logger = logging.getLogger(__name__)


class BatchedAtlasRenderer:
    """
    Manages offscreen framebuffer attachments and dynamic allocation for batched 2D grid atlas rendering.
    """

    def __init__(self, ctx: moderngl.Context):
        """
        Initialize the BatchedAtlasRenderer with an execution context.

        Args:
            ctx: Execution context instance.
        """
        self.ctx = ctx

        self._color_attachments: list[moderngl.Texture] = []
        self._depth_rb: moderngl.Renderbuffer | None = None
        self._fbo: moderngl.Framebuffer | None = None

        self._current_size = (0, 0)
        self._current_layout: list[tuple[int, str]] = []
        self._grid_state = (0, 0, 0, 0, 0)  # Caches (N, C, R, out_H, out_W)

        # Map texture data format specifiers to array dtypes
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
        N: int,
        C: int,
        R: int,
        out_H: int,
        out_W: int,
        layout: list[tuple[int, str]],
    ) -> None:
        """
        Configure target framebuffer resolution and execute instanced draw calls.

        Args:
            vao: Instanced vertex array object containing geometry and instance attributes.
            N: Total number of instanced viewpoints.
            C: Number of grid columns in atlas layout.
            R: Number of grid rows in atlas layout.
            out_H: Spatial height of each individual tile crop in pixels.
            out_W: Spatial width of each individual tile crop in pixels.
            layout: List of tuples specifying attachment channel count and dtype string (e.g. [(4, 'f4')]).
        """
        self._grid_state = (N, C, R, out_H, out_W)

        atlas_W, atlas_H = C * out_W, R * out_H
        self._prepare_resources(atlas_W, atlas_H, layout)

        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES, instances=N)

    def unpack_attachment(self, attachment_index: int) -> np.ndarray:
        """
        Read back a framebuffer color attachment texture and un-tile grid cells into a 4D array.

        Args:
            attachment_index: Index of the target color attachment in the G-Buffer layout.

        Returns:
            Un-tiled array of shape (N, out_H, out_W, components) in corresponding array dtype.
        """
        if not self._color_attachments or attachment_index >= len(self._color_attachments):
            raise IndexError(f"Attachment index {attachment_index} is invalid for the current layout.")

        N, C, R, out_H, out_W = self._grid_state

        tex = self._color_attachments[attachment_index]
        atlas_W, atlas_H = self._current_size

        components, type_str = self._current_layout[attachment_index]
        dtype = self._dtype_map.get(type_str, np.uint8)

        arr = np.frombuffer(tex.read(), dtype=dtype).reshape((atlas_H, atlas_W, components))

        # Un-tile 2D grid atlas (R * out_H, C * out_W, components) -> (N, out_H, out_W, components)
        arr = arr.reshape((R, out_H, C, out_W, components))
        arr = arr.transpose((0, 2, 1, 3, 4))

        return arr.reshape((R * C, out_H, out_W, components))[:N]

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
