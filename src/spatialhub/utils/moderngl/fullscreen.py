"""
Fullscreen quad shader execution and multi-pass buffer manager.
"""

from __future__ import annotations

import logging

import moderngl
import numpy as np

logger = logging.getLogger(__name__)


class FullscreenShader:
    """
    Manager for executing fullscreen quad fragment shader passes with texture ping-pong swapping.
    """

    def __init__(self, ctx: moderngl.Context | None = None):
        """
        Initialize the fullscreen shader manager.

        Args:
            ctx: Optional execution context instance.
        """
        self.ctx = ctx

        self._tex_in: moderngl.Texture | None = None
        self._tex_out: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._current_shape: tuple[int, int, int] = (0, 0, 0)

    def create_program(self, vert_shader: str, frag_shader: str) -> moderngl.Program:
        """
        Compile vertex and fragment shader sources into a shader program.

        Args:
            vert_shader: Vertex shader source string.
            frag_shader: Fragment shader source string.

        Returns:
            Compiled program object.
        """
        return self.ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

    def create_quad_vbo(self) -> moderngl.Buffer:
        """
        Allocate a 2D quad vertex buffer.

        Returns:
            Buffer containing 2D coordinate vertices.
        """
        quad_coords = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
        return self.ctx.buffer(quad_coords.tobytes())

    def create_vao(
        self,
        program: moderngl.Program,
        vbo: moderngl.Buffer,
        attributes: list[str] = ["in_position"],
    ) -> moderngl.VertexArray:
        """
        Create a Vertex Array Object linking a quad vertex buffer to a program.

        Args:
            program: Target program.
            vbo: Vertex buffer containing quad coordinates.
            attributes: List of vertex attribute names.

        Returns:
            Vertex array object instance.
        """
        return self.ctx.vertex_array(program, [(vbo, "2f", *attributes)])

    def _prepare_resources(self, width: int, height: int, components: int) -> None:
        """Allocate or resize input/output textures and framebuffer object."""
        if self._current_shape != (width, height, components):
            self.release_fbo()
            self._current_shape = (width, height, components)
            self._tex_in = self.ctx.texture((width, height), components, dtype="f4")
            self._tex_out = self.ctx.texture((width, height), components, dtype="f4")
            self._fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])

    def upload(self, input_data: np.ndarray) -> None:
        """
        Upload an input array into the input texture.

        Args:
            input_data: Input array of shape (H, W) or (H, W, C) in float32.
        """
        input_data = np.asarray(input_data, dtype=np.float32)

        if input_data.ndim == 2:
            H, W, C = *input_data.shape, 1
        elif input_data.ndim == 3:
            H, W, C = input_data.shape
        else:
            raise ValueError(f"Input array must be 2D or 3D, got shape {input_data.shape}")

        if C not in (1, 2, 3, 4):
            raise ValueError(f"Textures support 1-4 components. Got {C}.")

        self._prepare_resources(W, H, C)
        self._tex_in.write(input_data.tobytes())

    def render(
        self,
        vao: moderngl.VertexArray,
        prog: moderngl.Program,
        texture_uniform_name: str = "tex_in",
    ) -> None:
        """
        Execute shader pass drawing a fullscreen quad from input to output texture.

        Args:
            vao: Vertex array object for the fullscreen quad.
            prog: Program to execute.
            texture_uniform_name: Uniform name for input texture sampler.
        """
        self._tex_in.use(0)
        if texture_uniform_name in prog:
            prog[texture_uniform_name].value = 0

        self._fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

    def swap_buffers(self) -> None:
        """
        Swap input and output textures for sequential multi-pass processing.
        """
        self._tex_in, self._tex_out = self._tex_out, self._tex_in
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])

    def to_numpy(self) -> np.ndarray:
        """
        Retrieve the rendered output texture data as an array.

        Returns:
            Array of shape (H, W) or (H, W, C) in float32.
        """
        if not self._tex_out:
            raise RuntimeError("No rendered data available. Call upload() and render() first.")

        W, H, C = self._current_shape
        out_bytes = self._tex_out.read()
        out_arr = np.frombuffer(out_bytes, dtype=np.float32).reshape((H, W, C))
        return np.squeeze(out_arr, axis=2) if C == 1 else out_arr

    def process(
        self,
        vao: moderngl.VertexArray,
        prog: moderngl.Program,
        input_data: np.ndarray,
        texture_uniform_name: str = "tex_in",
    ) -> np.ndarray:
        """
        Execute upload, render, and retrieve steps in a single call.

        Args:
            vao: Vertex array object.
            prog: Program to execute.
            input_data: Input array of shape (H, W) or (H, W, C).
            texture_uniform_name: Uniform name for input texture sampler.

        Returns:
            Rendered output array.
        """
        self.upload(input_data)
        self.render(vao, prog, texture_uniform_name)
        return self.to_numpy()

    def release_fbo(self) -> None:
        """Release allocated textures and framebuffer objects."""
        if self._tex_in:
            self._tex_in.release()
            self._tex_out.release()
            self._fbo.release()
            self._tex_in = self._tex_out = self._fbo = None
            self._current_shape = (0, 0, 0)
