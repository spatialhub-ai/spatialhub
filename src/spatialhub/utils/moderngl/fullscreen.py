"""Fullscreen quad shader execution and multi-pass buffer manager."""

from __future__ import annotations

import logging

import moderngl
import numpy as np

from .context import create_moderngl_context, validate_moderngl_context

logger = logging.getLogger(__name__)


class FullscreenRenderer:
    """Executes fullscreen quad fragment shader passes with texture ping-pong swapping."""

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

        self._tex_in: moderngl.Texture | None = None
        self._tex_out: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._current_shape: tuple[int, int, int] = (0, 0, 0)

        # allocate fullscreen quad in 2d ndc [-1, 1]
        quad_coords = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
        self._quad_vbo: moderngl.Buffer = self.ctx.buffer(quad_coords.tobytes())
        
        self._vao_cache: dict[int, moderngl.VertexArray] = {}

    def create_program(self, vert_shader: str, frag_shader: str) -> moderngl.Program:
        """Compile vertex and fragment shader sources into a program.

        Args:
            vert_shader: Vertex shader source. Must declare `in vec2 in_position;`.
            frag_shader: Fragment shader source.

        Returns:
            Compiled program object.
        """
        return self.ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

    def _create_vao(
        self,
        program: moderngl.Program,
        vbo: moderngl.Buffer | None = None,
        attributes: list[str] = ["in_position"],
    ) -> moderngl.VertexArray:
        """Create a vertex array object linking a quad buffer to a program.

        Args:
            program: Target program. Must declare attributes matching `attributes`.
            vbo: Optional vertex buffer. Defaults to the internal quad buffer.
            attributes: Vertex attribute names (default: `["in_position"]`).

        Returns:
            Vertex array object instance.
        """
        target_vbo = vbo if vbo is not None else self._quad_vbo
        return self.ctx.vertex_array(program, [(target_vbo, "2f", *attributes)])

    def _get_vao(self, prog: moderngl.Program) -> moderngl.VertexArray:
        """Retrieve or create a cached vertex array object for the given program."""
        prog_id = prog.glo
        if prog_id not in self._vao_cache:
            self._vao_cache[prog_id] = self._create_vao(prog)
        return self._vao_cache[prog_id]

    def _prepare_resources(self, width: int, height: int, components: int) -> None:
        """Allocate or resize input/output textures and framebuffer object."""
        if self._current_shape != (width, height, components):
            self.release_fbo()
            self._current_shape = (width, height, components)
            self._tex_in = self.ctx.texture((width, height), components, dtype="f4")
            self._tex_out = self.ctx.texture((width, height), components, dtype="f4")
            self._fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])

    def upload(self, input_data: np.ndarray) -> None:
        """Upload an input array into the input texture.

        Args:
            input_data: Array of shape (H, W) or (H, W, C) in float32.
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

    def render_pass(
        self,
        prog: moderngl.Program,
        vao: moderngl.VertexArray | None = None,
        texture_uniform_name: str = "tex_in",
    ) -> None:
        """Execute a shader pass drawing a fullscreen quad from input to output texture.

        Args:
            prog: Program to execute. Must declare `in vec2 in_position;`.
            vao: Optional vertex array object. If None, uses the internal quad VAO for `prog`.
            texture_uniform_name: Uniform name for the input texture sampler.
        """
        target_vao = vao if vao is not None else self._get_vao(prog)

        self._tex_in.use(0)
        if texture_uniform_name in prog:
            prog[texture_uniform_name].value = 0

        self._fbo.use()
        target_vao.render(moderngl.TRIANGLE_STRIP)

    def swap_buffers(self) -> None:
        """Swap input and output textures for sequential multi-pass processing."""
        self._tex_in, self._tex_out = self._tex_out, self._tex_in
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])

    def to_numpy(self) -> np.ndarray:
        """Read the rendered output texture into a NumPy array.

        Returns:
            Array of shape (H, W) or (H, W, C) in float32.
        """
        if not self._tex_out:
            raise RuntimeError("No rendered data available. Call upload() and render_pass() first.")

        W, H, C = self._current_shape
        out_bytes = self._tex_out.read()
        out_arr = np.frombuffer(out_bytes, dtype=np.float32).reshape((H, W, C))
        return np.squeeze(out_arr, axis=2) if C == 1 else out_arr

    def apply(
        self,
        prog: moderngl.Program,
        input_data: np.ndarray,
        vao: moderngl.VertexArray | None = None,
        texture_uniform_name: str = "tex_in",
    ) -> np.ndarray:
        """Execute upload, render_pass, and to_numpy in a single call.

        Args:
            prog: Program to execute. Must declare `in vec2 in_position;`.
            input_data: Array of shape (H, W) or (H, W, C).
            vao: Optional vertex array object. If None, uses the internal quad VAO for `prog`.
            texture_uniform_name: Uniform name for the input texture sampler.

        Returns:
            Rendered output array.
        """
        self.upload(input_data)
        self.render_pass(prog=prog, vao=vao, texture_uniform_name=texture_uniform_name)
        return self.to_numpy()

    def release_fbo(self) -> None:
        """Release allocated textures and framebuffer objects."""
        if self._tex_in:
            self._tex_in.release()
            self._tex_out.release()
            self._fbo.release()
            self._tex_in = self._tex_out = self._fbo = None
            self._current_shape = (0, 0, 0)

    def release(self) -> None:
        """Release quad buffers, cached VAOs, textures, framebuffer, and owned context."""
        self.release_fbo()
        for vao in self._vao_cache.values():
            vao.release()
        self._vao_cache.clear()
        if hasattr(self, "_quad_vbo") and self._quad_vbo is not None:
            self._quad_vbo.release()
            self._quad_vbo = None
        if hasattr(self, "ctx") and self.ctx is not None and getattr(self, "_own_ctx", True):
            self.ctx.release()
            self.ctx = None

    def close(self) -> None:
        """Alias for release."""
        self.release()

    def __enter__(self) -> FullscreenRenderer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

