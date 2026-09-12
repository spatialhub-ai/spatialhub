"""Comprehensive unit tests for spatialhub.utils.moderngl backend."""

from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pytest

try:
    import moderngl
    from spatialhub.utils.moderngl.context import create_moderngl_context
    from spatialhub.utils.moderngl.atlas_renderer import BatchedAtlasRenderer
    from spatialhub.utils.moderngl.fullscreen import FullscreenShader

    RENDERER_AVAILABLE = True
except Exception:
    RENDERER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RENDERER_AVAILABLE,
    reason="ModernGL required for moderngl backend tests",
)


@pytest.fixture(scope="module")
def gl_ctx() -> moderngl.Context:
    """Create a shared ModernGL headless context for the test module."""
    try:
        ctx = create_moderngl_context(standalone=True, require_version=330)
        yield ctx
    except Exception as err:
        pytest.skip(f"ModernGL context creation failed: {err}")


# ============================================================================
# 1. Context Tests
# ============================================================================


class TestModernGLContext:
    """Tests for create_moderngl_context."""

    def test_create_moderngl_context_success(self, gl_ctx: moderngl.Context):
        assert gl_ctx is not None
        assert isinstance(gl_ctx, moderngl.Context)

    def test_create_moderngl_context_custom_args(self):
        try:
            ctx = create_moderngl_context(standalone=True, require_version=330)
            assert isinstance(ctx, moderngl.Context)
        except Exception as err:
            pytest.skip(f"Context creation unavailable: {err}")

    def test_create_moderngl_context_failure_raises(self):
        with patch("moderngl.create_context", side_effect=RuntimeError("GL context init error")):
            with pytest.raises(RuntimeError, match="GL context init error"):
                create_moderngl_context()


# ============================================================================
# 2. BatchedAtlasRenderer Tests
# ============================================================================


class TestBatchedAtlasRenderer:
    """Tests for BatchedAtlasRenderer framebuffer allocation, rendering, and un-tiling."""

    def test_init_state(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)
        assert renderer.ctx is gl_ctx
        assert renderer._color_attachments == []
        assert renderer._depth_rb is None
        assert renderer._fbo is None
        assert renderer._current_size == (0, 0)
        assert renderer._current_layout == []
        assert renderer._grid_state == (0, 0, 0, 0, 0)
        renderer.release_fbo()

    def test_prepare_resources_allocation_and_reuse(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)
        layout = [(4, "f4")]

        # First allocation
        renderer._prepare_resources(64, 64, layout)
        assert renderer._current_size == (64, 64)
        assert renderer._current_layout == layout
        assert len(renderer._color_attachments) == 1
        assert renderer._depth_rb is not None
        assert renderer._fbo is not None

        first_tex = renderer._color_attachments[0]
        first_fbo = renderer._fbo

        # Subsequent call with identical specs should reuse existing buffers
        renderer._prepare_resources(64, 64, layout)
        assert renderer._color_attachments[0] is first_tex
        assert renderer._fbo is first_fbo

        # Resize should release and reallocate new buffers
        renderer._prepare_resources(128, 128, layout)
        assert renderer._current_size == (128, 128)
        assert renderer._color_attachments[0] is not first_tex

        renderer.release_fbo()

    def test_prepare_resources_multi_attachment(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)
        layout = [(4, "u1"), (1, "f4"), (2, "f2")]

        renderer._prepare_resources(32, 32, layout)
        assert len(renderer._color_attachments) == 3
        assert renderer._current_layout == layout
        renderer.release_fbo()

    def test_render_and_unpack_attachment_single_tile(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)

        # Simple program that renders a red quad
        prog = gl_ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
                }
            """,
        )

        # Fullscreen-covering two-triangle quad
        vertices = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                -1.0,  1.0,
                 1.0, -1.0,
                 1.0,  1.0,
            ],
            dtype="f4",
        )
        vbo = gl_ctx.buffer(vertices.tobytes())
        vao = gl_ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])

        # Render 1 viewpoint into a 1x1 grid with size 16x16
        renderer.render(vao, N=1, C=1, R=1, out_H=16, out_W=16, layout=[(4, "u1")])
        out = renderer.unpack_attachment(0)

        assert out.shape == (1, 16, 16, 4)
        assert out.dtype == np.uint8
        # All pixels should be red [255, 0, 0, 255]
        assert np.all(out[0, :, :, 0] == 255)
        assert np.all(out[0, :, :, 1] == 0)
        assert np.all(out[0, :, :, 2] == 0)
        assert np.all(out[0, :, :, 3] == 255)

        vbo.release()
        vao.release()
        prog.release()
        renderer.release_fbo()

    def test_render_batched_grid_and_unpack_slicing(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)

        prog = gl_ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(0.5, 0.5, 0.5, 1.0);
                }
            """,
        )
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, 0.0, 1.0], dtype="f4")
        vbo = gl_ctx.buffer(vertices.tobytes())
        vao = gl_ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])

        # 3 viewpoints in a 2x2 grid (R=2, C=2, out_H=8, out_W=8)
        renderer.render(vao, N=3, C=2, R=2, out_H=8, out_W=8, layout=[(4, "f4")])
        out = renderer.unpack_attachment(0)

        # Must slice [:N], so returned shape is (3, 8, 8, 4) instead of 4
        assert out.shape == (3, 8, 8, 4)
        assert out.dtype == np.float32

        vbo.release()
        vao.release()
        prog.release()
        renderer.release_fbo()

    @pytest.mark.parametrize(
        ("dtype_str", "expected_dtype"),
        [
            ("f4", np.float32),
            ("f2", np.float16),
            ("i4", np.int32),
            ("u4", np.uint32),
            ("i2", np.int16),
            ("u2", np.uint16),
            ("i1", np.int8),
            ("u1", np.uint8),
            ("unknown_format", np.uint8),  # Fallback default
        ],
    )
    def test_unpack_attachment_dtypes(self, gl_ctx: moderngl.Context, dtype_str: str, expected_dtype: type):
        renderer = BatchedAtlasRenderer(gl_ctx)

        # Mock the grid state and texture read for various dtypes
        renderer._grid_state = (2, 2, 1, 4, 4)
        renderer._current_size = (8, 4)
        renderer._current_layout = [(1, dtype_str)]

        mock_tex = gl_ctx.texture((8, 4), 1, dtype="u1" if dtype_str == "unknown_format" else dtype_str)
        # Write dummy bytes
        dummy_data = np.zeros((4, 8, 1), dtype=expected_dtype)
        mock_tex.write(dummy_data.tobytes())
        renderer._color_attachments = [mock_tex]

        try:
            unpacked = renderer.unpack_attachment(0)
            assert unpacked.shape == (2, 4, 4, 1)
            assert unpacked.dtype == expected_dtype
        finally:
            mock_tex.release()
            renderer._color_attachments.clear()

    def test_unpack_attachment_invalid_index_raises(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)

        # Unpack when no color attachments are allocated
        with pytest.raises(IndexError, match="Attachment index 0 is invalid"):
            renderer.unpack_attachment(0)

        renderer._prepare_resources(16, 16, [(4, "u1")])
        with pytest.raises(IndexError, match="Attachment index 2 is invalid"):
            renderer.unpack_attachment(2)

        renderer.release_fbo()

    def test_release_fbo_idempotent(self, gl_ctx: moderngl.Context):
        renderer = BatchedAtlasRenderer(gl_ctx)
        renderer._prepare_resources(16, 16, [(4, "u1")])

        assert renderer._fbo is not None
        renderer.release_fbo()

        assert renderer._fbo is None
        assert renderer._depth_rb is None
        assert renderer._color_attachments == []
        assert renderer._current_size == (0, 0)
        assert renderer._current_layout == []

        # Calling again should not raise errors
        renderer.release_fbo()


# ============================================================================
# 3. FullscreenShader Tests
# ============================================================================

class TestFullscreenShader:
    """Tests for FullscreenShader quad setup, texture upload/download, and ping-pong swapping."""

    @pytest.fixture
    def vert_shader_src(self) -> str:
        return """
            #version 330
            in vec2 in_position;
            out vec2 v_uv;
            void main() {
                v_uv = (in_position + 1.0) * 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
        """

    @pytest.fixture
    def invert_frag_shader_src(self) -> str:
        return """
            #version 330
            uniform sampler2D tex_in;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                vec4 val = texture(tex_in, v_uv);
                fragColor = vec4(1.0 - val.r, 1.0 - val.g, 1.0 - val.b, val.a);
            }
        """

    def test_init_state(self, gl_ctx: moderngl.Context):
        fs = FullscreenShader(gl_ctx)
        assert fs.ctx is gl_ctx
        assert fs._tex_in is None
        assert fs._tex_out is None
        assert fs._fbo is None
        assert fs._current_shape == (0, 0, 0)
        fs.release_fbo()

    def test_create_program_and_quad_vao(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
        invert_frag_shader_src: str,
    ):
        fs = FullscreenShader(gl_ctx)
        prog = fs.create_program(vert_shader_src, invert_frag_shader_src)
        assert isinstance(prog, moderngl.Program)

        vbo = fs.create_quad_vbo()
        assert isinstance(vbo, moderngl.Buffer)
        assert vbo.size == 8 * 4  # 8 floats * 4 bytes

        vao = fs.create_vao(prog, vbo, attributes=["in_position"])
        assert isinstance(vao, moderngl.VertexArray)

        vao.release()
        vbo.release()
        prog.release()

    def test_upload_2d_and_3d(self, gl_ctx: moderngl.Context):
        fs = FullscreenShader(gl_ctx)

        # 2D Grayscale input (H=16, W=32) -> shape (32, 16, 1)
        img_2d = np.ones((16, 32), dtype=np.float32)
        fs.upload(img_2d)
        assert fs._current_shape == (32, 16, 1)
        assert fs._tex_in.size == (32, 16)
        assert fs._tex_in.components == 1

        # 3D 3-channel input (H=16, W=32, C=3)
        img_3d = np.ones((16, 32, 3), dtype=np.float32)
        fs.upload(img_3d)
        assert fs._current_shape == (32, 16, 3)
        assert fs._tex_in.components == 3

        # 3D 4-channel input (H=8, W=8, C=4)
        img_4d = np.ones((8, 8, 4), dtype=np.float32)
        fs.upload(img_4d)
        assert fs._current_shape == (8, 8, 4)
        assert fs._tex_in.components == 4

        fs.release_fbo()

    def test_upload_invalid_shapes_raises(self, gl_ctx: moderngl.Context):
        fs = FullscreenShader(gl_ctx)

        # 1D array
        with pytest.raises(ValueError, match="Input array must be 2D or 3D"):
            fs.upload(np.ones((10,), dtype=np.float32))

        # 4D array
        with pytest.raises(ValueError, match="Input array must be 2D or 3D"):
            fs.upload(np.ones((1, 10, 10, 3), dtype=np.float32))

        # 3D with 0 channels
        with pytest.raises(ValueError, match="Textures support 1-4 components"):
            fs.upload(np.ones((10, 10, 0), dtype=np.float32))

        # 3D with 5 channels
        with pytest.raises(ValueError, match="Textures support 1-4 components"):
            fs.upload(np.ones((10, 10, 5), dtype=np.float32))

    def test_render_and_to_numpy_2d(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
    ):
        fs = FullscreenShader(gl_ctx)

        # Multiplier shader: multiplies input by 2.0
        double_frag_shader = """
            #version 330
            uniform sampler2D tex_in;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                float val = texture(tex_in, v_uv).r;
                fragColor = vec4(val * 2.0, 0.0, 0.0, 1.0);
            }
        """
        prog = fs.create_program(vert_shader_src, double_frag_shader)
        vbo = fs.create_quad_vbo()
        vao = fs.create_vao(prog, vbo, attributes=["in_position"])

        input_data = np.full((16, 16), 0.25, dtype=np.float32)
        fs.upload(input_data)
        fs.render(vao, prog, texture_uniform_name="tex_in")
        output = fs.to_numpy()

        # For 1-channel input, output is squeezed to (H, W)
        assert output.shape == (16, 16)
        assert output.dtype == np.float32
        np.testing.assert_allclose(output, 0.5, atol=1e-5)

        vao.release()
        vbo.release()
        prog.release()
        fs.release_fbo()

    def test_to_numpy_without_render_raises(self, gl_ctx: moderngl.Context):
        fs = FullscreenShader(gl_ctx)
        with pytest.raises(RuntimeError, match="No rendered data available"):
            fs.to_numpy()

    def test_swap_buffers_multipass(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
    ):
        fs = FullscreenShader(gl_ctx)

        # Add 0.1 each pass
        add_frag_shader = """
            #version 330
            uniform sampler2D tex_in;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                vec4 val = texture(tex_in, v_uv);
                fragColor = vec4(val.r + 0.1, val.g + 0.1, val.b + 0.1, 1.0);
            }
        """
        prog = fs.create_program(vert_shader_src, add_frag_shader)
        vbo = fs.create_quad_vbo()
        vao = fs.create_vao(prog, vbo, attributes=["in_position"])

        input_data = np.zeros((8, 8, 3), dtype=np.float32)
        fs.upload(input_data)

        # Pass 1: 0.0 + 0.1 = 0.1
        fs.render(vao, prog)
        # Swap: output becomes next input
        fs.swap_buffers()
        # Pass 2: 0.1 + 0.1 = 0.2
        fs.render(vao, prog)

        result = fs.to_numpy()
        assert result.shape == (8, 8, 3)
        np.testing.assert_allclose(result[..., 0], 0.2, atol=1e-5)
        np.testing.assert_allclose(result[..., 1], 0.2, atol=1e-5)
        np.testing.assert_allclose(result[..., 2], 0.2, atol=1e-5)

        vao.release()
        vbo.release()
        prog.release()
        fs.release_fbo()

    def test_process_convenience_method(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
        invert_frag_shader_src: str,
    ):
        fs = FullscreenShader(gl_ctx)
        prog = fs.create_program(vert_shader_src, invert_frag_shader_src)
        vbo = fs.create_quad_vbo()
        vao = fs.create_vao(prog, vbo, attributes=["in_position"])

        input_data = np.array([[[0.2, 0.4, 0.6, 1.0]]], dtype=np.float32)
        output = fs.process(vao, prog, input_data)

        assert output.shape == (1, 1, 4)
        np.testing.assert_allclose(output[0, 0, 0], 0.8, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 1], 0.6, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 2], 0.4, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 3], 1.0, atol=1e-5)

        vao.release()
        vbo.release()
        prog.release()
        fs.release_fbo()

    def test_release_fbo_idempotent(self, gl_ctx: moderngl.Context):
        fs = FullscreenShader(gl_ctx)
        fs.upload(np.ones((8, 8), dtype=np.float32))

        assert fs._tex_in is not None
        assert fs._tex_out is not None
        assert fs._fbo is not None

        fs.release_fbo()
        assert fs._tex_in is None
        assert fs._tex_out is None
        assert fs._fbo is None
        assert fs._current_shape == (0, 0, 0)

        # Calling again should be a clean no-op
        fs.release_fbo()
