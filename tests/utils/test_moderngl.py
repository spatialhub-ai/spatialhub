"""Comprehensive unit tests for spatialhub.utils.moderngl backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

try:
    import moderngl
    import spatialhub.utils.moderngl as mgl
    from spatialhub.utils.moderngl.atlas_renderer import AtlasRenderer
    from spatialhub.utils.moderngl.context import create_moderngl_context, validate_moderngl_context
    from spatialhub.utils.moderngl.fullscreen import FullscreenRenderer

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


class TestModernGLPackage:
    """Tests for spatialhub.utils.moderngl lazy exports and package attributes."""

    def test_lazy_exports(self) -> None:
        assert mgl.create_moderngl_context is create_moderngl_context
        assert mgl.validate_moderngl_context is validate_moderngl_context
        assert mgl.AtlasRenderer is AtlasRenderer
        assert mgl.FullscreenRenderer is FullscreenRenderer

    def test_dir_contains_all(self) -> None:
        exported = dir(mgl)
        for name in [
            "create_moderngl_context",
            "validate_moderngl_context",
            "AtlasRenderer",
            "FullscreenRenderer",
        ]:
            assert name in exported

    def test_invalid_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
            _ = mgl.invalid_attr  # type: ignore[attr-defined]


class TestModernGLContext:
    """Tests for create_moderngl_context and validate_moderngl_context."""

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

    def test_validate_context_success(self, gl_ctx: moderngl.Context):
        validated = validate_moderngl_context(gl_ctx)
        assert validated is gl_ctx

    def test_validate_context_invalid_type(self):
        with pytest.raises(TypeError, match="Expected moderngl.Context, got str"):
            validate_moderngl_context("not_a_context")

    def test_validate_context_released(self):
        mock_released_ctx = MagicMock(spec=moderngl.Context)
        mock_invalid_obj = type("InvalidObject", (), {})()
        mock_released_ctx.mglo = mock_invalid_obj

        with pytest.raises(RuntimeError, match="ModernGL context has been released and is no longer valid"):
            validate_moderngl_context(mock_released_ctx)


class TestAtlasRenderer:
    """Tests for AtlasRenderer framebuffer allocation, multi-attachment rendering, and un-tiling."""

    def test_init_state(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)
        assert renderer.ctx is gl_ctx
        assert renderer._color_attachments == []
        assert renderer._depth_rb is None
        assert renderer._fbo is None
        assert renderer._current_size == (0, 0)
        assert renderer._current_layout == []
        assert renderer._grid_state == (0, 0, 0, 0, 0)
        renderer.release_fbo()

    def test_prepare_resources_allocation_and_reuse(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)
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

        # Same dimensions and layout should reuse cached resources
        renderer._prepare_resources(64, 64, layout)
        assert renderer._color_attachments[0] is first_tex
        assert renderer._fbo is first_fbo

        # Different dimensions should reallocate
        renderer._prepare_resources(128, 128, layout)
        assert renderer._current_size == (128, 128)
        assert renderer._color_attachments[0] is not first_tex

        # Different layout should reallocate
        layout_multi = [(4, "f4"), (4, "f4")]
        renderer._prepare_resources(128, 128, layout_multi)
        assert len(renderer._color_attachments) == 2

        renderer.release_fbo()

    def test_render_and_read_tiles_single_tile(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)

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
        renderer.render(
            vao,
            num_instances=1,
            grid_shape=(1, 1),
            tile_shape=(16, 16),
            layout=[(4, "u1")],
        )

        # Test read_tiles with default attachment_index=0
        out_default = renderer.read_tiles()
        assert out_default.shape == (1, 16, 16, 4)
        assert out_default.dtype == np.uint8
        # Red quad [255, 0, 0, 255]
        assert np.all(out_default[0, :, :, 0] == 255)
        assert np.all(out_default[0, :, :, 1] == 0)
        assert np.all(out_default[0, :, :, 2] == 0)
        assert np.all(out_default[0, :, :, 3] == 255)

        # Explicit index 0 matches default
        out_explicit = renderer.read_tiles(0)
        assert np.array_equal(out_default, out_explicit)

        vbo.release()
        vao.release()
        prog.release()
        renderer.release_fbo()

    def test_render_multi_target_g_buffer(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)

        # Multi-target program outputting to 2 color attachments:
        # attachment 0: RGBA color (0.2, 0.4, 0.6, 1.0)
        # attachment 1: Normal vector (0.0, 1.0, 0.0, 1.0)
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
                layout(location = 0) out vec4 out_color;
                layout(location = 1) out vec4 out_normal;
                void main() {
                    out_color = vec4(0.2, 0.4, 0.6, 1.0);
                    out_normal = vec4(0.0, 1.0, 0.0, 1.0);
                }
            """,
        )

        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
        vbo = gl_ctx.buffer(vertices.tobytes())
        vao = gl_ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])

        renderer.render(
            vao,
            num_instances=1,
            grid_shape=(1, 1),
            tile_shape=(8, 8),
            layout=[(4, "f4"), (4, "f4")],
        )

        color_tiles = renderer.read_tiles(0)
        normal_tiles = renderer.read_tiles(1)

        assert color_tiles.shape == (1, 8, 8, 4)
        assert normal_tiles.shape == (1, 8, 8, 4)
        np.testing.assert_allclose(color_tiles[0, 0, 0], [0.2, 0.4, 0.6, 1.0], atol=1e-4)
        np.testing.assert_allclose(normal_tiles[0, 0, 0], [0.0, 1.0, 0.0, 1.0], atol=1e-4)

        vbo.release()
        vao.release()
        prog.release()
        renderer.release_fbo()

    def test_render_batched_grid_and_read_tiles_slicing(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)

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
        renderer.render(
            vao,
            num_instances=3,
            grid_shape=(2, 2),
            tile_shape=(8, 8),
            layout=[(4, "f4")],
        )
        out = renderer.read_tiles(0)

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
            ("unknown_format", np.uint8),
        ],
    )
    def test_read_tiles_dtypes(self, gl_ctx: moderngl.Context, dtype_str: str, expected_dtype: type):
        renderer = AtlasRenderer(gl_ctx)

        renderer._grid_state = (2, 1, 2, 4, 4)
        renderer._current_size = (8, 4)
        renderer._current_layout = [(1, dtype_str)]

        mock_tex = gl_ctx.texture((8, 4), 1, dtype="u1" if dtype_str == "unknown_format" else dtype_str)
        dummy_data = np.zeros((4, 8, 1), dtype=expected_dtype)
        mock_tex.write(dummy_data.tobytes())
        renderer._color_attachments = [mock_tex]

        try:
            unpacked = renderer.read_tiles(0)
            assert unpacked.shape == (2, 4, 4, 1)
            assert unpacked.dtype == expected_dtype
        finally:
            mock_tex.release()
            renderer._color_attachments.clear()

    def test_read_tiles_invalid_index_raises(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)

        # No color attachments allocated
        with pytest.raises(IndexError, match="Attachment index 0 is invalid"):
            renderer.read_tiles(0)

        renderer._prepare_resources(16, 16, [(4, "u1")])
        with pytest.raises(IndexError, match="Attachment index 2 is invalid"):
            renderer.read_tiles(2)

        renderer.release_fbo()

    def test_release_fbo_idempotent(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)
        renderer._prepare_resources(16, 16, [(4, "u1")])

        assert renderer._fbo is not None
        renderer.release_fbo()

        assert renderer._fbo is None
        assert renderer._depth_rb is None
        assert renderer._color_attachments == []
        assert renderer._current_size == (0, 0)
        assert renderer._current_layout == []

        renderer.release_fbo()

    def test_close_alias(self, gl_ctx: moderngl.Context):
        renderer = AtlasRenderer(gl_ctx)
        renderer._prepare_resources(16, 16, [(4, "u1")])
        renderer.close()
        assert renderer._fbo is None

    def test_context_ownership_and_release(self):
        mock_ctx = MagicMock(spec=moderngl.Context)
        renderer_borrowed = AtlasRenderer(mock_ctx)
        assert not renderer_borrowed._own_ctx
        renderer_borrowed.release()
        mock_ctx.release.assert_not_called()
        assert renderer_borrowed.ctx is mock_ctx

        with patch("spatialhub.utils.moderngl.atlas_renderer.create_moderngl_context", return_value=mock_ctx):
            with AtlasRenderer() as renderer_owned:
                assert renderer_owned._own_ctx
            mock_ctx.release.assert_called_once()
            assert renderer_owned.ctx is None


class TestFullscreenRenderer:
    """Tests for FullscreenRenderer quad setup, shader execution, texture upload/download, and ping-pong swapping."""

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
        fs = FullscreenRenderer(gl_ctx)
        assert fs.ctx is gl_ctx
        assert fs._tex_in is None
        assert fs._tex_out is None
        assert fs._fbo is None
        assert fs._current_shape == (0, 0, 0)
        assert fs._quad_vbo is not None
        fs.release_fbo()

    def test_create_program_and_quad_vao(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
        invert_frag_shader_src: str,
    ):
        fs = FullscreenRenderer(gl_ctx)
        prog = fs.create_program(vert_shader_src, invert_frag_shader_src)
        assert isinstance(prog, moderngl.Program)

        assert isinstance(fs._quad_vbo, moderngl.Buffer)
        assert fs._quad_vbo.size == 8 * 4  # 8 floats * 4 bytes

        vao = fs._create_vao(prog, attributes=["in_position"])
        assert isinstance(vao, moderngl.VertexArray)

        cached_vao = fs._get_vao(prog)
        assert cached_vao is not None

        vao.release()
        prog.release()

    def test_upload_2d_and_3d(self, gl_ctx: moderngl.Context):
        fs = FullscreenRenderer(gl_ctx)

        # 2D Grayscale input (H=16, W=32)
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
        fs = FullscreenRenderer(gl_ctx)

        with pytest.raises(ValueError, match="Input array must be 2D or 3D"):
            fs.upload(np.ones((10,), dtype=np.float32))

        with pytest.raises(ValueError, match="Input array must be 2D or 3D"):
            fs.upload(np.ones((1, 10, 10, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="Textures support 1-4 components"):
            fs.upload(np.ones((10, 10, 0), dtype=np.float32))

        with pytest.raises(ValueError, match="Textures support 1-4 components"):
            fs.upload(np.ones((10, 10, 5), dtype=np.float32))

    def test_render_and_to_numpy_2d(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
    ):
        fs = FullscreenRenderer(gl_ctx)

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
        input_data = np.full((16, 16), 0.25, dtype=np.float32)
        fs.upload(input_data)
        fs.render_pass(prog, texture_uniform_name="tex_in")
        output = fs.to_numpy()

        assert output.shape == (16, 16)
        assert output.dtype == np.float32
        np.testing.assert_allclose(output, 0.5, atol=1e-5)

        prog.release()
        fs.release_fbo()

    def test_render_pass_custom_uniform_name_and_explicit_vao(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
    ):
        fs = FullscreenRenderer(gl_ctx)

        custom_frag_shader = """
            #version 330
            uniform sampler2D custom_sampler;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                fragColor = texture(custom_sampler, v_uv);
            }
        """
        prog = fs.create_program(vert_shader_src, custom_frag_shader)
        custom_vao = fs._create_vao(prog)

        input_data = np.full((8, 8, 4), 0.75, dtype=np.float32)
        fs.upload(input_data)
        fs.render_pass(prog, vao=custom_vao, texture_uniform_name="custom_sampler")
        output = fs.to_numpy()

        assert output.shape == (8, 8, 4)
        np.testing.assert_allclose(output, 0.75, atol=1e-5)

        custom_vao.release()
        prog.release()
        fs.release_fbo()

    def test_to_numpy_without_render_raises(self, gl_ctx: moderngl.Context):
        fs = FullscreenRenderer(gl_ctx)
        with pytest.raises(RuntimeError, match="No rendered data available"):
            fs.to_numpy()

    def test_swap_buffers_multipass(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
    ):
        fs = FullscreenRenderer(gl_ctx)

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

        input_data = np.zeros((8, 8, 3), dtype=np.float32)
        fs.upload(input_data)

        # Pass 1: 0.0 + 0.1 = 0.1
        fs.render_pass(prog)
        fs.swap_buffers()
        # Pass 2: 0.1 + 0.1 = 0.2
        fs.render_pass(prog)

        result = fs.to_numpy()
        assert result.shape == (8, 8, 3)
        np.testing.assert_allclose(result[..., 0], 0.2, atol=1e-5)
        np.testing.assert_allclose(result[..., 1], 0.2, atol=1e-5)
        np.testing.assert_allclose(result[..., 2], 0.2, atol=1e-5)

        prog.release()
        fs.release_fbo()

    def test_apply_convenience_method(
        self,
        gl_ctx: moderngl.Context,
        vert_shader_src: str,
        invert_frag_shader_src: str,
    ):
        fs = FullscreenRenderer(gl_ctx)
        prog = fs.create_program(vert_shader_src, invert_frag_shader_src)

        input_data = np.array([[[0.2, 0.4, 0.6, 1.0]]], dtype=np.float32)
        output = fs.apply(prog, input_data)

        assert output.shape == (1, 1, 4)
        np.testing.assert_allclose(output[0, 0, 0], 0.8, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 1], 0.6, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 2], 0.4, atol=1e-5)
        np.testing.assert_allclose(output[0, 0, 3], 1.0, atol=1e-5)

        prog.release()
        fs.release_fbo()

    def test_release_fbo_idempotent(self, gl_ctx: moderngl.Context):
        fs = FullscreenRenderer(gl_ctx)
        fs.upload(np.ones((8, 8), dtype=np.float32))

        assert fs._tex_in is not None
        assert fs._tex_out is not None
        assert fs._fbo is not None

        fs.release_fbo()
        assert fs._tex_in is None
        assert fs._tex_out is None
        assert fs._fbo is None
        assert fs._current_shape == (0, 0, 0)

        # Calling again is clean no-op
        fs.release_fbo()

    def test_close_alias(self, gl_ctx: moderngl.Context):
        fs = FullscreenRenderer(gl_ctx)
        fs.upload(np.ones((8, 8), dtype=np.float32))
        fs.close()
        assert fs._fbo is None

    def test_context_ownership_and_release(self):
        mock_ctx = MagicMock(spec=moderngl.Context)
        fs_borrowed = FullscreenRenderer(mock_ctx)
        assert not fs_borrowed._own_ctx
        fs_borrowed.release()
        mock_ctx.release.assert_not_called()
        assert fs_borrowed.ctx is mock_ctx

        with patch("spatialhub.utils.moderngl.fullscreen.create_moderngl_context", return_value=mock_ctx):
            with FullscreenRenderer() as fs_owned:
                assert fs_owned._own_ctx
            mock_ctx.release.assert_called_once()
            assert fs_owned.ctx is None
