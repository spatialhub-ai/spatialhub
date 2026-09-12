"""Comprehensive unit tests for spatialhub.utils.template_renderer module."""

from pathlib import Path
import numpy as np
import pytest

try:
    import trimesh
    from spatialhub.utils.moderngl.context import create_moderngl_context
    from spatialhub.utils.template_renderer import TemplateRenderer
    RENDERER_AVAILABLE = True
except Exception:
    RENDERER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RENDERER_AVAILABLE, reason="Rendering extras (trimesh, moderngl) required")


class TestTemplateRenderer:
    """Test suite for TemplateRenderer offscreen CAD rendering."""

    @pytest.fixture
    def sample_box_mesh(self) -> "trimesh.Trimesh":
        return trimesh.creation.box(extents=[0.1, 0.2, 0.3])

    def test_template_renderer_initialization(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model_path=sample_box_mesh, ctx=ctx)
        try:
            assert renderer.mesh is not None
            assert renderer.prog is not None
            assert renderer.pos_buffer is not None
        finally:
            renderer.release()

    def test_template_renderer_render_templates(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model_path=sample_box_mesh, ctx=ctx)
        try:
            results = renderer.render_templates(
                width=64,
                height=64,
                intrinsics=[50.0, 50.0, 32.0, 32.0],
                num_viewpoints=4,
                radius=0.5,
            )

            assert len(results) == 4
            for res in results:
                assert "rgba" in res and "depth" in res
                assert res["rgba"].shape == (64, 64, 4)
                assert res["rgba"].dtype == np.uint8
                assert res["depth"].shape == (64, 64)
                assert res["depth"].dtype == np.float32

                # Should have non-empty rendered pixels
                assert np.any(res["rgba"][..., 3] > 0)
                assert np.any(res["depth"] > 0)
        finally:
            renderer.release()

    def test_template_renderer_invalid_intrinsics_raises(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model_path=sample_box_mesh, ctx=ctx)
        try:
            with pytest.raises(ValueError, match="Intrinsics must be a 3x3 camera matrix"):
                renderer.render_templates(width=64, height=64, intrinsics=[50.0, 50.0])
        finally:
            renderer.release()

    def test_template_renderer_save_and_cleanup(self, sample_box_mesh: "trimesh.Trimesh", tmp_path: Path):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model_path=sample_box_mesh, ctx=ctx)
        try:
            results = renderer.render_templates(
                width=32,
                height=32,
                intrinsics=[30.0, 30.0, 16.0, 16.0],
                num_viewpoints=2,
                radius=0.4,
            )

            rgba_paths, depth_paths = renderer.save(
                results,
                output_dir=tmp_path,
                save_scene=True,
                save_depth=True,
            )

            assert len(rgba_paths) == 2
            assert len(depth_paths) == 2
            assert all(p.exists() for p in rgba_paths)
            assert all(p.exists() for p in depth_paths)
        finally:
            renderer.release()
