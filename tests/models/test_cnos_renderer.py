"""Comprehensive unit tests for spatialhub.models.cnos.renderer module."""

from pathlib import Path
import numpy as np
import pytest

try:
    import trimesh
    from spatialhub.structures import CameraExtrinsics, CameraIntrinsics, ImageBuffers
    from spatialhub.utils.camera import sample_sphere_poses
    from spatialhub.utils.moderngl.context import create_moderngl_context
    from spatialhub.models.cnos.renderer import TemplateRenderer, parse_poses
    RENDERER_AVAILABLE = True
except Exception:
    RENDERER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RENDERER_AVAILABLE, reason="Rendering extras (trimesh, moderngl) required")


class TestTemplateRenderer:
    """Test suite for CNOS TemplateRenderer offscreen CAD rendering."""

    @pytest.fixture
    def sample_box_mesh(self) -> "trimesh.Trimesh":
        return trimesh.creation.box(extents=[0.1, 0.2, 0.3])

    def test_template_renderer_initialization(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model=sample_box_mesh, ctx=ctx)
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

        renderer = TemplateRenderer(model=sample_box_mesh, ctx=ctx)
        try:
            results = renderer.render_templates(
                width=64,
                height=64,
                intrinsics=[50.0, 50.0, 32.0, 32.0],
                radius=0.5,
            )

            assert len(results) == 42
            assert results.rgba is not None and results.depth is not None
            assert results.rgba.shape == (42, 64, 64, 4)
            assert results.rgba.dtype == np.uint8
            assert results.depth.shape == (42, 64, 64)
            assert results.depth.dtype == np.float32

            for res in results:
                assert res.rgba is not None and res.depth is not None
                assert res.rgba.shape == (1, 64, 64, 4)
                assert res.rgba.dtype == np.uint8
                assert res.depth.shape == (1, 64, 64)
                assert res.depth.dtype == np.float32
                assert np.any(res.rgba[..., 3] > 0)
                assert np.any(res.depth > 0)
        finally:
            renderer.release()

    def test_template_renderer_invalid_intrinsics_raises(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model=sample_box_mesh, ctx=ctx)
        try:
            with pytest.raises(ValueError, match="Intrinsics must be a 3x3 camera matrix"):
                renderer.render_templates(width=64, height=64, intrinsics=[50.0, 50.0])
        finally:
            renderer.release()

    def test_parse_poses_helper(self):
        unit_poses = np.eye(4, dtype=np.float32)[np.newaxis, ...]
        unit_poses[0, :3, 3] = [100.0, 200.0, 300.0]

        poses_m = parse_poses(unit_poses, pose_unit="mm")
        assert poses_m.shape == (1, 4, 4)
        np.testing.assert_allclose(poses_m[0, :3, 3], [0.1, 0.2, 0.3], rtol=1e-5)

        single_pose = np.eye(4, dtype=np.float32)
        poses_2d = parse_poses(single_pose)
        assert poses_2d.shape == (1, 4, 4)

        extrinsics = CameraExtrinsics(matrix=unit_poses)
        poses_ext = parse_poses(extrinsics, pose_unit="mm")
        assert poses_ext.shape == (1, 4, 4)
        np.testing.assert_allclose(poses_ext[0, :3, 3], [0.1, 0.2, 0.3], rtol=1e-5)

        pose_list = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]
        poses_list_parsed = parse_poses(pose_list)
        assert poses_list_parsed.shape == (2, 4, 4)

        pose_tuple = (np.eye(4, dtype=np.float32),)
        poses_tuple_parsed = parse_poses(pose_tuple)
        assert poses_tuple_parsed.shape == (1, 4, 4)

    def test_template_renderer_default_intrinsics_and_context_manager(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        with TemplateRenderer(model=sample_box_mesh, width=64, height=48, ctx=ctx) as renderer:
            assert renderer.width == 64
            assert renderer.height == 48
            assert renderer.znear == 0.001
            assert renderer.zfar == 100.0

            results = renderer.render_templates(radius=0.4)
            assert len(results) == 42
            assert results.rgba is not None and results.rgba.shape == (42, 48, 64, 4)
            assert results.depth is not None and results.depth.shape == (42, 48, 64)

    def test_template_renderer_intrinsics_matrix_3x3(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        k_mat = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        with TemplateRenderer(model=sample_box_mesh, width=64, height=48, intrinsics=k_mat, ctx=ctx) as renderer:
            results = renderer.render_templates(radius=0.4)
            assert len(results) == 42
            assert results.rgba is not None and results.rgba.shape == (42, 48, 64, 4)

    def test_template_renderer_sub_batch_chunking(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        poses = sample_sphere_poses(35, radius=0.4)

        with TemplateRenderer(
            model=sample_box_mesh,
            width=32,
            height=32,
            max_chunk_size=8,
            ctx=ctx,
        ) as renderer:
            results = renderer.render_templates(
                poses=poses,
                passes=("rgba", "depth", "normals", "masks"),
            )
            assert len(results) == 35
            assert results.rgba is not None and results.rgba.shape == (35, 32, 32, 4)
            assert results.depth is not None and results.depth.shape == (35, 32, 32)
            assert results.normal is not None and results.normal.shape == (35, 32, 32, 3)
            assert results.mask is not None and results.mask.shape == (35, 32, 32)
            assert results.mask.dtype == bool

    def test_template_renderer_unshaded_mode(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        with TemplateRenderer(model=sample_box_mesh, width=32, height=32, unshaded=True, ctx=ctx) as renderer:
            results = renderer.render_templates(radius=0.4, passes=("rgba", "depth"))
            assert len(results) == 42
            assert results.rgba is not None
            assert results.depth is not None
            assert results.normal is None
            assert results.mask is None

    def test_template_renderer_custom_num_viewpoints_and_center(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        with TemplateRenderer(model=sample_box_mesh, center=False, width=32, height=32, ctx=ctx) as renderer:
            assert renderer.center is False
            results = renderer.render_templates(num_viewpoints=16, radius=0.4)
            assert len(results) == 16
            assert results.rgba is not None and results.rgba.shape == (16, 32, 32, 4)

    def test_template_renderer_release_idempotent(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        renderer = TemplateRenderer(model=sample_box_mesh, ctx=ctx)
        renderer.release()
        assert renderer.ctx is None
        # Subsequent release should not fail
        renderer.release()

    def test_template_renderer_camera_structures(self, sample_box_mesh: "trimesh.Trimesh"):
        try:
            ctx = create_moderngl_context()
        except Exception as err:
            pytest.skip(f"Could not initialize ModernGL context: {err}")

        k_mat = np.array([[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics = CameraIntrinsics(matrix=k_mat)
        poses = sample_sphere_poses(4, radius=0.4)
        extrinsics = CameraExtrinsics(matrix=poses)

        with TemplateRenderer(model=sample_box_mesh, width=32, height=32, ctx=ctx) as renderer:
            results = renderer.render_templates(
                intrinsics=intrinsics,
                poses=extrinsics,
            )
            assert len(results) == 4
            assert results.rgba is not None and results.rgba.shape == (4, 32, 32, 4)

    def test_save_templates(self, tmp_path: Path):
        rgba = np.zeros((3, 32, 32, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        depth = np.ones((3, 32, 32), dtype=np.float32)
        mask = np.ones((3, 32, 32), dtype=bool)
        buffers = ImageBuffers(rgba=rgba, depth=depth, mask=mask)

        saved = TemplateRenderer.save_templates(
            buffers,
            tmp_path / "templates",
            passes=("rgba", "depth", "masks"),
            prefix="custom",
        )
        assert "rgba" in saved
        assert "depth" in saved
        assert "masks" in saved
        assert len(saved["rgba"]) == 3
        assert saved["rgba"][0].name == "custom_0000.png"
        assert saved["rgba"][0].exists()
        assert saved["depth"][0].name == "custom_0000.npy"
        assert saved["depth"][0].exists()
        assert saved["masks"][0].name == "custom_0000.png"
        assert saved["masks"][0].exists()
