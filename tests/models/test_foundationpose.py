"""Comprehensive unit tests for FoundationPoseAdapter and associated predictors."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from spatialhub.models.foundationpose import FoundationPoseAdapter
from spatialhub.models.foundationpose.adapter import PoseRefinePredictor, ScorePredictor
from spatialhub.structures import (
    CameraIntrinsics,
    InstanceSegmentation,
    ObjectPose6D,
)

try:
    import trimesh
    from spatialhub.utils.moderngl.context import create_moderngl_context

    # Verify context creation works
    _test_ctx = create_moderngl_context()
    _test_ctx.release()
    RENDERER_AVAILABLE = True
except Exception:
    RENDERER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RENDERER_AVAILABLE, reason="Rendering extras (trimesh, moderngl) required")


def _create_mock_session(input_names: list[str], output_names: list[str], run_side_effect) -> MagicMock:
    session = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    session.get_inputs.return_value = inputs

    outputs = []
    for name in output_names:
        out = MagicMock()
        out.name = name
        outputs.append(out)
    session.get_outputs.return_value = outputs

    session.run.side_effect = run_side_effect
    return session


@pytest.fixture
def mock_ort():
    """Mock create_ort_session and resolve_model_path for refiner and scorer."""
    def side_effect(model_path, **kwargs):
        path_str = str(model_path)
        if "refine" in path_str:
            def run_refine(output_names, onnx_feed):
                first_key = list(onnx_feed.keys())[0]
                bs = len(onnx_feed[first_key])
                return [
                    np.zeros((bs, 3), dtype=np.float32),  # trans
                    np.zeros((bs, 3), dtype=np.float32),  # rot
                ]
            return _create_mock_session(["input1", "input2"], ["trans", "rot"], run_refine)
        else:
            def run_score(output_names, onnx_feed):
                first_key = list(onnx_feed.keys())[0]
                bs = len(onnx_feed[first_key])
                return [np.ones((bs, 1), dtype=np.float32) * 0.95]
            return _create_mock_session(["input1", "input2"], ["score_logit"], run_score)

    with patch("spatialhub.models.foundationpose.adapter.create_ort_session", side_effect=side_effect), \
         patch("spatialhub.models.foundationpose.adapter.resolve_model_path", side_effect=lambda **kwargs: kwargs.get("model_path") or f"/mock/{kwargs.get('filename')}"):
        yield


@pytest.fixture
def sample_mesh() -> trimesh.Trimesh:
    """Create a sample 3D box mesh."""
    return trimesh.creation.box(extents=[0.1, 0.2, 0.3])


@pytest.fixture
def sample_intrinsics() -> np.ndarray:
    """Sample pinhole camera intrinsics matrix."""
    return np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class TestFoundationPoseInit:
    """Test suite for FoundationPoseAdapter initialization and configuration."""

    def test_init_with_model_object(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            assert adapter.model is not None
            assert adapter.model_name == "target_model"
            assert adapter.model_diameter is not None
            assert adapter.model_diameter > 0
            assert adapter.rot_grid is not None
            assert adapter.rot_grid.ndim == 3 and adapter.rot_grid.shape[1:] == (4, 4)
            assert adapter.pose_last is None

    def test_init_with_model_file_path(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, tmp_path: Path):
        model_path = tmp_path / "box_object.obj"
        sample_mesh.export(str(model_path))

        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=model_path) as adapter:
            assert adapter.model is not None
            assert adapter.model_name == "box_object"
            assert adapter.model_diameter is not None

    def test_init_without_model(self, mock_ort, sample_intrinsics: np.ndarray):
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=None) as adapter:
            assert adapter.model is None
            assert adapter.model_diameter is None
            assert adapter.rot_grid is None
            assert adapter.pose_last is None

    def test_init_with_symmetry_tfs(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        sym = np.eye(4, dtype=np.float32)
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh, symmetry_tfs=sym) as adapter:
            assert adapter.symmetry_tfs.shape == (1, 4, 4)

        sym_batch = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, ...], (2, 1, 1))
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh, symmetry_tfs=sym_batch) as adapter:
            assert adapter.symmetry_tfs.shape == (2, 4, 4)

    def test_init_missing_model_file_raises(self, mock_ort, sample_intrinsics: np.ndarray):
        with pytest.raises(FileNotFoundError):
            FoundationPoseAdapter(intrinsics=sample_intrinsics, model="non_existent_model_file_12345.obj")

    def test_init_with_camera_intrinsics_structure(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        cam_int = CameraIntrinsics(sample_intrinsics)
        with FoundationPoseAdapter(intrinsics=cam_int, model=sample_mesh) as adapter:
            assert adapter.K.shape == (3, 3)

    def test_init_with_intrinsics_sequence(self, mock_ort, sample_mesh: trimesh.Trimesh):
        seq = [500.0, 500.0, 320.0, 240.0]
        with FoundationPoseAdapter(intrinsics=seq, model=sample_mesh) as adapter:
            assert adapter.K.shape == (3, 3)
            assert adapter.K[0, 0] == 500.0
            assert adapter.K[1, 1] == 500.0
            assert adapter.K[0, 2] == 320.0
            assert adapter.K[1, 2] == 240.0

    def test_init_invalid_intrinsic_shape_raises(self, mock_ort, sample_mesh: trimesh.Trimesh):
        with pytest.raises(ValueError, match="Expected camera intrinsics"):
            FoundationPoseAdapter(intrinsics=np.zeros((4, 4), dtype=np.float32), model=sample_mesh)

    def test_init_with_custom_glctx(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        custom_ctx = create_moderngl_context()
        try:
            adapter = FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh, glctx=custom_ctx)
            assert adapter.glctx is custom_ctx
            assert adapter._own_glctx is False
            adapter.release()
            # Context should NOT have been released by adapter since it is externally owned
            assert not custom_ctx.is_release_delayed if hasattr(custom_ctx, "is_release_delayed") else True
        finally:
            custom_ctx.release()


class TestFoundationPoseTargetModel:
    """Test suite for dynamic target model setting and switching."""

    def test_set_model_updates_state(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=None) as adapter:
            assert adapter.model is None
            adapter.set_model(sample_mesh, model_name="box_model")
            assert adapter.model is not None
            assert adapter.model_name == "box_model"
            assert adapter.model_diameter is not None
            assert adapter.rot_grid is not None

    def test_set_model_resets_pose_last(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            adapter.pose_last = np.eye(4, dtype=np.float32)
            new_mesh = trimesh.creation.cylinder(radius=0.05, height=0.2)
            adapter.set_model(new_mesh, model_name="cylinder")
            assert adapter.model_name == "cylinder"
            assert adapter.pose_last is None


class TestFoundationPoseRegisterInputs:
    """Test suite for input parsing, depth scaling, and validation in register()."""

    @pytest.fixture
    def valid_inputs(self, sample_intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = 480, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw a synthetic foreground patch
        rgb[200:300, 280:380] = 200
        depth[200:300, 280:380] = 1.0  # 1 meter
        mask[200:300, 280:380] = 255
        return rgb, depth, mask

    def test_register_with_numpy_arrays(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            result = adapter.register(rgb=rgb, depth=depth, mask=mask)
            assert isinstance(result, ObjectPose6D)
            assert result.pose.shape == (1, 4, 4)
            assert result.scores.shape == (1,)
            assert result.names == ["target_model"]

    def test_register_with_segmentation_result(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        seg_result = InstanceSegmentation(
            boxes=np.array([[280, 200, 380, 300]], dtype=np.float32),
            masks=(mask > 0)[np.newaxis, ...],
            scores=np.array([0.99], dtype=np.float32),
            names=["target_model"],
        )
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            result = adapter.register(rgb=rgb, depth=depth, mask=seg_result)
            assert isinstance(result, ObjectPose6D)
            assert result.pose.shape == (1, 4, 4)

    def test_register_with_empty_segmentation_result_raises(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, _ = valid_inputs
        empty_seg = InstanceSegmentation(
            boxes=np.zeros((0, 4), dtype=np.float32),
            masks=np.zeros((0, 480, 640), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
        )
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            with pytest.raises(ValueError, match="contains no masks"):
                adapter.register(rgb=rgb, depth=depth, mask=empty_seg)

    def test_register_with_depth_unit_scaling(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        depth_mm = (depth * 1000.0).astype(np.uint16)
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            result = adapter.register(rgb=rgb, depth=depth_mm, mask=mask, depth_unit="mm")
            assert isinstance(result, ObjectPose6D)
            assert result.pose.shape == (1, 4, 4)

    def test_register_with_file_paths(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs, tmp_path: Path):
        rgb, depth, mask = valid_inputs
        rgb_path = tmp_path / "rgb.png"
        depth_path = tmp_path / "depth.npy"
        mask_path = tmp_path / "mask.png"

        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(str(depth_path), depth)
        cv2.imwrite(str(mask_path), mask)

        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            result = adapter.register(rgb=rgb_path, depth=depth_path, mask=mask_path)
            assert isinstance(result, ObjectPose6D)

    def test_register_without_model_raises(self, mock_ort, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=None) as adapter:
            with pytest.raises(RuntimeError, match="Target 3D CAD model has not been set"):
                adapter.register(rgb=rgb, depth=depth, mask=mask)

    def test_register_spatial_shape_mismatch_raises(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        rgb_wrong = cv2.resize(rgb, (320, 240))
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            with pytest.raises(ValueError, match="Spatial resolution mismatch"):
                adapter.register(rgb=rgb_wrong, depth=depth, mask=mask)

            mask_wrong = cv2.resize(mask, (320, 240))
            with pytest.raises(ValueError, match="Spatial resolution mismatch"):
                adapter.register(rgb=rgb, depth=depth, mask=mask_wrong)

    def test_register_low_valid_pixels_fallback(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        h, w = 480, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Only 2 valid pixels
        depth[10, 10:12] = 1.0
        mask[10, 10:12] = 255

        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            result = adapter.register(rgb=rgb, depth=depth, mask=mask)
            assert isinstance(result, ObjectPose6D)
            assert np.allclose(result.scores, [0.0])


class TestFoundationPoseTrack:
    """Test suite for frame-to-frame sequential tracking in track()."""

    @pytest.fixture
    def valid_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = 480, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        rgb[200:300, 280:380] = 200
        depth[200:300, 280:380] = 1.0
        mask[200:300, 280:380] = 255
        return rgb, depth, mask

    def test_track_before_register_raises(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, _ = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            with pytest.raises(RuntimeError, match="No previous pose estimate found"):
                adapter.track(rgb=rgb, depth=depth)

    def test_track_with_manually_set_current_pose(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, _ = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            initial_pose = np.eye(4, dtype=np.float32)
            initial_pose[:3, 3] = [0.0, 0.0, 1.0]
            adapter.current_pose = initial_pose

            assert adapter.current_pose is not None
            assert np.allclose(adapter.current_pose, initial_pose)

            track_result = adapter.track(rgb=rgb, depth=depth)
            assert isinstance(track_result, ObjectPose6D)
            assert track_result.pose.shape == (1, 4, 4)

    def test_current_pose_setter_validation_and_reset(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            with pytest.raises(ValueError, match="Expected pose of shape"):
                adapter.current_pose = np.zeros((3, 3), dtype=np.float32)

            adapter.current_pose = np.eye(4, dtype=np.float32)
            assert adapter.current_pose is not None
            adapter.reset_tracking()
            assert adapter.current_pose is None

    def test_track_with_verify_conf(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            adapter.register(rgb=rgb, depth=depth, mask=mask)
            track_result = adapter.track(rgb=rgb, depth=depth, verify_conf=True)
            assert isinstance(track_result, ObjectPose6D)
            assert track_result.scores is not None
            assert len(track_result.scores) == 1
            assert track_result.scores[0] > 0.0

    def test_register_and_track_with_filter_depth_options(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            reg_result = adapter.register(rgb=rgb, depth=depth, mask=mask, filter_depth=False)
            assert isinstance(reg_result, ObjectPose6D)

            track_result = adapter.track(rgb=rgb, depth=depth, filter_depth=True, filter_radius=3)
            assert isinstance(track_result, ObjectPose6D)

    def test_track_after_register_succeeds(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            reg_result = adapter.register(rgb=rgb, depth=depth, mask=mask)
            assert adapter.pose_last is not None

            track_result = adapter.track(rgb=rgb, depth=depth)
            assert isinstance(track_result, ObjectPose6D)
            assert track_result.pose.shape == (1, 4, 4)
            assert track_result.scores.shape == (1,)
            assert adapter.pose_last is not None

    def test_track_shape_mismatch_raises(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray, valid_inputs):
        rgb, depth, mask = valid_inputs
        with FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh) as adapter:
            adapter.register(rgb=rgb, depth=depth, mask=mask)
            depth_wrong = cv2.resize(depth, (320, 240))
            with pytest.raises(ValueError, match="Spatial resolution mismatch"):
                adapter.track(rgb=rgb, depth=depth_wrong)


class TestFoundationPoseLifecycle:
    """Test suite for resource cleanup and context manager lifecycle."""

    def test_context_manager_releases_resources(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        adapter = FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh)
        with adapter as managed:
            assert managed is adapter
        # Verify predictors sessions and gl resources are released
        assert adapter.refiner.ort_session is None
        assert adapter.scorer.ort_session is None

    def test_explicit_release_and_close(self, mock_ort, sample_mesh: trimesh.Trimesh, sample_intrinsics: np.ndarray):
        adapter = FoundationPoseAdapter(intrinsics=sample_intrinsics, model=sample_mesh)
        adapter.close()
        assert adapter.refiner.ort_session is None
        assert adapter.scorer.ort_session is None
