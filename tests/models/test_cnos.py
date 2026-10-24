"""Unit tests for CNOSAdapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub.models.cnos import CNOSAdapter
from spatialhub.structures import (
    CameraExtrinsics,
    CameraIntrinsics,
    FeatureEmbedding,
    ImageBuffers,
    InstanceSegmentation,
)


class TestCNOSInit:
    """Test suite for CNOSAdapter initialization and factory constructors."""

    @pytest.fixture
    def mock_descriptor(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_segmentor(self) -> MagicMock:
        return MagicMock()

    def test_init_with_reference_features_array(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        feats = np.random.randn(42, 1024).astype(np.float32)
        adapter = CNOSAdapter(
            reference_features=feats,
            descriptor=mock_descriptor,
            segmentor=mock_segmentor,
            object_name="test_obj",
        )
        assert adapter.reference_features is not None
        assert adapter.reference_features.shape == (42, 1024)
        assert adapter.object_name == "test_obj"

        norms = np.linalg.norm(adapter.reference_features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_init_with_feature_embedding(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        raw_feats = np.random.randn(10, 1024).astype(np.float32)
        norm_feats = raw_feats / np.linalg.norm(raw_feats, axis=-1, keepdims=True)
        feat_result = FeatureEmbedding(
            features=norm_feats,
            embedding_type="global",
            l2_normalized=True,
        )

        adapter = CNOSAdapter(
            reference_features=feat_result,
            descriptor=mock_descriptor,
            segmentor=mock_segmentor,
        )
        assert adapter.reference_features is not None
        assert adapter.reference_features.shape == (10, 1024)

    def test_dimension_mismatch_raises_on_inference(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        wrong_dim_feats = np.random.randn(10, 512).astype(np.float32)
        adapter = CNOSAdapter(
            reference_features=wrong_dim_feats,
            descriptor=mock_descriptor,
            segmentor=mock_segmentor,
        )
        mock_segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
            masks=np.ones((1, 100, 100), dtype=bool),
            scores=np.array([0.95], dtype=np.float32),
        )
        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )
        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            adapter.inference(scene_img)

    def test_invalid_reference_features_shape_raises(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        wrong_shape = np.random.randn(1024).astype(np.float32)
        with pytest.raises(ValueError, match="Reference features must have shape"):
            CNOSAdapter(
                reference_features=wrong_shape,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

    def test_empty_reference_features_raises(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        empty_features = np.empty((0, 512), dtype=np.float32)
        with pytest.raises(ValueError, match="Reference features must not be empty"):
            CNOSAdapter(
                reference_features=empty_features,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

    def test_from_templates_with_image_buffers(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        rgba = np.random.randint(0, 255, (5, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(
            rgba=rgba,
        )

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(5, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        adapter = CNOSAdapter.from_templates(
            templates=render_res,
            descriptor=mock_descriptor,
            segmentor=mock_segmentor,
            object_name="box",
        )
        assert adapter.reference_features is not None
        assert adapter.reference_features.shape == (5, 1024)
        assert adapter.object_name == "box"

        called_crops = mock_descriptor.extract_features.call_args[0][0]
        for crop in called_crops:
            assert crop.ndim == 3 and crop.shape[2] == 3

    def test_from_templates_empty_image_buffers_raises(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        render_res = ImageBuffers(
            depth=np.zeros((1, 64, 64), dtype=np.float32),
            rgba=None,
        )
        with pytest.raises(ValueError, match="contains no image data"):
            CNOSAdapter.from_templates(
                templates=render_res,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

    def test_from_templates_with_unsegmented_rgb_logs_warning(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        # 3-channel image where all pixels are non-zero (full-frame bbox)
        full_rgb = np.ones((64, 64, 3), dtype=np.uint8) * 128
        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        with patch("spatialhub.models.cnos.adapter.logger.warning") as mock_warn:
            CNOSAdapter.from_templates(
                templates=[full_rgb],
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )
            mock_warn.assert_called_once()
            assert "3 channels without alpha" in mock_warn.call_args[0][0]

    def test_from_templates_invalid_type_raises(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        with pytest.raises(TypeError, match="Unsupported templates input type"):
            CNOSAdapter.from_templates(
                templates=12345,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

    def test_from_templates_with_tuple(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )
        adapter = CNOSAdapter.from_templates(
            templates=(img,),
            descriptor=mock_descriptor,
            segmentor=mock_segmentor,
        )
        assert adapter.reference_features.shape == (1, 1024)

    def test_from_cad_model(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        mock_renderer = MagicMock()
        mock_renderer.model = Path("fake_model.ply")
        mock_renderer.model_unit = "mm"

        rgba = np.random.randint(0, 255, (2, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(
            rgba=rgba,
        )
        mock_renderer.render_templates.return_value = render_res

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(2, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            adapter = CNOSAdapter.from_cad_model(
                model="fake_model.ply",
                renderer=mock_renderer,
                radius=0.75,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

        assert adapter.reference_features is not None
        assert adapter.object_name == "fake_model"
        mock_renderer.render_templates.assert_called_once()
        assert mock_renderer.render_templates.call_args[1]["radius"] == 0.75

    def test_from_cad_model_with_camera_structures(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        mock_renderer = MagicMock()
        mock_renderer.model = Path("mesh.obj")
        mock_renderer.model_unit = "m"

        rgba = np.random.randint(0, 255, (2, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(rgba=rgba)
        mock_renderer.render_templates.return_value = render_res

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(2, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        extrinsics = CameraExtrinsics(matrix=np.eye(4, dtype=np.float32))
        intrinsics = CameraIntrinsics(matrix=np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32))

        with patch("pathlib.Path.exists", return_value=True):
            adapter = CNOSAdapter.from_cad_model(
                model="mesh.obj",
                renderer=mock_renderer,
                poses=extrinsics,
                intrinsics=intrinsics,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

        assert adapter.reference_features is not None
        assert adapter.object_name == "mesh"
        mock_renderer.render_templates.assert_called_once()

    def test_from_cad_model_with_custom_poses_and_radius(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        mock_renderer = MagicMock()
        mock_renderer.model = Path("obj.ply")
        mock_renderer.model_unit = "m"

        rgba = np.zeros((1, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(
            rgba=rgba,
        )
        mock_renderer.render_templates.return_value = render_res

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        unit_poses = np.eye(4, dtype=np.float32)[np.newaxis, ...]
        unit_poses[0, :3, 3] = [0.0, 0.0, 1.0]

        with patch("pathlib.Path.exists", return_value=True):
            CNOSAdapter.from_cad_model(
                model="obj.ply",
                poses=unit_poses,
                radius=0.6,
                renderer=mock_renderer,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

        mock_renderer.render_templates.assert_called_once()
        assert mock_renderer.render_templates.call_args[1]["radius"] == 0.6
        np.testing.assert_allclose(mock_renderer.render_templates.call_args[1]["poses"], unit_poses)

    def test_from_cad_model_forwards_num_viewpoints(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        mock_renderer = MagicMock()
        mock_renderer.model = Path("obj.ply")
        mock_renderer.model_unit = "m"

        rgba = np.zeros((1, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(
            rgba=rgba,
        )
        mock_renderer.render_templates.return_value = render_res

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            CNOSAdapter.from_cad_model(
                model="obj.ply",
                num_viewpoints=162,
                renderer=mock_renderer,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

        mock_renderer.render_templates.assert_called_once()
        assert mock_renderer.render_templates.call_args[1]["num_viewpoints"] == 162

    def test_from_cad_model_forwards_intrinsics(self, mock_descriptor: MagicMock, mock_segmentor: MagicMock):
        mock_renderer = MagicMock()
        mock_renderer.model = Path("obj.ply")
        mock_renderer.model_unit = "m"

        rgba = np.zeros((1, 64, 64, 4), dtype=np.uint8)
        render_res = ImageBuffers(
            rgba=rgba,
        )
        mock_renderer.render_templates.return_value = render_res

        mock_descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.random.randn(1, 1024).astype(np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        custom_k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        with patch("pathlib.Path.exists", return_value=True):
            CNOSAdapter.from_cad_model(
                model="obj.ply",
                intrinsics=custom_k,
                renderer=mock_renderer,
                descriptor=mock_descriptor,
                segmentor=mock_segmentor,
            )

        mock_renderer.render_templates.assert_called_once()
        np.testing.assert_allclose(mock_renderer.render_templates.call_args[1]["intrinsics"], custom_k)



class TestCNOSInference:
    """Test suite for CNOSAdapter inference pipeline and matching logic."""

    @pytest.fixture
    def mock_adapter(self) -> CNOSAdapter:
        descriptor = MagicMock()
        descriptor.output_dim = 128

        segmentor = MagicMock()

        ref_features = np.zeros((4, 128), dtype=np.float32)
        ref_features[0, 0] = 1.0

        adapter = CNOSAdapter(
            reference_features=ref_features,
            descriptor=descriptor,
            segmentor=segmentor,
            object_name="target_item",
        )
        return adapter

    def test_inference_with_matching_proposals(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=np.float32)
        mock_masks = np.zeros((2, 100, 100), dtype=bool)
        mock_masks[0, 10:50, 10:50] = True
        mock_masks[1, 60:90, 60:90] = True
        mock_scores = np.array([0.95, 0.90], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        scene_feats = np.zeros((2, 128), dtype=np.float32)
        scene_feats[0, 0] = 1.0
        scene_feats[1, 1] = 1.0

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=scene_feats,
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mock_adapter.inference(scene_img, conf_threshold=0.15)

        assert isinstance(result, InstanceSegmentation)
        assert len(result.boxes) == 1
        assert len(result.masks) == 1
        assert len(result.scores) == 1
        assert result.names == ["target_item"]
        np.testing.assert_allclose(result.boxes[0], [10, 10, 50, 50])

    def test_inference_no_proposals_returns_empty(self, mock_adapter: CNOSAdapter):
        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=np.empty((0, 4), dtype=np.float32),
            masks=np.empty((0, 100, 100), dtype=bool),
            scores=np.empty((0,), dtype=np.float32),
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mock_adapter.inference(scene_img)

        assert len(result.boxes) == 0
        assert len(result.masks) == 0
        assert len(result.scores) == 0

    def test_dynamic_target_switching(self, mock_adapter: CNOSAdapter):
        new_feats = np.random.randn(8, 128).astype(np.float32)
        mock_adapter.set_reference_features(new_feats, object_name="second_obj")

        assert mock_adapter.reference_features.shape == (8, 128)
        assert mock_adapter.object_name == "second_obj"

    def test_inference_filters_degenerate_boxes(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 10, 50], [20, 20, 50, 50]], dtype=np.float32)
        mock_masks = np.zeros((2, 100, 100), dtype=bool)
        mock_scores = np.array([0.9, 0.8], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.ones((1, 128), dtype=np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mock_adapter.inference(scene_img, conf_threshold=0.0)

        # only 1 non-degenerate box should pass to descriptor and output
        assert len(result.boxes) == 1
        np.testing.assert_allclose(result.boxes[0], [20, 20, 50, 50])
        mock_adapter.descriptor.extract_features.assert_called_once()
        crops_passed = mock_adapter.descriptor.extract_features.call_args[0][0]
        assert len(crops_passed) == 1

    def test_inference_sorts_detections_by_score(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 40, 40], [50, 50, 80, 80]], dtype=np.float32)
        mock_masks = np.zeros((2, 100, 100), dtype=bool)
        mock_scores = np.array([0.9, 0.8], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        # first proposal gets lower similarity, second proposal gets higher similarity
        feats = np.zeros((2, 128), dtype=np.float32)
        feats[0, 0] = 0.3
        feats[1, 0] = 0.9

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=feats,
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mock_adapter.inference(scene_img, num_max_dets=5, conf_threshold=0.05)

        assert len(result.boxes) == 2
        assert result.scores[0] > result.scores[1]
        np.testing.assert_allclose(result.boxes[0], [50, 50, 80, 80])
        np.testing.assert_allclose(result.boxes[1], [10, 10, 40, 40])

    def test_inference_forwards_proposal_thresholds(self, mock_adapter: CNOSAdapter):
        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=np.empty((0, 4), dtype=np.float32),
            masks=np.empty((0, 100, 100), dtype=bool),
            scores=np.empty((0,), dtype=np.float32),
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_adapter.inference(
            scene_img,
            proposal_conf_threshold=0.12,
            proposal_iou_threshold=0.85,
        )

        mock_adapter.segmentor.generate_masks.assert_called_once()
        call_args, call_kwargs = mock_adapter.segmentor.generate_masks.call_args
        np.testing.assert_array_equal(call_args[0], scene_img)
        assert call_kwargs == {"conf_threshold": 0.12, "iou_threshold": 0.85}

    def test_inference_background_masking(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        mock_masks = np.zeros((1, 100, 100), dtype=bool)
        mock_masks[0, 20:40, 20:40] = True
        mock_scores = np.array([0.95], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.ones((1, 128), dtype=np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_adapter.inference(scene_img)

        passed_crop = mock_adapter.descriptor.extract_features.call_args[0][0][0]
        # foreground region (20:40 relative to 10:50 is 10:30) should be non-zero
        assert np.all(passed_crop[10:30, 10:30] == 255)
        # background region outside mask should be 0
        assert np.all(passed_crop[0:10, :] == 0)

    def test_inference_with_top_k_override(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        mock_masks = np.ones((1, 100, 100), dtype=bool)
        mock_scores = np.array([0.95], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        # 4 reference features: 1.0, 0.8, 0.4, 0.2
        mock_adapter.reference_features = np.array(
            [[1.0, 0.0], [0.8, 0.0], [0.4, 0.0], [0.2, 0.0]], dtype=np.float32
        )

        # scene feature: [1.0, 0.0]
        scene_feat = np.array([[1.0, 0.0]], dtype=np.float32)
        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=scene_feat,
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # top_k=1 -> mean of top 1 score: 1.0
        res_k1 = mock_adapter.inference(scene_img, top_k=1, conf_threshold=0.0)
        assert np.isclose(res_k1.scores[0], 1.0)

        # top_k=2 -> mean of top 2 scores: (1.0 + 0.8) / 2 = 0.9
        res_k2 = mock_adapter.inference(scene_img, top_k=2, conf_threshold=0.0)
        assert np.isclose(res_k2.scores[0], 0.9)

    def test_inference_with_nms_deduplication(self, mock_adapter: CNOSAdapter):
        # two overlapping boxes (IoU ~0.8)
        mock_boxes = np.array([[10, 10, 50, 50], [12, 12, 50, 50]], dtype=np.float32)
        mock_masks = np.ones((2, 100, 100), dtype=bool)
        mock_scores = np.array([0.9, 0.8], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        feats = np.zeros((2, 128), dtype=np.float32)
        feats[0, 0] = 0.95
        feats[1, 0] = 0.90

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=feats,
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # without NMS, both proposals survive
        res_no_nms = mock_adapter.inference(scene_img, top_k=1, conf_threshold=0.5)
        assert len(res_no_nms.boxes) == 2

        # with NMS (iou_threshold=0.5), duplicate proposal is suppressed
        res_nms = mock_adapter.inference(scene_img, top_k=1, conf_threshold=0.5, nms_iou_threshold=0.5)
        assert len(res_nms.boxes) == 1
        np.testing.assert_allclose(res_nms.boxes[0], [10, 10, 50, 50])

    def test_generate_proposals_classmethod(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        mock_masks = np.ones((1, 100, 100), dtype=bool)
        mock_scores = np.array([0.95], dtype=np.float32)

        mock_adapter.segmentor.generate_masks.return_value = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        proposals = CNOSAdapter.generate_proposals(
            scene_img,
            segmentor=mock_adapter.segmentor,
            proposal_conf_threshold=0.08,
            proposal_iou_threshold=0.88,
        )

        assert isinstance(proposals, InstanceSegmentation)
        assert len(proposals.boxes) == 1
        mock_adapter.segmentor.generate_masks.assert_called_once()
        call_args, call_kwargs = mock_adapter.segmentor.generate_masks.call_args
        np.testing.assert_array_equal(call_args[0], scene_img)
        assert call_kwargs == {"conf_threshold": 0.08, "iou_threshold": 0.88}

    def test_inference_with_precomputed_proposals(self, mock_adapter: CNOSAdapter):
        mock_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        mock_masks = np.ones((1, 100, 100), dtype=bool)
        mock_scores = np.array([0.95], dtype=np.float32)

        precomputed_proposals = InstanceSegmentation(
            boxes=mock_boxes,
            masks=mock_masks,
            scores=mock_scores,
        )

        mock_adapter.descriptor.extract_features.return_value = FeatureEmbedding(
            features=np.ones((1, 128), dtype=np.float32),
            embedding_type="global",
            l2_normalized=True,
        )

        scene_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mock_adapter.inference(
            scene_img,
            proposals=precomputed_proposals,
            conf_threshold=0.0,
        )

        assert len(result.boxes) == 1
        # segmentor.generate_masks should NOT have been called
        mock_adapter.segmentor.generate_masks.assert_not_called()


