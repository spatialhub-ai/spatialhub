"""Unit tests for SAMAdapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub.models.sam.adapter import (
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
    SAMAdapter,
    masks_to_boxes,
)
from spatialhub.structures import SegmentationResult


class TestSAMAdapterInit:
    """Test suite for SAM adapter initialization, preallocation, and resource management."""

    def test_init_defaults(self):
        """Test default parameter initialization and precomputed grids."""
        with patch("spatialhub.models.sam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.sam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/sam.onnx")
            mock_session.return_value = MagicMock()

            adapter = SAMAdapter(model_variant="vit_b")

            assert adapter.target_size == 1024
            assert adapter.points_per_side == 32
            assert adapter.points_per_batch == 64
            assert adapter.points_rel.shape == (1024, 2)
            assert adapter._cached_mask_input.shape == (64, 1, 256, 256)
            assert adapter._cached_has_mask_input.shape == (64,)
            assert adapter._cached_point_labels.shape == (64, 1)
            np.testing.assert_array_equal(adapter.pixel_mean, SAM_PIXEL_MEAN)
            np.testing.assert_array_equal(adapter.pixel_std, SAM_PIXEL_STD)

    def test_init_custom_target_size_and_points(self):
        """Test custom target size and grid density."""
        with patch("spatialhub.models.sam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.sam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/sam.onnx")
            mock_session.return_value = MagicMock()

            adapter = SAMAdapter(
                model_variant="vit_l",
                target_size=512,
                points_per_side=16,
                points_per_batch=32,
            )

            assert adapter.target_size == 512
            assert adapter.points_per_side == 16
            assert adapter.points_per_batch == 32
            assert adapter.points_rel.shape == (256, 2)
            assert adapter._cached_mask_input.shape == (32, 1, 256, 256)

    def test_context_manager_and_close(self):
        """Test __enter__, __exit__, and close methods for session cleanup."""
        with patch("spatialhub.models.sam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.sam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/sam.onnx")
            mock_session.return_value = MagicMock()

            with SAMAdapter() as adapter:
                assert adapter.encoder_session is not None
                assert adapter.decoder_session is not None

            assert adapter.encoder_session is None
            assert adapter.decoder_session is None

    def test_init_unsupported_variant(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model_variant"):
            SAMAdapter(model_variant="invalid_variant")


class TestSAMUtils:
    """Test suite for SAM utility functions."""

    def test_masks_to_boxes_empty(self):
        """Test bounding box calculation on all-false masks."""
        masks = np.zeros((3, 100, 100), dtype=bool)
        boxes = masks_to_boxes(masks)
        assert boxes.shape == (3, 4)
        np.testing.assert_array_equal(boxes, 0.0)

    def test_masks_to_boxes_valid(self):
        """Test bounding box calculation on known binary masks."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, 10:30, 20:50] = True
        masks[1, 40:80, 50:90] = True

        boxes = masks_to_boxes(masks)
        np.testing.assert_array_equal(boxes[0], [20, 10, 50, 30])
        np.testing.assert_array_equal(boxes[1], [50, 40, 90, 80])

    def test_point_grid_generation_coordinates(self):
        """Test that relative coordinates are symmetric and strictly within (0, 1)."""
        with patch("spatialhub.models.sam.adapter.resolve_model_path"), \
             patch("spatialhub.models.sam.adapter.create_ort_session"):
            adapter = SAMAdapter(points_per_side=4)
            grid = adapter._generate_point_grid(4)

            assert grid.shape == (16, 2)
            assert np.all(grid > 0.0)
            assert np.all(grid < 1.0)
            assert np.isclose(grid[0, 0], 1.0 / 8.0)
            assert np.isclose(grid[-1, -1], 1.0 - 1.0 / 8.0)


class TestSAMInference:
    """Test suite for SAM inference and mask generation."""

    def _create_mock_adapter(self):
        with patch("spatialhub.models.sam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.sam.adapter.create_ort_session") as mock_session:

            def resolve_factory(model_path=None, filename=None, **kwargs):
                return Path(f"/fake/{filename or Path(model_path).name}")

            mock_resolve.side_effect = resolve_factory

            mock_enc_session = MagicMock()
            mock_dec_session = MagicMock()

            mock_enc_session.run.return_value = [np.random.randn(1, 256, 64, 64).astype(np.float32)]

            def mock_decoder_run(output_names, ort_inputs):
                batch_size = ort_inputs["point_coords"].shape[0]
                orig_h = int(ort_inputs["orig_im_size"][0])
                orig_w = int(ort_inputs["orig_im_size"][1])
                masks = np.ones((batch_size, 1, orig_h, orig_w), dtype=np.float32) * 5.0
                ious = np.ones((batch_size, 1), dtype=np.float32) * 0.95
                return [masks, ious, None]

            mock_dec_session.run.side_effect = mock_decoder_run

            def session_factory(model_path, **kwargs):
                if "encoder" in str(model_path):
                    return mock_enc_session
                return mock_dec_session

            mock_session.side_effect = session_factory

            return SAMAdapter(
                encoder_onnx_path="/fake/vit_b_encoder.onnx",
                decoder_onnx_path="/fake/vit_b_decoder.onnx",
                points_per_side=4,
                points_per_batch=8,
            )

    def test_generate_masks_execution(self):
        """Test full mock inference pipeline of generate_masks."""
        adapter = self._create_mock_adapter()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.generate_masks(dummy_img)

        assert isinstance(result, SegmentationResult)
        assert result.masks.shape[1:] == (100, 100)
        assert result.masks.dtype == bool
        assert result.boxes.shape[1] == 4
        assert len(result.scores) == len(result.masks)

    def test_generate_masks_input_formats(self):
        """Test mask generation with string path, Path object, and 3-channel RGB array inputs."""
        adapter = self._create_mock_adapter()

        rgb_img = np.zeros((80, 120, 3), dtype=np.uint8)
        result_rgb = adapter.generate_masks(rgb_img)
        assert result_rgb.masks.shape[1:] == (80, 120)

        with patch("spatialhub.models.sam.adapter.load_image") as mock_load:
            mock_load.return_value = rgb_img
            result_str = adapter.generate_masks("/fake/path/to/image.jpg")
            assert result_str.masks.shape[1:] == (80, 120)
            mock_load.assert_called_with("/fake/path/to/image.jpg", color_mode="RGB")

            path_obj = Path("/fake/path/to/image.png")
            result_path = adapter.generate_masks(path_obj)
            assert result_path.masks.shape[1:] == (80, 120)
            mock_load.assert_called_with(path_obj, color_mode="RGB")

    def test_preprocess_and_encode_stages(self):
        """Test modular preprocessing and encoder forward pass."""
        adapter = self._create_mock_adapter()
        dummy_img = np.zeros((200, 400, 3), dtype=np.uint8)

        input_tensor, scale, (orig_h, orig_w) = adapter._preprocess_image(dummy_img)
        assert input_tensor.shape == (1, 3, 1024, 1024)
        assert input_tensor.dtype == np.float32
        assert (orig_h, orig_w) == (200, 400)
        assert scale == 1024.0 / 400.0

        embedding = adapter._encode_image(input_tensor)
        assert embedding.shape == (1, 256, 64, 64)

    def test_generate_masks_points_per_side_override(self):
        """Test overriding points_per_side during generate_masks call."""
        adapter = self._create_mock_adapter()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.generate_masks(dummy_img, points_per_side=8)
        assert isinstance(result, SegmentationResult)
        assert result.masks.shape[1:] == (100, 100)

    def test_generate_masks_empty_detections(self):
        """Test handling when no mask candidates pass IoU threshold."""
        adapter = self._create_mock_adapter()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = adapter.generate_masks(dummy_img, pred_iou_thresh=0.99)
        assert isinstance(result, SegmentationResult)
        assert result.boxes.shape == (0, 4)
        assert result.masks.shape == (0, 100, 100)
        assert result.scores.shape == (0,)

    def test_generate_masks_closed_session_raises(self):
        """Test exception when attempting inference after closing session resources."""
        adapter = self._create_mock_adapter()
        adapter.close()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(AttributeError):
            adapter.generate_masks(dummy_img)

    def test_postprocess_stage(self):
        """Test modular _postprocess method directly with mock decoder logits."""
        adapter = self._create_mock_adapter()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        raw_masks = [np.ones((2, 100, 100), dtype=np.float32) * 5.0]
        iou_preds = [np.array([0.95, 0.92], dtype=np.float32)]

        result = adapter._postprocess(
            image=dummy_img,
            all_raw_masks=raw_masks,
            all_iou_preds=iou_preds,
            orig_h=100,
            orig_w=100,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            box_nms_thresh=0.7,
        )

        assert isinstance(result, SegmentationResult)
        assert len(result.masks) >= 1
        assert result.masks.shape[1:] == (100, 100)
        assert result.masks.dtype == bool
