"""Unit tests for FastSAMAdapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub.models.fastsam.adapter import FastSAMAdapter, crop_mask
from spatialhub.structures import SegmentationResult


class TestFastSAMAdapterInit:
    """Test suite for FastSAM adapter initialization and resource management."""

    def test_init_defaults(self):
        """Test default parameter initialization and path resolution."""
        with patch("spatialhub.models.fastsam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.fastsam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/FastSAM-x.onnx")
            mock_session_inst = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "images"
            mock_session_inst.get_inputs.return_value = [mock_input]
            mock_session.return_value = mock_session_inst

            adapter = FastSAMAdapter()

            assert adapter.imgsz == 640
            assert adapter.default_conf_threshold == 0.05
            assert adapter.default_iou_threshold == 0.7
            assert adapter.default_max_det == 50
            assert adapter.input_name == "images"
            mock_resolve.assert_called_once_with(
                model_path=None,
                repo_id="SpatialHub/fastsam-onnx",
                filename="FastSAM-x.onnx",
            )

    def test_init_custom_variant_and_params(self):
        """Test custom variant resolution (short name 's') and custom parameters."""
        with patch("spatialhub.models.fastsam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.fastsam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/FastSAM-s.onnx")
            mock_session_inst = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input_0"
            mock_session_inst.get_inputs.return_value = [mock_input]
            mock_session.return_value = mock_session_inst

            adapter = FastSAMAdapter(
                model_variant="s",
                imgsz=320,
                conf_threshold=0.25,
                iou_threshold=0.5,
                max_det=100,
            )

            assert adapter.imgsz == 320
            assert adapter.default_conf_threshold == 0.25
            assert adapter.default_iou_threshold == 0.5
            assert adapter.default_max_det == 100
            assert adapter.input_name == "input_0"
            mock_resolve.assert_called_once_with(
                model_path=None,
                repo_id="SpatialHub/fastsam-onnx",
                filename="FastSAM-s.onnx",
            )

    def test_context_manager_and_close(self):
        """Test context manager lifecycle and close cleanup."""
        with patch("spatialhub.models.fastsam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.fastsam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/FastSAM-x.onnx")
            mock_session_inst = MagicMock()
            mock_session_inst.get_inputs.return_value = [MagicMock(name="images")]
            mock_session.return_value = mock_session_inst

            with FastSAMAdapter() as adapter:
                assert adapter.session is not None

            assert adapter.session is None

    def test_init_unsupported_variant(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model_variant"):
            FastSAMAdapter(model_variant="invalid_variant")


class TestFastSAMUtils:
    """Test suite for FastSAM internal helper operations."""

    def test_crop_mask(self):
        """Test that crop_mask zeros out pixels falling outside bounding boxes."""
        masks = np.ones((1, 100, 100), dtype=np.float32)
        boxes = np.array([[20, 30, 60, 80]], dtype=np.float32)

        cropped = crop_mask(masks, boxes)

        assert np.all(cropped[0, :30, :] == 0.0)
        assert np.all(cropped[0, 80:, :] == 0.0)
        assert np.all(cropped[0, :, :20] == 0.0)
        assert np.all(cropped[0, :, 60:] == 0.0)
        assert np.all(cropped[0, 30:80, 20:60] == 1.0)


class TestFastSAMInference:
    """Test suite for FastSAM inference, decoding, and mask generation."""

    def _create_mock_adapter(self, imgsz: int = 640):
        with patch("spatialhub.models.fastsam.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.fastsam.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/FastSAM-x.onnx")
            mock_session_inst = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "images"
            mock_session_inst.get_inputs.return_value = [mock_input]
            mock_session.return_value = mock_session_inst

            return FastSAMAdapter(imgsz=imgsz)

    def test_preprocess(self):
        """Test image resizing, letterboxing, and normalization."""
        adapter = self._create_mock_adapter(imgsz=640)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        input_tensor, letterbox = adapter._preprocess(dummy_img, orig_h=480, orig_w=640)

        assert input_tensor.shape == (1, 3, 640, 640)
        assert input_tensor.dtype == np.float32
        scale, pad_top, pad_bottom, pad_left, pad_right = letterbox
        assert scale == 1.0
        assert pad_left == 0
        assert pad_right == 0
        assert pad_top == 80
        assert pad_bottom == 80

    def test_decode_predictions_empty(self):
        """Test empty prediction array handling."""
        adapter = self._create_mock_adapter()
        # Shape: (1, 37, 100) -> preds is (100, 37)
        # All confidence scores are 0.0
        dummy_output = np.zeros((1, 37, 100), dtype=np.float32)

        boxes, scores, mask_coeffs = adapter._decode_predictions(
            output=dummy_output,
            conf_threshold=0.5,
            iou_threshold=0.7,
        )

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert mask_coeffs.shape == (0, 32)

    def test_scale_boxes(self):
        """Test scaling bounding boxes back to original image space."""
        adapter = self._create_mock_adapter(imgsz=640)
        boxes = np.array([[0, 80, 640, 560]], dtype=np.float32)
        letterbox = (1.0, 80, 80, 0, 0)

        scaled = adapter._scale_boxes(
            boxes=boxes,
            image_height=480,
            image_width=640,
            letterbox=letterbox,
        )

        np.testing.assert_allclose(scaled[0], [0, 0, 640, 480], atol=1e-5)

    def test_generate_masks_execution(self):
        """Test full mock inference pipeline of generate_masks."""
        adapter = self._create_mock_adapter(imgsz=640)

        # Mock model outputs:
        # Output 0: (1, 37, 10) - 4 bbox (cx, cy, w, h) + 1 conf + 32 mask coeffs
        out0 = np.zeros((1, 37, 10), dtype=np.float32)
        out0[0, :4, 0] = [320, 320, 200, 200]  # Box
        out0[0, 4, 0] = 0.9                   # Confidence score > 0.05
        out0[0, 5:, 0] = 1.0                  # 32 Mask coefficients

        # Output 1: (1, 32, 160, 160) - 32 Prototype masks
        out1 = np.ones((1, 32, 160, 160), dtype=np.float32)

        adapter.session.run.return_value = [out0, out1]

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = adapter.generate_masks(dummy_img)

        assert isinstance(result, SegmentationResult)
        assert result.boxes.shape == (1, 4)
        assert result.masks.shape == (1, 480, 640)
        assert result.masks.dtype == bool
        assert len(result.scores) == 1
        assert np.isclose(result.scores[0], 0.9)

        with patch("spatialhub.models.fastsam.adapter.load_image") as mock_load:
            mock_load.return_value = dummy_img
            result_str = adapter.generate_masks("/fake/test_image.jpg")
            assert result_str.masks.shape == (1, 480, 640)
            mock_load.assert_called_with("/fake/test_image.jpg", color_mode="RGB")

            path_obj = Path("/fake/test_image.png")
            result_path = adapter.generate_masks(path_obj)
            assert result_path.masks.shape == (1, 480, 640)
            mock_load.assert_called_with(path_obj, color_mode="RGB")

    def test_postprocess_stage(self):
        """Test modular _postprocess method directly."""
        adapter = self._create_mock_adapter(imgsz=640)

        out0 = np.zeros((1, 37, 10), dtype=np.float32)
        out0[0, :4, 0] = [320, 320, 200, 200]
        out0[0, 4, 0] = 0.95
        out0[0, 5:, 0] = 1.0

        out1 = np.ones((1, 32, 160, 160), dtype=np.float32)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        letterbox = (1.0, 80, 80, 0, 0)

        result = adapter._postprocess(
            image=dummy_img,
            outputs=[out0, out1],
            orig_h=480,
            orig_w=640,
            letterbox=letterbox,
            conf_threshold=0.5,
            iou_threshold=0.7,
        )

        assert isinstance(result, SegmentationResult)
        assert result.boxes.shape == (1, 4)
        assert result.masks.shape == (1, 480, 640)
        assert len(result.scores) == 1

    def test_generate_masks_empty(self):
        """Test generate_masks fallback when no detections exceed threshold."""
        adapter = self._create_mock_adapter(imgsz=640)

        out0 = np.zeros((1, 37, 10), dtype=np.float32)
        out1 = np.ones((1, 32, 160, 160), dtype=np.float32)
        adapter.session.run.return_value = [out0, out1]

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = adapter.generate_masks(dummy_img, conf_threshold=0.5)

        assert isinstance(result, SegmentationResult)
        assert result.boxes.shape == (0, 4)
        assert result.masks.shape == (0, 480, 640)
        assert result.scores.shape == (0,)

    def test_generate_masks_max_det(self):
        """Test that max_det parameter constrains total returned detections."""
        adapter = self._create_mock_adapter(imgsz=640)

        # 5 distinct detections
        out0 = np.zeros((1, 37, 5), dtype=np.float32)
        for i in range(5):
            out0[0, :4, i] = [100 * (i + 1), 100 * (i + 1), 50, 50]
            out0[0, 4, i] = 0.5 + 0.1 * i
            out0[0, 5:, i] = 1.0

        out1 = np.ones((1, 32, 160, 160), dtype=np.float32)
        adapter.session.run.return_value = [out0, out1]

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = adapter.generate_masks(dummy_img, max_det=2)

        assert len(result.scores) == 2
        assert result.boxes.shape == (2, 4)
        assert result.masks.shape == (2, 480, 640)

    def test_generate_masks_closed_session_raises(self):
        """Test exception when attempting inference after closing session resources."""
        adapter = self._create_mock_adapter(imgsz=640)
        adapter.close()
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(AttributeError):
            adapter.generate_masks(dummy_img)
