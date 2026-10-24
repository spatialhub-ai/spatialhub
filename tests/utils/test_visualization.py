"""Comprehensive unit tests for spatialhub.utils.visualization module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spatialhub.utils.visualization import (
    _resolve_colors,
    blend_masks,
    draw_3d_axis,
    draw_3d_box,
    draw_boxes_2d,
    visualize_dense_features,
    visualize_depth,
    visualize_global_features,
    visualize_matched_keypoints,
)


class TestResolveColors:
    """Test suite for _resolve_colors internal helper."""

    def test_none_generates_distinct_colors(self) -> None:
        colors = _resolve_colors(None, 5)
        assert len(colors) == 5
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)

    def test_single_color(self) -> None:
        colors = _resolve_colors((255, 0, 0), 3)
        assert len(colors) == 1
        assert colors[0] == (255, 0, 0)

    def test_color_list(self) -> None:
        raw_list = [(255, 0, 0), (0, 255, 0)]
        colors = _resolve_colors(raw_list, 2)
        assert len(colors) == 2
        assert colors == [(255, 0, 0), (0, 255, 0)]

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="colors list cannot be empty"):
            _resolve_colors([], 2)


class TestDraw3DBox:
    """Test suite for draw_3d_box."""

    @pytest.fixture
    def test_canvas(self) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def camera_intrinsics(self) -> np.ndarray:
        return np.array([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    @pytest.fixture
    def cube_corners(self) -> np.ndarray:
        # 8 corners of unit cube [-0.5, 0.5]
        return np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ], dtype=np.float32)

    def test_draw_single_box(
        self,
        test_canvas: np.ndarray,
        camera_intrinsics: np.ndarray,
        cube_corners: np.ndarray,
    ) -> None:
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 2.0  # 2 meters in front of camera

        annotated = draw_3d_box(
            test_canvas,
            pose=pose,
            intrinsics=camera_intrinsics,
            bbox_corners_3d=cube_corners,
            color=(0, 255, 0),
        )

        assert annotated.shape == (100, 100, 3)
        assert not np.array_equal(annotated, test_canvas)  # Wireframe drawn
        assert np.any(annotated[:, :, 1] == 255)  # Green color drawn

    def test_draw_batched_boxes(
        self,
        test_canvas: np.ndarray,
        camera_intrinsics: np.ndarray,
        cube_corners: np.ndarray,
    ) -> None:
        poses = np.zeros((2, 4, 4), dtype=np.float32)
        poses[0] = np.eye(4)
        poses[0, 2, 3] = 2.0
        poses[1] = np.eye(4)
        poses[1, 2, 3] = 3.0

        annotated = draw_3d_box(
            test_canvas,
            pose=poses,
            intrinsics=camera_intrinsics,
            bbox_corners_3d=cube_corners,
        )
        assert annotated.shape == (100, 100, 3)

    def test_invalid_corners_shape_raises(
        self,
        test_canvas: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> None:
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 2.0
        with pytest.raises(ValueError, match="bbox_corners_3d must have 8 corners"):
            draw_3d_box(test_canvas, pose, camera_intrinsics, np.zeros((7, 3), dtype=np.float32))


class TestDraw3DAxis:
    """Test suite for draw_3d_axis."""

    def test_draw_single_axis(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 2.0

        annotated = draw_3d_axis(img, pose=pose, intrinsics=k, axis_length=0.1)
        assert annotated.shape == (100, 100, 3)
        assert not np.array_equal(annotated, img)


class TestBlendMasks:
    """Test suite for blend_masks."""

    def test_blend_single_and_batched_masks(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True

        blended_single = blend_masks(img, mask, alpha=0.5, colors=(255, 0, 0))
        assert blended_single.shape == (50, 50, 3)
        assert blended_single[15, 15, 0] > 0

        batched_masks = np.stack([mask, mask], axis=0)
        blended_batch = blend_masks(img, batched_masks, draw_border=True)
        assert blended_batch.shape == (50, 50, 3)

    def test_empty_masks_returns_copy(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        empty = np.zeros((0, 50, 50), dtype=bool)
        out = blend_masks(img, empty)
        assert np.array_equal(out, img)

    def test_mismatched_dimensions_raises(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        bad_mask = np.zeros((30, 30), dtype=bool)
        with pytest.raises(ValueError, match="mask spatial shape .* does not match"):
            blend_masks(img, bad_mask)


class TestDrawBoxes2D:
    """Test suite for draw_boxes_2d."""

    def test_draw_boxes_with_labels_and_scores(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=np.float32)
        scores = np.array([0.95, 0.80], dtype=np.float32)
        names = ["cat", "dog"]

        annotated = draw_boxes_2d(img, boxes=boxes, scores=scores, names=names, colors=[(255, 0, 0), (0, 255, 0)])
        assert annotated.shape == (100, 100, 3)
        assert not np.array_equal(annotated, img)

    def test_empty_boxes_returns_copy(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = draw_boxes_2d(img, boxes=np.zeros((0, 4), dtype=np.float32))
        assert np.array_equal(out, img)


class TestVisualizeDepth:
    """Test suite for visualize_depth."""

    def test_visualize_depth_color(self) -> None:
        depth = np.linspace(0.5, 5.0, 10000, dtype=np.float32).reshape(100, 100)
        vis = visualize_depth(depth, colormap="turbo")
        assert vis.shape == (100, 100, 3)
        assert vis.dtype == np.uint8

    def test_visualize_depth_all_zero(self) -> None:
        depth = np.zeros((50, 50), dtype=np.float32)
        vis = visualize_depth(depth)
        assert vis.shape == (50, 50, 3)
        assert np.all(vis == 0)


class TestVisualizeMatchedKeypoints:
    """Test suite for visualize_matched_keypoints."""

    def test_visualize_matches(self) -> None:
        img_a = np.zeros((100, 80, 3), dtype=np.uint8)
        img_b = np.zeros((100, 80, 3), dtype=np.uint8)
        kpts_a = np.array([[10, 10], [20, 20]], dtype=np.float32)
        kpts_b = np.array([[15, 15], [25, 25]], dtype=np.float32)
        conf = np.array([0.9, 0.8], dtype=np.float32)

        vis = visualize_matched_keypoints(img_a, img_b, kpts_a, kpts_b, confidence=conf, top_k=1)
        assert vis.ndim == 3
        assert vis.shape[2] == 3
        assert vis.shape[1] > 80  # Two images concatenated horizontally


class TestVisualizeFeatures:
    """Test suite for visualize_dense_features and visualize_global_features."""

    def test_visualize_dense_features(self) -> None:
        feats = np.random.rand(32, 32, 64).astype(np.float32)
        vis = visualize_dense_features(feats)
        assert vis.shape == (32, 32, 3)
        assert vis.dtype == np.uint8

    def test_visualize_dense_features_low_dim(self) -> None:
        feats_2d = np.random.rand(16, 16, 2).astype(np.float32)
        vis = visualize_dense_features(feats_2d)
        assert vis.shape == (16, 16, 3)

    def test_visualize_global_features(self) -> None:
        feats = np.random.rand(10, 128).astype(np.float32)
        vis = visualize_global_features(feats, colormap="viridis")
        assert vis.shape == (10, 10, 3)
        assert vis.dtype == np.uint8
