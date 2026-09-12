"""Comprehensive unit tests for spatialhub.utils.viz module."""

from pathlib import Path
import numpy as np
import pytest

from spatialhub.utils.viz import (
    draw_3d_axis,
    draw_3d_box,
    visualize_masks,
    visualize_matches,
)


class TestVisualizeMatches:
    """Test suite for visualize_matches."""

    @pytest.fixture
    def match_data(self):
        img0 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1 = np.zeros((120, 140, 3), dtype=np.uint8)
        mkpts0 = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        mkpts1 = np.array([[15, 15], [25, 25], [35, 35]], dtype=np.float32)
        mconf = np.array([0.9, 0.4, 0.85], dtype=np.float32)
        return img0, img1, mkpts0, mkpts1, mconf

    def test_visualize_matches_basic(self, match_data):
        img0, img1, mkpts0, mkpts1, mconf = match_data
        vis = visualize_matches(img0, img1, mkpts0, mkpts1, mconf=mconf, conf_thresh=0.5)

        assert isinstance(vis, np.ndarray)
        assert vis.ndim == 3
        assert vis.shape[2] == 3
        # Combined width must equal w0 + w1 (scaled to max_side)
        assert vis.shape[1] == 100 + 140

    def test_visualize_matches_from_file_paths(self, sample_pair_image_files: tuple[str, str]):
        path_a, path_b = sample_pair_image_files
        mkpts0 = np.array([[50, 50]], dtype=np.float32)
        mkpts1 = np.array([[60, 60]], dtype=np.float32)

        vis = visualize_matches(path_a, path_b, mkpts0, mkpts1)
        assert isinstance(vis, np.ndarray)
        assert vis.ndim == 3

    def test_visualize_matches_top_k(self, match_data):
        img0, img1, mkpts0, mkpts1, mconf = match_data
        vis = visualize_matches(img0, img1, mkpts0, mkpts1, mconf=mconf, top_k=1)
        assert isinstance(vis, np.ndarray)

    def test_visualize_matches_save_to_disk(self, match_data, tmp_path: Path):
        img0, img1, mkpts0, mkpts1, mconf = match_data
        out_file = tmp_path / "matches.png"

        vis = visualize_matches(img0, img1, mkpts0, mkpts1, save_path=out_file)
        assert out_file.exists()
        assert out_file.stat().st_size > 0


class TestVisualizeMasks:
    """Test suite for visualize_masks."""

    def test_visualize_masks_basic(self, sample_rgb_image: np.ndarray):
        h, w = sample_rgb_image.shape[:2]
        boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32)
        masks = np.zeros((2, h, w), dtype=bool)
        masks[0, 10:50, 10:50] = True
        masks[1, 60:100, 60:100] = True
        scores = np.array([0.95, 0.88], dtype=np.float32)

        vis = visualize_masks(sample_rgb_image, boxes=boxes, masks=masks, scores=scores)
        assert isinstance(vis, np.ndarray)
        assert vis.shape == sample_rgb_image.shape

    def test_visualize_masks_zero_detections(self, sample_rgb_image: np.ndarray):
        boxes = np.empty((0, 4), dtype=np.float32)
        masks = np.empty((0, 480, 640), dtype=bool)
        scores = np.empty((0,), dtype=np.float32)

        vis = visualize_masks(sample_rgb_image, boxes=boxes, masks=masks, scores=scores)
        assert np.array_equal(vis, sample_rgb_image)

    def test_visualize_masks_save_to_disk(self, sample_rgb_image: np.ndarray, tmp_path: Path):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        masks = np.zeros((1, 480, 640), dtype=bool)
        masks[0, 10:50, 10:50] = True
        scores = np.array([0.9], dtype=np.float32)
        out_file = tmp_path / "masks.png"

        vis = visualize_masks(sample_rgb_image, boxes=boxes, masks=masks, scores=scores, save_path=out_file)
        assert out_file.exists()


class TestDraw3DBoxAndAxis:
    """Test suite for 3D bounding box and coordinate axis drawing."""

    @pytest.fixture
    def scene_setup(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 1.5  # 1.5m in front of camera

        bbox_corners_3d = np.array([
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
            [0.1, 0.1, -0.1],
            [-0.1, 0.1, -0.1],
            [-0.1, -0.1, 0.1],
            [0.1, -0.1, 0.1],
            [0.1, 0.1, 0.1],
            [-0.1, 0.1, 0.1],
        ], dtype=np.float32)

        return img, k, pose, bbox_corners_3d

    def test_draw_3d_box_visible(self, scene_setup):
        img, k, pose, bbox_corners = scene_setup
        vis = draw_3d_box(img, pose=pose, intrinsics=k, bbox_corners_3d=bbox_corners)
        assert vis.shape == (480, 640, 3)
        # Should contain non-zero drawn line pixels
        assert np.any(vis != 0)

    def test_draw_3d_box_behind_camera_clipped(self, scene_setup):
        img, k, pose, bbox_corners = scene_setup
        pose_behind = pose.copy()
        pose_behind[2, 3] = -1.0  # Behind camera

        vis = draw_3d_box(img, pose=pose_behind, intrinsics=k, bbox_corners_3d=bbox_corners)
        # Returns unedited image copy
        assert np.array_equal(vis, img)

    def test_draw_3d_axis_visible(self, scene_setup):
        img, k, pose, _ = scene_setup
        vis = draw_3d_axis(img, pose=pose, intrinsics=k, axis_length=0.1)
        assert vis.shape == (480, 640, 3)
        assert np.any(vis != 0)

    def test_draw_3d_axis_behind_camera_clipped(self, scene_setup):
        img, k, pose, _ = scene_setup
        pose_behind = pose.copy()
        pose_behind[2, 3] = -2.0  # Behind camera

        vis = draw_3d_axis(img, pose=pose_behind, intrinsics=k)
        assert np.array_equal(vis, img)
