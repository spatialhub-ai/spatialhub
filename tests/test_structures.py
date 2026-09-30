"""Unit tests for spatialhub.structures return contracts and validation."""

from __future__ import annotations

import numpy as np
import pytest

from spatialhub.structures import (
    DepthPredictionResult,
    FeatureExtractionResult,
    MatchResult,
    PoseEstimationResult,
    SegmentationResult,
    _check_array,
)


class TestCheckArray:
    """Unit tests for the internal _check_array validation helper."""

    def test_valid_array(self) -> None:
        arr = np.zeros((2, 3, 3), dtype=np.float32)
        _check_array("test", arr, ndim=3, shape_suffix=(3, 3))

    def test_none_handling(self) -> None:
        _check_array("test", None, optional=True)
        with pytest.raises(ValueError, match="is required and cannot be None"):
            _check_array("test", None, optional=False)

    def test_non_numpy_type(self) -> None:
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            _check_array("test", [1, 2, 3])  # type: ignore[arg-type]

    def test_invalid_ndim(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="expected ndim=3"):
            _check_array("test", arr, ndim=3)

    def test_tuple_ndim(self) -> None:
        arr = np.zeros((2, 3))
        _check_array("test", arr, ndim=(2, 3))

        arr_4d = np.zeros((1, 2, 3, 4))
        with pytest.raises(ValueError, match=r"expected ndim=\(2, 3\)"):
            _check_array("test", arr_4d, ndim=(2, 3))

    def test_invalid_shape_suffix(self) -> None:
        arr = np.zeros((2, 4, 4))
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(3, 3\)"):
            _check_array("test", arr, shape_suffix=(3, 3))


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_valid_match_result(self) -> None:
        kpts_a = np.zeros((10, 2), dtype=np.float32)
        kpts_b = np.zeros((10, 2), dtype=np.float32)
        conf = np.ones((10,), dtype=np.float32)
        res = MatchResult(
            image_a="a.png",
            image_b="b.png",
            keypoints_a=kpts_a,
            keypoints_b=kpts_b,
            confidence=conf,
        )
        assert len(res.keypoints_a) == 10
        assert len(res.keypoints_b) == 10
        assert len(res.confidence) == 10

    def test_length_mismatch_raises(self) -> None:
        kpts_a = np.zeros((10, 2), dtype=np.float32)
        kpts_b = np.zeros((8, 2), dtype=np.float32)
        conf = np.ones((10,), dtype=np.float32)
        with pytest.raises(ValueError, match="Array length mismatch"):
            MatchResult(
                image_a="a.png",
                image_b="b.png",
                keypoints_a=kpts_a,
                keypoints_b=kpts_b,
                confidence=conf,
            )

    def test_invalid_keypoint_shape_raises(self) -> None:
        kpts_a = np.zeros((10, 3), dtype=np.float32)
        kpts_b = np.zeros((10, 3), dtype=np.float32)
        conf = np.ones((10,), dtype=np.float32)
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(2,\)"):
            MatchResult(
                image_a="a.png",
                image_b="b.png",
                keypoints_a=kpts_a,
                keypoints_b=kpts_b,
                confidence=conf,
            )


class TestDepthPredictionResult:
    """Tests for DepthPredictionResult dataclass."""

    def test_2d_depth_promoted_to_3d(self) -> None:
        depth = np.ones((100, 100), dtype=np.float32)
        res = DepthPredictionResult(image="test.png", depth=depth)
        assert res.depth.shape == (1, 100, 100)

    def test_3d_depth_preserved(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        res = DepthPredictionResult(image="test.png", depth=depth)
        assert res.depth.shape == (2, 100, 100)

    def test_conf_shape_matching(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        conf = np.ones((2, 100, 100), dtype=np.float32)
        res = DepthPredictionResult(image="test.png", depth=depth, conf=conf)
        assert res.conf is not None and res.conf.shape == (2, 100, 100)

    def test_conf_shape_mismatch_raises(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        conf = np.ones((2, 50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="conf shape .* must match depth shape"):
            DepthPredictionResult(image="test.png", depth=depth, conf=conf)

    def test_intrinsics_and_extrinsics_promotion(self) -> None:
        depth = np.ones((1, 100, 100), dtype=np.float32)
        k = np.eye(3, dtype=np.float32)
        t = np.eye(4, dtype=np.float32)
        res = DepthPredictionResult(
            image="test.png",
            depth=depth,
            intrinsics=k,
            extrinsics=t,
        )
        assert res.intrinsics is not None and res.intrinsics.shape == (1, 3, 3)
        assert res.extrinsics is not None and res.extrinsics.shape == (1, 4, 4)

    def test_invalid_intrinsics_shape_raises(self) -> None:
        depth = np.ones((1, 100, 100), dtype=np.float32)
        k = np.eye(4, dtype=np.float32)
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(3, 3\)"):
            DepthPredictionResult(image="test.png", depth=depth, intrinsics=k)


class TestFeatureExtractionResult:
    """Tests for FeatureExtractionResult dataclass."""

    def test_global_embeddings(self) -> None:
        features = np.zeros((4, 768), dtype=np.float32)
        res = FeatureExtractionResult(
            images=np.zeros((4, 224, 224, 3), dtype=np.uint8),
            features=features,
            embedding_type="global",
            l2_normalized=True,
        )
        assert res.features.shape == (4, 768)
        assert res.l2_normalized is True

    def test_dense_embeddings(self) -> None:
        features = np.zeros((2, 16, 16, 768), dtype=np.float32)
        res = FeatureExtractionResult(
            images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
            features=features,
            embedding_type="dense",
        )
        assert res.features.shape == (2, 16, 16, 768)

    def test_invalid_global_dim_raises(self) -> None:
        features = np.zeros((2, 16, 16, 768), dtype=np.float32)
        with pytest.raises(ValueError, match="Global feature embeddings must have ndim=2"):
            FeatureExtractionResult(
                images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
                features=features,
                embedding_type="global",
            )


class TestSegmentationResult:
    """Tests for SegmentationResult dataclass."""

    def test_valid_segmentation(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.zeros((3, 4), dtype=np.float32)
        masks = np.zeros((3, 480, 640), dtype=bool)
        scores = np.ones((3,), dtype=np.float32)
        class_ids = np.array([0, 1, 2], dtype=int)
        class_names = ["a", "b", "c"]

        res = SegmentationResult(
            image=img,
            boxes=boxes,
            masks=masks,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
        )
        assert len(res.boxes) == 3

    def test_empty_segmentation(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.zeros((0, 4), dtype=np.float32)
        masks = np.zeros((0, 480, 640), dtype=bool)
        scores = np.zeros((0,), dtype=np.float32)

        res = SegmentationResult(image=img, boxes=boxes, masks=masks, scores=scores)
        assert len(res.boxes) == 0

    def test_length_mismatch_raises(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.zeros((3, 4), dtype=np.float32)
        masks = np.zeros((2, 480, 640), dtype=bool)
        scores = np.ones((3,), dtype=np.float32)

        with pytest.raises(ValueError, match="Array length mismatch"):
            SegmentationResult(image=img, boxes=boxes, masks=masks, scores=scores)

    def test_class_ids_mismatch_raises(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.zeros((3, 4), dtype=np.float32)
        masks = np.zeros((3, 480, 640), dtype=bool)
        scores = np.ones((3,), dtype=np.float32)
        class_ids = np.array([0, 1], dtype=int)

        with pytest.raises(ValueError, match="class_ids length .* does not match boxes"):
            SegmentationResult(
                image=img,
                boxes=boxes,
                masks=masks,
                scores=scores,
                class_ids=class_ids,
            )


class TestPoseEstimationResult:
    """Tests for PoseEstimationResult dataclass."""

    def test_single_pose_promoted(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        k = np.eye(3, dtype=np.float32)
        res = PoseEstimationResult(
            image=img,
            poses=pose,
            intrinsics=k,
            scores=np.array([0.95], dtype=np.float32),
            labels=["object_a"],
        )
        assert res.poses.shape == (1, 4, 4)
        assert len(res) == 1
        assert np.isclose(res.best_score, 0.95)
        assert np.allclose(res.best_pose, np.eye(4))

    def test_scalar_score_promoted(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        k = np.eye(3, dtype=np.float32)
        res = PoseEstimationResult(
            image=img,
            poses=pose,
            intrinsics=k,
            scores=0.88,  # type: ignore[arg-type]
            labels="single_label",  # type: ignore[arg-type]
        )
        assert res.scores is not None and res.scores.shape == (1,)
        assert res.labels == ["single_label"]

    def test_best_pose_multi(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.zeros((2, 4, 4), dtype=np.float32)
        poses[0, 0, 3] = 10.0
        poses[1, 0, 3] = 20.0
        scores = np.array([0.3, 0.9], dtype=np.float32)
        k = np.eye(3, dtype=np.float32)

        res = PoseEstimationResult(
            image=img,
            poses=poses,
            intrinsics=k,
            scores=scores,
        )
        assert res.best_score == pytest.approx(0.9)
        assert res.best_pose[0, 3] == pytest.approx(20.0)

    def test_invalid_pose_shape_raises(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = np.eye(3, dtype=np.float32)
        k = np.eye(3, dtype=np.float32)
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(4, 4\)"):
            PoseEstimationResult(image=img, poses=pose, intrinsics=k)
