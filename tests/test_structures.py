"""Comprehensive unit tests for spatialhub.structures return contracts and validation."""

from __future__ import annotations

import numpy as np
import pytest

from spatialhub.structures import (
    BoundingBox3D,
    CameraExtrinsics,
    CameraIntrinsics,
    DepthMap,
    FeatureEmbedding,
    ImageBuffers,
    InstanceSegmentation,
    KeypointMatches,
    ObjectPose6D,
    RigidTransform3D,
)


class TestRigidTransform3D:
    """Tests for RigidTransform3D and base _BatchedArray container."""

    def test_2d_matrix_promoted_to_batch(self) -> None:
        mat = np.eye(4, dtype=np.float32)
        transform = RigidTransform3D(matrix=mat)
        assert transform.matrix.shape == (1, 4, 4)
        assert len(transform) == 1
        assert np.allclose(transform.matrix[0], np.eye(4))

    def test_3d_batch_preserved(self) -> None:
        mat = np.zeros((3, 4, 4), dtype=np.float32)
        transform = RigidTransform3D(matrix=mat)
        assert transform.matrix.shape == (3, 4, 4)
        assert len(transform) == 3

    def test_indexing_and_slicing(self) -> None:
        mat = np.zeros((4, 4, 4), dtype=np.float32)
        for i in range(4):
            mat[i, 0, 3] = float(i)
        transform = RigidTransform3D(matrix=mat)

        single = transform[2]
        assert isinstance(single, RigidTransform3D)
        assert single.matrix.shape == (1, 4, 4)
        assert single.matrix[0, 0, 3] == pytest.approx(2.0)

        sliced = transform[1:3]
        assert isinstance(sliced, RigidTransform3D)
        assert len(sliced) == 2
        assert sliced.matrix[0, 0, 3] == pytest.approx(1.0)
        assert sliced.matrix[1, 0, 3] == pytest.approx(2.0)

        mask = np.array([True, False, True, False])
        filtered = transform[mask]
        assert len(filtered) == 2

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(4, 4\)"):
            RigidTransform3D(matrix=np.eye(3, dtype=np.float32))

        with pytest.raises(ValueError, match=r"expected ndim="):
            RigidTransform3D(matrix=np.zeros((1, 2, 4, 4), dtype=np.float32))

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            RigidTransform3D(matrix=np.eye(4, dtype=np.int32))


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics dataclass."""

    def test_2d_intrinsics_promoted_to_batch(self) -> None:
        k = np.eye(3, dtype=np.float32)
        intrinsics = CameraIntrinsics(matrix=k)
        assert intrinsics.matrix.shape == (1, 3, 3)
        assert len(intrinsics) == 1

    def test_3d_batch_intrinsics(self) -> None:
        k_batch = np.zeros((5, 3, 3), dtype=np.float32)
        intrinsics = CameraIntrinsics(matrix=k_batch)
        assert intrinsics.matrix.shape == (5, 3, 3)
        assert len(intrinsics) == 5

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(3, 3\)"):
            CameraIntrinsics(matrix=np.eye(4, dtype=np.float32))


class TestCameraExtrinsics:
    """Tests for CameraExtrinsics dataclass."""

    def test_subclass_of_rigid_transform(self) -> None:
        extr = CameraExtrinsics(matrix=np.eye(4, dtype=np.float32))
        assert isinstance(extr, RigidTransform3D)
        assert extr.matrix.shape == (1, 4, 4)


class TestBoundingBox3D:
    """Tests for BoundingBox3D dataclass."""

    def test_2d_corners_promoted_to_batch(self) -> None:
        corners = np.zeros((8, 3), dtype=np.float32)
        bbox = BoundingBox3D(corners=corners)
        assert bbox.corners.shape == (1, 8, 3)
        assert len(bbox) == 1

    def test_3d_batch_corners(self) -> None:
        corners = np.zeros((4, 8, 3), dtype=np.float32)
        bbox = BoundingBox3D(corners=corners)
        assert bbox.corners.shape == (4, 8, 3)
        assert len(bbox) == 4

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(8, 3\)"):
            BoundingBox3D(corners=np.zeros((7, 3), dtype=np.float32))


class TestKeypointMatches:
    """Tests for KeypointMatches dataclass."""

    def test_valid_matches(self) -> None:
        kpts_a = np.zeros((10, 2), dtype=np.float32)
        kpts_b = np.zeros((10, 2), dtype=np.float32)
        conf = np.ones((10,), dtype=np.float32)
        res = KeypointMatches(keypoints_a=kpts_a, keypoints_b=kpts_b, confidence=conf)
        assert len(res) == 10
        assert res.keypoints_a.shape == (10, 2)
        assert res.keypoints_b.shape == (10, 2)
        assert res.confidence.shape == (10,)

    def test_empty_matches(self) -> None:
        res = KeypointMatches(
            keypoints_a=np.zeros((0, 2), dtype=np.float32),
            keypoints_b=np.zeros((0, 2), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
        )
        assert len(res) == 0

    def test_length_mismatch_raises(self) -> None:
        kpts_a = np.zeros((10, 2), dtype=np.float32)
        kpts_b = np.zeros((8, 2), dtype=np.float32)
        conf = np.ones((10,), dtype=np.float32)
        with pytest.raises(ValueError, match="Array length mismatch"):
            KeypointMatches(keypoints_a=kpts_a, keypoints_b=kpts_b, confidence=conf)

    def test_invalid_keypoint_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(2,\)"):
            KeypointMatches(
                keypoints_a=np.zeros((10, 3), dtype=np.float32),
                keypoints_b=np.zeros((10, 2), dtype=np.float32),
                confidence=np.ones((10,), dtype=np.float32),
            )

    def test_indexing_and_slicing(self) -> None:
        kpts_a = np.zeros((10, 2), dtype=np.float32)
        kpts_b = np.zeros((10, 2), dtype=np.float32)
        conf = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        res = KeypointMatches(keypoints_a=kpts_a, keypoints_b=kpts_b, confidence=conf)

        sliced = res[:5]
        assert isinstance(sliced, KeypointMatches)
        assert len(sliced) == 5

        single = res[3]
        assert isinstance(single, KeypointMatches)
        assert len(single) == 1
        assert single.keypoints_a.shape == (1, 2)

        mask = res.confidence > 0.5
        filtered = res[mask]
        assert len(filtered) == 5

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            KeypointMatches(
                keypoints_a=np.zeros((5, 2), dtype=np.int32),
                keypoints_b=np.zeros((5, 2), dtype=np.float32),
                confidence=np.ones((5,), dtype=np.float32),
            )


class TestDepthMap:
    """Tests for DepthMap dataclass."""

    def test_2d_depth_promoted_to_3d(self) -> None:
        depth = np.ones((100, 100), dtype=np.float32)
        res = DepthMap(depth=depth)
        assert res.depth.shape == (1, 100, 100)
        assert len(res) == 1
        assert res.depth_type == "metric"
        assert res.confidence is None

    def test_3d_depth_preserved(self) -> None:
        depth = np.ones((3, 100, 100), dtype=np.float32)
        res = DepthMap(depth=depth, depth_type="relative")
        assert res.depth.shape == (3, 100, 100)
        assert len(res) == 3
        assert res.depth_type == "relative"

    def test_confidence_matching(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        conf = np.zeros((2, 100, 100), dtype=np.float32)
        res = DepthMap(depth=depth, confidence=conf)
        assert res.confidence is not None
        assert res.confidence.shape == (2, 100, 100)

    def test_confidence_2d_promoted_with_2d_depth(self) -> None:
        depth = np.ones((100, 100), dtype=np.float32)
        conf = np.zeros((100, 100), dtype=np.float32)
        res = DepthMap(depth=depth, confidence=conf)
        assert res.depth.shape == (1, 100, 100)
        assert res.confidence is not None and res.confidence.shape == (1, 100, 100)

    def test_confidence_shape_mismatch_raises(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        conf = np.ones((2, 50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="confidence shape .* must match depth shape"):
            DepthMap(depth=depth, confidence=conf)

    def test_indexing_and_slicing(self) -> None:
        depth = np.ones((4, 100, 100), dtype=np.float32)
        conf = np.ones((4, 100, 100), dtype=np.float32)
        res = DepthMap(depth=depth, confidence=conf, depth_type="inverse")

        single = res[1]
        assert isinstance(single, DepthMap)
        assert single.depth.shape == (1, 100, 100)
        assert single.confidence is not None and single.confidence.shape == (1, 100, 100)
        assert single.depth_type == "inverse"

        sliced = res[1:3]
        assert len(sliced) == 2

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            DepthMap(depth=np.ones((100, 100), dtype=np.int32))

    def test_intrinsics_and_extrinsics_ndarray_conversion(self) -> None:
        depth = np.ones((100, 100), dtype=np.float32)
        K = np.eye(3, dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        res = DepthMap(depth=depth, intrinsics=K, extrinsics=T)

        assert isinstance(res.intrinsics, CameraIntrinsics)
        assert res.intrinsics.matrix.shape == (1, 3, 3)
        assert isinstance(res.extrinsics, CameraExtrinsics)
        assert res.extrinsics.matrix.shape == (1, 4, 4)

    def test_intrinsics_and_extrinsics_instances(self) -> None:
        depth = np.ones((2, 100, 100), dtype=np.float32)
        cam_int = CameraIntrinsics(np.zeros((2, 3, 3), dtype=np.float32))
        cam_ext = CameraExtrinsics(np.zeros((2, 4, 4), dtype=np.float32))
        res = DepthMap(depth=depth, intrinsics=cam_int, extrinsics=cam_ext)

        assert res.intrinsics is cam_int
        assert res.extrinsics is cam_ext

    def test_intrinsics_batch_mismatch_raises(self) -> None:
        depth = np.ones((3, 100, 100), dtype=np.float32)
        K = np.zeros((2, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="intrinsics batch size .* must match depth batch size"):
            DepthMap(depth=depth, intrinsics=K)

    def test_extrinsics_batch_mismatch_raises(self) -> None:
        depth = np.ones((3, 100, 100), dtype=np.float32)
        T = np.zeros((2, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="extrinsics batch size .* must match depth batch size"):
            DepthMap(depth=depth, extrinsics=T)

    def test_invalid_intrinsics_and_extrinsics_type_raises(self) -> None:
        depth = np.ones((100, 100), dtype=np.float32)
        with pytest.raises(TypeError, match="intrinsics must be CameraIntrinsics"):
            DepthMap(depth=depth, intrinsics="invalid")
        with pytest.raises(TypeError, match="extrinsics must be CameraExtrinsics"):
            DepthMap(depth=depth, extrinsics="invalid")

    def test_indexing_preserves_intrinsics_and_extrinsics(self) -> None:
        depth = np.ones((3, 100, 100), dtype=np.float32)
        K = np.zeros((3, 3, 3), dtype=np.float32)
        T = np.zeros((3, 4, 4), dtype=np.float32)
        res = DepthMap(depth=depth, intrinsics=K, extrinsics=T)

        single = res[0]
        assert isinstance(single, DepthMap)
        assert isinstance(single.intrinsics, CameraIntrinsics)
        assert single.intrinsics.matrix.shape == (1, 3, 3)
        assert isinstance(single.extrinsics, CameraExtrinsics)
        assert single.extrinsics.matrix.shape == (1, 4, 4)

        sliced = res[1:3]
        assert len(sliced) == 2
        assert len(sliced.intrinsics) == 2
        assert len(sliced.extrinsics) == 2



class TestFeatureEmbedding:
    """Tests for FeatureEmbedding dataclass."""

    def test_valid_global_embedding(self) -> None:
        feats = np.zeros((4, 768), dtype=np.float32)
        res = FeatureEmbedding(features=feats, embedding_type="global", l2_normalized=True)
        assert len(res) == 4
        assert res.features.shape == (4, 768)
        assert res.embedding_type == "global"
        assert res.l2_normalized is True

    def test_valid_dense_embedding(self) -> None:
        feats = np.zeros((2, 16, 16, 384), dtype=np.float32)
        res = FeatureEmbedding(features=feats, embedding_type="dense", l2_normalized=False)
        assert len(res) == 2
        assert res.features.shape == (2, 16, 16, 384)
        assert res.embedding_type == "dense"

    def test_global_dimension_mismatch_raises(self) -> None:
        feats = np.zeros((2, 16, 16, 768), dtype=np.float32)
        with pytest.raises(ValueError, match="Global feature embeddings must have ndim=2"):
            FeatureEmbedding(features=feats, embedding_type="global")

    def test_dense_dimension_mismatch_raises(self) -> None:
        feats = np.zeros((2, 768), dtype=np.float32)
        with pytest.raises(ValueError, match="Dense feature embeddings must have ndim=4"):
            FeatureEmbedding(features=feats, embedding_type="dense")

    def test_indexing_and_slicing(self) -> None:
        feats = np.zeros((4, 768), dtype=np.float32)
        res = FeatureEmbedding(features=feats)
        single = res[0]
        assert isinstance(single, FeatureEmbedding)
        assert single.features.shape == (1, 768)
        assert len(single) == 1

        sliced = res[1:3]
        assert len(sliced) == 2

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            FeatureEmbedding(features=np.zeros((2, 768), dtype=np.int32))


class TestInstanceSegmentation:
    """Tests for InstanceSegmentation dataclass."""

    def test_valid_segmentation(self) -> None:
        boxes = np.zeros((3, 4), dtype=np.float32)
        masks = np.zeros((3, 64, 64), dtype=bool)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        ids = np.array([1, 2, 3], dtype=np.int64)
        names = ["cat", "dog", "bird"]

        res = InstanceSegmentation(masks=masks, boxes=boxes, scores=scores, ids=ids, names=names)
        assert len(res) == 3
        assert res.masks.shape == (3, 64, 64)
        assert res.boxes.shape == (3, 4)
        assert res.scores.shape == (3,)
        assert res.ids is not None and res.ids.shape == (3,)
        assert res.names == ["cat", "dog", "bird"]

    def test_single_instance_promoted(self) -> None:
        box = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        score = 0.95
        res = InstanceSegmentation(masks=mask, boxes=box, scores=score, ids=1, names=["cat"])
        assert len(res) == 1
        assert res.masks.shape == (1, 64, 64)
        assert res.boxes.shape == (1, 4)
        assert res.scores.shape == (1,)
        assert res.ids is not None and res.ids.shape == (1,)

    def test_empty_segmentation(self) -> None:
        res = InstanceSegmentation(
            masks=np.zeros((0, 64, 64), dtype=bool),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
        )
        assert len(res) == 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Array length mismatch"):
            InstanceSegmentation(
                masks=np.zeros((2, 64, 64), dtype=bool),
                boxes=np.zeros((3, 4), dtype=np.float32),
                scores=np.ones((3,), dtype=np.float32),
            )

    def test_ids_and_names_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"ids length .* does not match boxes"):
            InstanceSegmentation(
                masks=np.zeros((2, 64, 64), dtype=bool),
                boxes=np.zeros((2, 4), dtype=np.float32),
                scores=np.ones((2,), dtype=np.float32),
                ids=np.array([1]),
            )

        with pytest.raises(ValueError, match=r"names length .* does not match boxes"):
            InstanceSegmentation(
                masks=np.zeros((2, 64, 64), dtype=bool),
                boxes=np.zeros((2, 4), dtype=np.float32),
                scores=np.ones((2,), dtype=np.float32),
                names=["cat"],
            )

    def test_indexing_and_slicing(self) -> None:
        boxes = np.zeros((3, 4), dtype=np.float32)
        masks = np.zeros((3, 64, 64), dtype=bool)
        scores = np.array([0.2, 0.9, 0.6], dtype=np.float32)
        ids = np.array([10, 20, 30], dtype=np.int64)
        names = ["a", "b", "c"]

        res = InstanceSegmentation(masks=masks, boxes=boxes, scores=scores, ids=ids, names=names)

        single = res[1]
        assert isinstance(single, InstanceSegmentation)
        assert len(single) == 1
        assert single.names == ["b"]
        assert single.ids is not None and single.ids[0] == 20

        sliced = res[:2]
        assert len(sliced) == 2
        assert sliced.names == ["a", "b"]

        mask_filter = res.scores > 0.5
        filtered = res[mask_filter]
        assert len(filtered) == 2
        assert filtered.names == ["b", "c"]

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            InstanceSegmentation(
                masks=np.zeros((2, 64, 64), dtype=np.float32),
                boxes=np.zeros((2, 4), dtype=np.float32),
                scores=np.ones((2,), dtype=np.float32),
            )


class TestObjectPose6D:
    """Tests for ObjectPose6D dataclass."""

    def test_single_pose_promoted(self) -> None:
        pose = np.eye(4, dtype=np.float32)
        res = ObjectPose6D(pose=pose, scores=0.85, ids=1, names="cube")
        assert len(res) == 1
        assert res.pose.shape == (1, 4, 4)
        assert res.scores is not None and res.scores.shape == (1,)
        assert res.ids is not None and res.ids.shape == (1,)
        assert res.names == ["cube"]

    def test_batched_poses(self) -> None:
        poses = np.zeros((3, 4, 4), dtype=np.float32)
        scores = np.array([0.9, 0.4, 0.8], dtype=np.float32)
        ids = np.array([1, 2, 3], dtype=np.int64)
        names = ["obj1", "obj2", "obj3"]

        res = ObjectPose6D(pose=poses, scores=scores, ids=ids, names=names)
        assert len(res) == 3
        assert res.pose.shape == (3, 4, 4)

    def test_length_mismatch_raises(self) -> None:
        poses = np.zeros((2, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="scores count .* does not match pose count"):
            ObjectPose6D(pose=poses, scores=np.ones((3,), dtype=np.float32))

        with pytest.raises(ValueError, match="ids count .* does not match pose count"):
            ObjectPose6D(pose=poses, ids=np.array([1, 2, 3], dtype=np.int64))

        with pytest.raises(ValueError, match="names count .* does not match pose count"):
            ObjectPose6D(pose=poses, names=["a"])

    def test_indexing_and_slicing(self) -> None:
        poses = np.zeros((3, 4, 4), dtype=np.float32)
        poses[0, 0, 3] = 1.0
        poses[1, 0, 3] = 2.0
        poses[2, 0, 3] = 3.0
        scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        names = ["drill", "clamp", "hammer"]

        res = ObjectPose6D(pose=poses, scores=scores, names=names)

        single = res[1]
        assert isinstance(single, ObjectPose6D)
        assert len(single) == 1
        assert single.pose[0, 0, 3] == pytest.approx(2.0)
        assert single.names == ["clamp"]

        sliced = res[:2]
        assert len(sliced) == 2
        assert sliced.names == ["drill", "clamp"]

        mask = res.scores > 0.4
        filtered = res[mask]
        assert len(filtered) == 2
        assert filtered.names == ["clamp", "hammer"]

    def test_invalid_shape_and_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match=r"trailing dimensions must be \(4, 4\)"):
            ObjectPose6D(pose=np.eye(3, dtype=np.float32))

        with pytest.raises(TypeError, match=r"expected dtype"):
            ObjectPose6D(pose=np.eye(4, dtype=np.int32))


class TestImageBuffers:
    """Tests for ImageBuffers dataclass."""

    def test_valid_multimodal_buffers(self) -> None:
        rgba = np.zeros((3, 64, 64, 4), dtype=np.uint8)
        depth = np.zeros((3, 64, 64), dtype=np.float32)
        normal = np.zeros((3, 64, 64, 3), dtype=np.float32)
        mask = np.ones((3, 64, 64), dtype=bool)

        buffers = ImageBuffers(rgba=rgba, depth=depth, normal=normal, mask=mask)
        assert len(buffers) == 3
        assert buffers.rgba is not None and buffers.rgba.shape == (3, 64, 64, 4)
        assert buffers.depth is not None and buffers.depth.shape == (3, 64, 64)
        assert buffers.normal is not None and buffers.normal.shape == (3, 64, 64, 3)
        assert buffers.mask is not None and buffers.mask.shape == (3, 64, 64)

    def test_single_view_promotion(self) -> None:
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        depth = np.zeros((64, 64), dtype=np.float32)
        normal = np.zeros((64, 64, 3), dtype=np.float32)
        mask = np.ones((64, 64), dtype=bool)

        buffers = ImageBuffers(rgba=rgba, depth=depth, normal=normal, mask=mask)
        assert len(buffers) == 1
        assert buffers.rgba is not None and buffers.rgba.shape == (1, 64, 64, 4)
        assert buffers.depth is not None and buffers.depth.shape == (1, 64, 64)
        assert buffers.normal is not None and buffers.normal.shape == (1, 64, 64, 3)
        assert buffers.mask is not None and buffers.mask.shape == (1, 64, 64)

    def test_sparse_buffers_allowed(self) -> None:
        depth = np.zeros((2, 32, 32), dtype=np.float32)
        buffers = ImageBuffers(depth=depth)
        assert len(buffers) == 2
        assert buffers.rgba is None
        assert buffers.normal is None
        assert buffers.mask is None

    def test_all_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="requires at least one non-null buffer"):
            ImageBuffers()

    def test_view_count_mismatch_raises(self) -> None:
        rgba = np.zeros((2, 64, 64, 4), dtype=np.uint8)
        depth = np.zeros((3, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match batch size"):
            ImageBuffers(rgba=rgba, depth=depth)

    def test_spatial_resolution_mismatch_raises(self) -> None:
        rgba = np.zeros((2, 64, 64, 4), dtype=np.uint8)
        depth = np.zeros((2, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match buffer resolution"):
            ImageBuffers(rgba=rgba, depth=depth)

    def test_indexing_and_slicing(self) -> None:
        rgba = np.zeros((4, 64, 64, 4), dtype=np.uint8)
        depth = np.zeros((4, 64, 64), dtype=np.float32)
        buffers = ImageBuffers(rgba=rgba, depth=depth)

        single = buffers[2]
        assert isinstance(single, ImageBuffers)
        assert len(single) == 1
        assert single.rgba is not None and single.rgba.shape == (1, 64, 64, 4)
        assert single.depth is not None and single.depth.shape == (1, 64, 64)
        assert single.normal is None

        sliced = buffers[1:3]
        assert len(sliced) == 2

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected dtype"):
            ImageBuffers(rgba=np.zeros((64, 64, 4), dtype=np.float32))

        with pytest.raises(TypeError, match=r"expected dtype"):
            ImageBuffers(depth=np.zeros((64, 64), dtype=np.int32))

        with pytest.raises(TypeError, match=r"expected dtype"):
            ImageBuffers(normal=np.zeros((64, 64, 3), dtype=np.int32))
