"""Standardized spatial and geometric data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from .utils.validation import validate_array


@dataclass
class _BatchedArray:
    """Base container for batched arrays with fixed trailing dimensions.

    Attributes:
        _array_field: Name of the primary array field.
        _shape_suffix: Expected trailing dimension tuple.
    """

    _array_field: ClassVar[str] = "array"
    _shape_suffix: ClassVar[tuple[int, ...]] = ()

    def __post_init__(self) -> None:
        """Validate array dimensions and standardize the leading batch dimension."""
        arr = getattr(self, self._array_field)
        validate_array(self._array_field, arr, ndim=(2, 3), shape_suffix=self._shape_suffix, dtype=np.floating,)
        if arr.ndim == 2:
            setattr(self, self._array_field, arr[np.newaxis, ...])

    def __len__(self) -> int:
        """Return the batch size along the leading dimension."""
        return len(getattr(self, self._array_field))

    def __getitem__(self, index: int | slice | np.ndarray):
        """Slice the array along the leading batch dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new instance of the same class containing the sliced array.
        """
        arr = getattr(self, self._array_field)
        sliced = arr[index]
        if isinstance(index, int):
            sliced = sliced[np.newaxis, ...]
        return self.__class__(**{self._array_field: sliced})


@dataclass
class RigidTransform3D(_BatchedArray):
    """Rigid 3D transformation matrices in SE(3).

    Attributes:
        matrix: Homogeneous transformation matrices of shape (N, 4, 4) mapping
            3D homogeneous coordinates between reference frames.
    """

    matrix: np.ndarray
    _array_field: ClassVar[str] = "matrix"
    _shape_suffix: ClassVar[tuple[int, ...]] = (4, 4)


@dataclass
class CameraIntrinsics(_BatchedArray):
    """Camera intrinsic calibration matrices.

    Attributes:
        matrix: Calibration matrices of shape (N, 3, 3) mapping 3D camera-frame
            coordinates to 2D homogeneous pixel coordinates.
    """

    matrix: np.ndarray
    _array_field: ClassVar[str] = "matrix"
    _shape_suffix: ClassVar[tuple[int, ...]] = (3, 3)


@dataclass
class CameraExtrinsics(RigidTransform3D):
    """Rigid transformation matrices defining camera pose in SE(3).

    Attributes:
        matrix: Homogeneous transformation matrices of shape (N, 4, 4) mapping
            coordinates between camera space and an external reference frame.
    """

    pass


@dataclass
class BoundingBox3D(_BatchedArray):
    """3D bounding box corner vertices in a 3D coordinate frame.

    Attributes:
        corners: Bounding box corner coordinates of shape (N, 8, 3) defining
            eight 3D vertices per box.
    """

    corners: np.ndarray
    _array_field: ClassVar[str] = "corners"
    _shape_suffix: ClassVar[tuple[int, ...]] = (8, 3)


@dataclass
class KeypointMatches:
    """Pairwise 2D keypoint correspondences between two views.

    Attributes:
        keypoints_a: Matched keypoints in view A of shape (N, 2) in (x, y) coordinates.
        keypoints_b: Matched keypoints in view B of shape (N, 2) in (x, y) coordinates.
        confidence: Match confidence scores of shape (N,) in [0.0, 1.0].
    """

    keypoints_a: np.ndarray
    keypoints_b: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        """Validate keypoint coordinates, confidence array dimensions, and dtypes."""
        validate_array("keypoints_a", self.keypoints_a, ndim=2, shape_suffix=(2,), dtype=np.floating)
        validate_array("keypoints_b", self.keypoints_b, ndim=2, shape_suffix=(2,), dtype=np.floating)
        validate_array("confidence", self.confidence, ndim=1, dtype=np.floating)
        if not (len(self.keypoints_a) == len(self.keypoints_b) == len(self.confidence)):
            raise ValueError(
                f"Array length mismatch: keypoints_a ({len(self.keypoints_a)}), "
                f"keypoints_b ({len(self.keypoints_b)}), confidence ({len(self.confidence)})"
            )

    def __len__(self) -> int:
        """Return the number of matched keypoint pairs."""
        return len(self.confidence)

    def __getitem__(self, index: int | slice | np.ndarray) -> KeypointMatches:
        """Slice keypoints and confidence scores along the correspondence dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new KeypointMatches instance with sliced correspondences.
        """
        kpts_a = self.keypoints_a[index]
        kpts_b = self.keypoints_b[index]
        conf = self.confidence[index]

        if isinstance(index, int):
            kpts_a = kpts_a[np.newaxis, ...]
            kpts_b = kpts_b[np.newaxis, ...]
            conf = conf[np.newaxis, ...]

        return KeypointMatches(
            keypoints_a=kpts_a,
            keypoints_b=kpts_b,
            confidence=conf,
        )


@dataclass
class DepthMap:
    """Estimated depth field in camera coordinate frame.

    Attributes:
        depth: Depth maps of shape (N, H, W) in meters or relative scale.
        confidence: Optional per-pixel confidence maps of shape (N, H, W).
        depth_type: Representation scale of the predicted depth.
        intrinsics: Optional camera intrinsic calibration matrices.
        extrinsics: Optional camera extrinsic transformation matrices in SE(3).
    """

    depth: np.ndarray
    confidence: np.ndarray | None = None
    depth_type: Literal["metric", "relative", "inverse", "disparity"] = "metric"
    intrinsics: CameraIntrinsics | np.ndarray | None = None
    extrinsics: CameraExtrinsics | np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array dimensions, dtypes, and standardize the leading batch dimension."""
        validate_array("depth", self.depth, ndim=(2, 3), dtype=np.floating)
        if self.depth.ndim == 2:
            self.depth = self.depth[np.newaxis, ...]

        if self.confidence is not None:
            validate_array("confidence", self.confidence, ndim=(2, 3), dtype=np.floating)
            if self.confidence.ndim == 2:
                self.confidence = self.confidence[np.newaxis, ...]
            if self.confidence.shape != self.depth.shape:
                raise ValueError(
                    f"confidence shape {self.confidence.shape} must match depth shape {self.depth.shape}"
                )

        if self.intrinsics is not None:
            if isinstance(self.intrinsics, np.ndarray):
                self.intrinsics = CameraIntrinsics(self.intrinsics)
            elif not isinstance(self.intrinsics, CameraIntrinsics):
                raise TypeError(
                    f"intrinsics must be CameraIntrinsics, np.ndarray, or None, got {type(self.intrinsics).__name__}"
                )
            if len(self.intrinsics) != len(self.depth):
                raise ValueError(
                    f"intrinsics batch size ({len(self.intrinsics)}) must match depth batch size ({len(self.depth)})"
                )

        if self.extrinsics is not None:
            if isinstance(self.extrinsics, np.ndarray):
                self.extrinsics = CameraExtrinsics(self.extrinsics)
            elif not isinstance(self.extrinsics, CameraExtrinsics):
                raise TypeError(
                    f"extrinsics must be CameraExtrinsics, np.ndarray, or None, got {type(self.extrinsics).__name__}"
                )
            if len(self.extrinsics) != len(self.depth):
                raise ValueError(
                    f"extrinsics batch size ({len(self.extrinsics)}) must match depth batch size ({len(self.depth)})"
                )

    def __len__(self) -> int:
        """Return the batch size along the leading dimension."""
        return len(self.depth)

    def __getitem__(self, index: int | slice | np.ndarray) -> DepthMap:
        """Slice depth, confidence, and camera geometry along the leading batch dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new DepthMap instance containing the sliced arrays.
        """
        sliced_depth = self.depth[index]
        if isinstance(index, int):
            sliced_depth = sliced_depth[np.newaxis, ...]

        sliced_conf = None
        if self.confidence is not None:
            sliced_conf = self.confidence[index]
            if isinstance(index, int):
                sliced_conf = sliced_conf[np.newaxis, ...]

        sliced_intrinsics = self.intrinsics[index] if self.intrinsics is not None else None
        sliced_extrinsics = self.extrinsics[index] if self.extrinsics is not None else None

        return DepthMap(
            depth=sliced_depth,
            confidence=sliced_conf,
            depth_type=self.depth_type,
            intrinsics=sliced_intrinsics,
            extrinsics=sliced_extrinsics,
        )


@dataclass
class FeatureEmbedding:
    """Batched global or dense feature representations.

    Attributes:
        features: Feature embedding tensor of shape (N, D) for global embeddings,
            or (N, H, W, D) for dense embeddings.
        embedding_type: Representation type ('global' or 'dense').
        l2_normalized: Whether embeddings are unit L2-normalized.
    """

    features: np.ndarray
    embedding_type: Literal["global", "dense"] = "global"
    l2_normalized: bool = False

    def __post_init__(self) -> None:
        """Validate feature embedding array dimensions and dtypes."""
        validate_array("features", self.features, ndim=(2, 3, 4), dtype=np.floating)
        if self.embedding_type == "global" and self.features.ndim != 2:
            raise ValueError(
                f"Global feature embeddings must have ndim=2 (N, D), got shape {self.features.shape}"
            )

        if self.embedding_type == "dense" and self.features.ndim != 4:
            raise ValueError(
                f"Dense feature embeddings must have ndim=4 (N, H, W, D), got shape {self.features.shape}"
            )

    def __len__(self) -> int:
        """Return the batch size along the leading dimension."""
        return len(self.features)

    def __getitem__(self, index: int | slice | np.ndarray) -> FeatureEmbedding:
        """Slice feature embeddings along the leading batch dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new FeatureEmbedding instance containing sliced features.
        """
        sliced_features = self.features[index]

        if isinstance(index, int):
            sliced_features = sliced_features[np.newaxis, ...]

        return FeatureEmbedding(
            features=sliced_features,
            embedding_type=self.embedding_type,
            l2_normalized=self.l2_normalized,
        )


@dataclass
class InstanceSegmentation:
    """Detected 2D object instances with binary masks and bounding boxes.

    Attributes:
        masks: Binary masks of shape (N, H, W) as boolean or uint8 array.
        boxes: Bounding boxes of shape (N, 4) in [x1, y1, x2, y2] coordinate format.
        scores: Confidence scores of shape (N,) as float array in [0.0, 1.0].
        ids: Optional class or instance identifier indices of shape (N,).
        names: Optional class label names of length N.
    """

    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    ids: np.ndarray | None = None
    names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate segmentation arrays, coordinate bounds, dtypes, and standardize batch dimensions."""
        validate_array("masks", self.masks, ndim=(2, 3), dtype=(np.bool_, np.integer))
        if self.masks.ndim == 2:
            self.masks = self.masks[np.newaxis, ...]

        validate_array("boxes", self.boxes, ndim=(1, 2), shape_suffix=(4,), dtype=(np.floating, np.integer))
        if self.boxes.ndim == 1:
            self.boxes = self.boxes[np.newaxis, ...]

        if isinstance(self.scores, (int, float, np.generic)):
            self.scores = np.asarray([self.scores], dtype=np.float32)
        else:
            validate_array("scores", self.scores, ndim=(0, 1), dtype=np.floating)
            if self.scores.ndim == 0:
                self.scores = self.scores[np.newaxis, ...]

        num_boxes = len(self.boxes)
        if len(self.masks) != num_boxes or len(self.scores) != num_boxes:
            raise ValueError(
                f"Array length mismatch: boxes ({num_boxes}), "
                f"masks ({len(self.masks)}), scores ({len(self.scores)})"
            )

        if self.ids is not None:
            if isinstance(self.ids, (int, np.generic)):
                self.ids = np.asarray([self.ids], dtype=np.int64)
            else:
                validate_array("ids", self.ids, ndim=(0, 1), dtype=np.integer)
                if self.ids.ndim == 0:
                    self.ids = self.ids[np.newaxis, ...]
            if len(self.ids) != num_boxes:
                raise ValueError(
                    f"ids length ({len(self.ids)}) does not match boxes ({num_boxes})"
                )

        if self.names is not None and len(self.names) != num_boxes:
            raise ValueError(
                f"names length ({len(self.names)}) does not match boxes ({num_boxes})"
            )

    def __len__(self) -> int:
        """Return the number of detected instances."""
        return len(self.boxes)

    def __getitem__(self, index: int | slice | np.ndarray) -> InstanceSegmentation:
        """Slice instances across masks, boxes, scores, ids, and names.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new InstanceSegmentation instance containing sliced detections.
        """
        sliced_names = None
        if self.names is not None:
            if isinstance(index, int):
                sliced_names = [self.names[index]]
            elif isinstance(index, slice):
                sliced_names = self.names[index]
            elif isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.bool_):
                sliced_names = [name for name, keep in zip(self.names, index) if keep]
            elif isinstance(index, list) and len(index) > 0 and isinstance(index[0], (bool, np.bool_)):
                sliced_names = [name for name, keep in zip(self.names, index) if keep]
            elif isinstance(index, (list, np.ndarray)):
                sliced_names = [self.names[int(i)] for i in index]

        sliced_masks = self.masks[index]
        if isinstance(index, int):
            sliced_masks = sliced_masks[np.newaxis, ...]

        sliced_boxes = self.boxes[index]
        if isinstance(index, int):
            sliced_boxes = sliced_boxes[np.newaxis, ...]

        sliced_scores = self.scores[index]
        if isinstance(index, int):
            sliced_scores = sliced_scores[np.newaxis, ...]

        sliced_ids = None
        if self.ids is not None:
            sliced_ids = self.ids[index]
            if isinstance(index, int):
                sliced_ids = np.asarray([sliced_ids])

        return InstanceSegmentation(
            masks=sliced_masks,
            boxes=sliced_boxes,
            scores=sliced_scores,
            ids=sliced_ids,
            names=sliced_names,
        )


@dataclass
class ObjectPose6D:
    """Estimated 6D rigid object poses in SE(3).

    Attributes:
        pose: Array of 4x4 rigid transformation matrices of shape (N, 4, 4)
            mapping 3D coordinates from an object's reference frame into a target
            reference frame.
        scores: Optional confidence scores of shape (N,).
        ids: Optional class or instance identifier indices of shape (N,).
        names: Optional class or model names for each detected pose of length N.
    """

    pose: np.ndarray
    scores: np.ndarray | None = None
    ids: np.ndarray | None = None
    names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate pose matrix dimensions, dtypes, and standardize the leading batch dimension."""
        validate_array("pose", self.pose, ndim=(2, 3), shape_suffix=(4, 4), dtype=np.floating)
        if self.pose.ndim == 2:
            self.pose = self.pose[np.newaxis, ...]

        num_poses = len(self.pose)

        if self.scores is not None:
            if isinstance(self.scores, (int, float, np.generic)):
                self.scores = np.asarray([self.scores], dtype=np.float32)
            else:
                validate_array("scores", self.scores, ndim=(0, 1), dtype=np.floating)
                if self.scores.ndim == 0:
                    self.scores = self.scores[np.newaxis, ...]
            if len(self.scores) != num_poses:
                raise ValueError(
                    f"scores count ({len(self.scores)}) does not match pose count ({num_poses})"
                )

        if self.ids is not None:
            if isinstance(self.ids, (int, np.generic)):
                self.ids = np.asarray([self.ids], dtype=np.int64)
            else:
                validate_array("ids", self.ids, ndim=(0, 1), dtype=np.integer)
                if self.ids.ndim == 0:
                    self.ids = self.ids[np.newaxis, ...]
            if len(self.ids) != num_poses:
                raise ValueError(
                    f"ids count ({len(self.ids)}) does not match pose count ({num_poses})"
                )

        if self.names is not None:
            if isinstance(self.names, str):
                self.names = [self.names]
            if len(self.names) != num_poses:
                raise ValueError(
                    f"names count ({len(self.names)}) does not match pose count ({num_poses})"
                )

    def __len__(self) -> int:
        """Return the number of estimated poses."""
        return len(self.pose)

    def __getitem__(self, index: int | slice | np.ndarray) -> ObjectPose6D:
        """Slice poses, scores, ids, and names along the instance dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new ObjectPose6D instance containing sliced poses and attributes.
        """
        sliced_pose = self.pose[index]
        if isinstance(index, int):
            sliced_pose = sliced_pose[np.newaxis, ...]

        sliced_scores = None
        if self.scores is not None:
            sliced_scores = self.scores[index]
            if isinstance(index, int):
                sliced_scores = sliced_scores[np.newaxis, ...]

        sliced_ids = None
        if self.ids is not None:
            sliced_ids = self.ids[index]
            if isinstance(index, int):
                sliced_ids = np.asarray([sliced_ids])

        sliced_names = None
        if self.names is not None:
            if isinstance(index, int):
                sliced_names = [self.names[index]]
            elif isinstance(index, slice):
                sliced_names = self.names[index]
            elif isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.bool_):
                sliced_names = [name for name, keep in zip(self.names, index) if keep]
            elif isinstance(index, list) and len(index) > 0 and isinstance(index[0], (bool, np.bool_)):
                sliced_names = [name for name, keep in zip(self.names, index) if keep]
            elif isinstance(index, (list, np.ndarray)):
                sliced_names = [self.names[int(i)] for i in index]

        return ObjectPose6D(
            pose=sliced_pose,
            scores=sliced_scores,
            ids=sliced_ids,
            names=sliced_names,
        )


@dataclass
class ImageBuffers:
    """Multimodal 2D image buffers aligned on the image plane.

    Attributes:
        rgba: Optional color images with alpha channel of shape (N, H, W, 4).
        depth: Optional depth maps of shape (N, H, W).
        normal: Optional surface normal vector maps of shape (N, H, W, 3).
        mask: Optional binary foreground masks of shape (N, H, W).
    """

    rgba: np.ndarray | None = None
    depth: np.ndarray | None = None
    normal: np.ndarray | None = None
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate buffer dimensions, dtypes, and standardize leading batch dimensions."""
        buffers = [b for b in (self.rgba, self.depth, self.normal, self.mask) if b is not None]
        if not buffers:
            raise ValueError("ImageBuffers requires at least one non-null buffer.")

        num_views: int | None = None
        spatial_hw: tuple[int, int] | None = None

        if self.rgba is not None:
            validate_array("rgba", self.rgba, ndim=(3, 4), shape_suffix=(4,), dtype=(np.uint8, np.integer))
            if self.rgba.ndim == 3:
                self.rgba = self.rgba[np.newaxis, ...]
            num_views = len(self.rgba)
            spatial_hw = self.rgba.shape[1:3]

        if self.depth is not None:
            validate_array("depth", self.depth, ndim=(2, 3), dtype=np.floating)
            if self.depth.ndim == 2:
                self.depth = self.depth[np.newaxis, ...]
            if num_views is None:
                num_views = len(self.depth)
            elif len(self.depth) != num_views:
                raise ValueError(
                    f"depth count ({len(self.depth)}) does not match batch size ({num_views})"
                )
            if spatial_hw is None:
                spatial_hw = self.depth.shape[1:3]
            elif self.depth.shape[1:3] != spatial_hw:
                raise ValueError(
                    f"depth resolution {self.depth.shape[1:3]} does not match buffer resolution {spatial_hw}"
                )

        if self.normal is not None:
            validate_array("normal", self.normal, ndim=(3, 4), shape_suffix=(3,), dtype=np.floating)
            if self.normal.ndim == 3:
                self.normal = self.normal[np.newaxis, ...]
            if num_views is None:
                num_views = len(self.normal)
            elif len(self.normal) != num_views:
                raise ValueError(
                    f"normal count ({len(self.normal)}) does not match batch size ({num_views})"
                )
            if spatial_hw is None:
                spatial_hw = self.normal.shape[1:3]
            elif self.normal.shape[1:3] != spatial_hw:
                raise ValueError(
                    f"normal resolution {self.normal.shape[1:3]} does not match buffer resolution {spatial_hw}"
                )

        if self.mask is not None:
            validate_array("mask", self.mask, ndim=(2, 3), dtype=(np.bool_, np.integer))
            if self.mask.ndim == 2:
                self.mask = self.mask[np.newaxis, ...]
            if num_views is None:
                num_views = len(self.mask)
            elif len(self.mask) != num_views:
                raise ValueError(
                    f"mask count ({len(self.mask)}) does not match batch size ({num_views})"
                )
            if spatial_hw is None:
                spatial_hw = self.mask.shape[1:3]
            elif self.mask.shape[1:3] != spatial_hw:
                raise ValueError(
                    f"mask resolution {self.mask.shape[1:3]} does not match buffer resolution {spatial_hw}"
                )

    def __len__(self) -> int:
        """Return the number of frames along the leading batch dimension."""
        for buf in (self.rgba, self.depth, self.normal, self.mask):
            if buf is not None:
                return len(buf)
        return 0

    def __getitem__(self, index: int | slice | np.ndarray) -> ImageBuffers:
        """Slice all present image buffers along the leading batch dimension.

        Args:
            index: Integer index, slice, or boolean/integer array.

        Returns:
            A new ImageBuffers instance containing sliced buffers.
        """
        def _slice_opt(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            sub = arr[index]
            if isinstance(index, int):
                sub = sub[np.newaxis, ...]
            return sub

        return ImageBuffers(
            rgba=_slice_opt(self.rgba),
            depth=_slice_opt(self.depth),
            normal=_slice_opt(self.normal),
            mask=_slice_opt(self.mask),
        )


__all__ = [
    "BoundingBox3D",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "DepthMap",
    "FeatureEmbedding",
    "ImageBuffers",
    "InstanceSegmentation",
    "KeypointMatches",
    "ObjectPose6D",
    "RigidTransform3D",
]
