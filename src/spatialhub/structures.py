from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np

from spatialhub.utils import draw_3d_axis, draw_3d_box, visualize_masks, visualize_matches

Intrinsics3x3 = Annotated[
    np.ndarray, "shape: (3, 3) | (N, 3, 3), dtype: float32, pinhole camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]"
]
CameraToWorld = Annotated[
    np.ndarray, "shape: (4, 4) | (N, 4, 4), dtype: float32, SE(3) camera-to-world rigid transform matrix"
]
ObjectToCamera = Annotated[
    np.ndarray, "shape: (4, 4) | (N, 4, 4), dtype: float32, SE(3) object-to-camera rigid transform matrix"
]
RigidTransform = Annotated[
    np.ndarray, "shape: (4, 4) | (N, 4, 4), dtype: float32, SE(3) homogeneous transform matrix"
]

BoxCorners3D = Annotated[
    np.ndarray, "shape: (8, 3) | (N, 8, 3), dtype: float32, 3D bounding box corners in canonical mesh space"
]
Boxes2D = Annotated[
    np.ndarray, "shape: (N, 4), dtype: float32, 2D bounding boxes in [x1, y1, x2, y2] pixel format"
]
Keypoints2D = Annotated[
    np.ndarray, "shape: (N, 2), dtype: float32, 2D keypoint coordinates [x, y] in pixel space"
]

DepthMap = Annotated[
    np.ndarray, "shape: (N, H, W) | (H, W), dtype: float32, depth maps in meters or relative scale"
]
ConfidenceMap = Annotated[
    np.ndarray, "shape: (N, H, W) | (H, W), dtype: float32, per-pixel prediction confidence"
]
BinaryMask = Annotated[
    np.ndarray, "shape: (N, H, W), dtype: bool, binary instance masks"
]

ImageRGB = Annotated[
    np.ndarray, "shape: (H, W, 3) | (N, H, W, 3), dtype: uint8, RGB channel order"
]
Scores = Annotated[
    np.ndarray, "shape: (N,), dtype: float32, confidence scores in [0.0, 1.0]"
]
FeatureEmbedding = Annotated[
    np.ndarray, "shape: (N, D) | (N, H, W, D) | (N, C, H, W), dtype: float32, feature embeddings"
]


def _check_array(
    name: str,
    arr: np.ndarray | None,
    *,
    ndim: int | tuple[int, ...] | None = None,
    shape_suffix: tuple[int, ...] | None = None,
    optional: bool = False,
) -> None:
    """Validates array type, ndim, and trailing dimensions.

    Args:
        name: Name of the attribute being validated.
        arr: Array instance or None.
        ndim: Expected number of dimensions or tuple of acceptable dimensions.
        shape_suffix: Expected trailing dimension tuple (e.g. (3, 3) or (4,)).
        optional: If True, allows None values without raising an error.

    Raises:
        TypeError: If arr is not an instance of np.ndarray and not None.
        ValueError: If ndim or shape_suffix does not match the array shape.
    """
    if arr is None:
        if not optional:
            raise ValueError(f"{name} is required and cannot be None")
        return

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")

    if ndim is not None:
        valid_ndim = (arr.ndim in ndim) if isinstance(ndim, tuple) else (arr.ndim == ndim)
        if not valid_ndim:
            raise ValueError(f"{name} expected ndim={ndim}, got ndim={arr.ndim} with shape {arr.shape}")

    if shape_suffix is not None:
        actual_suffix = arr.shape[-len(shape_suffix):]
        if actual_suffix != shape_suffix:
            raise ValueError(
                f"{name} trailing dimensions must be {shape_suffix}, got shape {arr.shape}"
            )


@dataclass
class MatchResult:
    """Dataclass holding the results of a 2D feature matching operation.

    Attributes:
        image_a: First input image path or RGB array of shape (H, W, 3).
        image_b: Second input image path or RGB array of shape (H, W, 3).
        keypoints_a: Matched keypoints in image_a of shape (N, 2) in (x, y) coordinates.
        keypoints_b: Matched keypoints in image_b of shape (N, 2) in (x, y) coordinates.
        confidence: Match confidence scores of shape (N,) in [0.0, 1.0].
    """

    image_a: str | Path | np.ndarray
    image_b: str | Path | np.ndarray
    keypoints_a: Keypoints2D
    keypoints_b: Keypoints2D
    confidence: Scores

    def __post_init__(self) -> None:
        """Validates keypoint coordinates and confidence array dimensions."""
        _check_array("keypoints_a", self.keypoints_a, ndim=2, shape_suffix=(2,))
        _check_array("keypoints_b", self.keypoints_b, ndim=2, shape_suffix=(2,))
        _check_array("confidence", self.confidence, ndim=1)
        if not (len(self.keypoints_a) == len(self.keypoints_b) == len(self.confidence)):
            raise ValueError(
                f"Array length mismatch: keypoints_a ({len(self.keypoints_a)}), "
                f"keypoints_b ({len(self.keypoints_b)}), confidence ({len(self.confidence)})"
            )

    def visualize(
        self,
        conf_thresh: float = 0.5,
        max_side: int = 800,
        top_k: int | None = None,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualizes top-k matches between two images using keypoints and confidence scores.

        Args:
            conf_thresh: Minimum confidence score threshold.
            max_side: Maximum canvas dimension in pixels.
            top_k: Optional limit on the number of matches to display.
            save_path: Optional output path to write the annotated image.

        Returns:
            Visualization canvas as a uint8 BGR array.
        """
        return visualize_matches(
            self.image_a,
            self.image_b,
            self.keypoints_a,
            self.keypoints_b,
            self.confidence,
            conf_thresh,
            max_side,
            top_k,
            save_path,
        )


@dataclass
class DepthPredictionResult:
    """Output of a monocular or multi-view depth estimation model.

    Attributes:
        image: Original input image array, path, or batch list.
        depth: Depth maps of shape (N, H, W) in meters or relative scale.
        conf: Optional per-pixel confidence maps of shape (N, H, W).
        intrinsics: Optional pinhole camera intrinsic matrices of shape (N, 3, 3).
        extrinsics: Optional camera-to-world rigid transforms of shape (N, 4, 4).
        depth_type: Representation scale of the predicted depth.
    """

    image: np.ndarray | str | Path | list[np.ndarray | str | Path]
    depth: DepthMap
    conf: ConfidenceMap | None = None
    intrinsics: Intrinsics3x3 | None = None
    extrinsics: CameraToWorld | None = None
    depth_type: Literal["metric", "relative", "inverse", "disparity"] = "metric"

    def __post_init__(self) -> None:
        """Normalizes and validates depth maps and camera matrices."""
        _check_array("depth", self.depth, ndim=(2, 3))
        if self.depth.ndim == 2:
            self.depth = self.depth[np.newaxis, ...]

        if self.conf is not None:
            _check_array("conf", self.conf, ndim=(2, 3))
            if self.conf.ndim == 2:
                self.conf = self.conf[np.newaxis, ...]
            if self.conf.shape != self.depth.shape:
                raise ValueError(
                    f"conf shape {self.conf.shape} must match depth shape {self.depth.shape}"
                )

        if self.intrinsics is not None:
            _check_array("intrinsics", self.intrinsics, ndim=(2, 3), shape_suffix=(3, 3))
            if self.intrinsics.ndim == 2:
                self.intrinsics = self.intrinsics[np.newaxis, ...]

        if self.extrinsics is not None:
            _check_array("extrinsics", self.extrinsics, ndim=(2, 3), shape_suffix=(4, 4))
            if self.extrinsics.ndim == 2:
                self.extrinsics = self.extrinsics[np.newaxis, ...]


@dataclass
class FeatureExtractionResult:
    """Result produced by dense or global feature representation models.

    Attributes:
        images: Original input image array or batched images.
        features: Feature embedding tensor of shape (N, D) for global embeddings,
            or (N, H, W, D) / (N, C, H, W) for dense embeddings.
        embedding_type: Representation type ('global' or 'dense').
        l2_normalized: Whether embeddings are L2-normalized.
    """

    images: np.ndarray | list[np.ndarray | str | Path] | str | Path
    features: FeatureEmbedding
    embedding_type: Literal["global", "dense"] = "global"
    l2_normalized: bool = False

    def __post_init__(self) -> None:
        """Validates feature embedding array dimensions."""
        _check_array("features", self.features, ndim=(2, 3, 4))
        if self.embedding_type == "global" and self.features.ndim != 2:
            raise ValueError(
                f"Global feature embeddings must have ndim=2 (N, D), got shape {self.features.shape}"
            )


@dataclass
class SegmentationResult:
    """Result produced by instance segmentation or zero-shot mask proposal models.

    Attributes:
        image: Original RGB input image of shape (H, W, 3).
        boxes: Bounding boxes of shape (N, 4) in [x1, y1, x2, y2] coordinate format.
        masks: Binary masks of shape (N, H, W) as boolean array.
        scores: Confidence scores of shape (N,) as float array.
        class_ids: Optional class identifier indices of shape (N,).
        class_names: Optional class label names of length N.
    """

    image: ImageRGB
    boxes: Boxes2D
    masks: BinaryMask
    scores: Scores
    class_ids: np.ndarray | None = None
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validates segmentation arrays and coordinate bounds."""
        _check_array("image", self.image, ndim=3)
        _check_array("boxes", self.boxes, ndim=2, shape_suffix=(4,))
        _check_array("masks", self.masks, ndim=3)
        _check_array("scores", self.scores, ndim=1)

        num_boxes = len(self.boxes)
        if len(self.masks) != num_boxes or len(self.scores) != num_boxes:
            raise ValueError(
                f"Array length mismatch: boxes ({num_boxes}), "
                f"masks ({len(self.masks)}), scores ({len(self.scores)})"
            )

        if self.class_ids is not None:
            _check_array("class_ids", self.class_ids, ndim=1)
            if len(self.class_ids) != num_boxes:
                raise ValueError(
                    f"class_ids length ({len(self.class_ids)}) does not match boxes ({num_boxes})"
                )

        if self.class_names is not None and len(self.class_names) != num_boxes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) does not match boxes ({num_boxes})"
            )

    def visualize_mask(self, save_path: str | Path | None = None) -> np.ndarray:
        """Visualizes instance segmentation masks overlaid on the input image.

        Args:
            save_path: Optional output path to write the annotated image.

        Returns:
            Annotated RGB image array of shape (H, W, 3).
        """
        return visualize_masks(self.image, self.boxes, self.masks, self.scores, save_path=save_path)


@dataclass
class PoseEstimationResult:
    """Output of a 6D rigid object pose estimation model.

    Attributes:
        image: Original RGB input image of shape (H, W, 3).
        poses: Array of 4x4 object-to-camera transformation matrices of shape (N, 4, 4).
        intrinsics: Camera intrinsic matrix of shape (3, 3) or (N, 3, 3).
        scores: Optional confidence scores of shape (N,).
        labels: Optional class or model names for each detected pose of length N.
        bbox_3d: 3D bounding box corners of shape (N, 8, 3) or (8, 3) in canonical centered mesh space.
        to_origin: Mesh centering transformation matrix of shape (N, 4, 4) or (4, 4).
    """

    image: ImageRGB
    poses: ObjectToCamera
    intrinsics: Intrinsics3x3
    scores: Scores | None = None
    labels: list[str] | None = None
    bbox_3d: BoxCorners3D | None = None
    to_origin: RigidTransform | None = None

    def __post_init__(self) -> None:
        """Standardizes array dimensions for poses, intrinsics, and scores."""
        _check_array("image", self.image, ndim=3)
        _check_array("poses", self.poses, ndim=(2, 3), shape_suffix=(4, 4))
        if self.poses.ndim == 2:
            self.poses = self.poses[np.newaxis, ...]

        _check_array("intrinsics", self.intrinsics, ndim=(2, 3), shape_suffix=(3, 3))

        if self.scores is not None:
            if isinstance(self.scores, (int, float)):
                self.scores = np.array([self.scores], dtype=np.float32)
            else:
                _check_array("scores", self.scores, ndim=(0, 1))
                if self.scores.ndim == 0:
                    self.scores = self.scores[np.newaxis, ...]
            if len(self.scores) != len(self.poses):
                raise ValueError(
                    f"scores count ({len(self.scores)}) does not match poses count ({len(self.poses)})"
                )

        if self.labels is not None:
            if isinstance(self.labels, str):
                self.labels = [self.labels]
            if len(self.labels) != len(self.poses):
                raise ValueError(
                    f"labels count ({len(self.labels)}) does not match poses count ({len(self.poses)})"
                )

        if self.bbox_3d is not None:
            _check_array("bbox_3d", self.bbox_3d, ndim=(2, 3), shape_suffix=(8, 3))

        if self.to_origin is not None:
            _check_array("to_origin", self.to_origin, ndim=(2, 3), shape_suffix=(4, 4))

    def __len__(self) -> int:
        """Returns the number of estimated poses."""
        return len(self.poses)

    @property
    def best_pose(self) -> np.ndarray:
        """Returns the single top-scoring 4x4 pose matrix."""
        if len(self.poses) == 1 or self.scores is None:
            return self.poses[0]
        return self.poses[int(np.argmax(self.scores))]

    @property
    def best_score(self) -> float | None:
        """Returns the highest confidence score."""
        if self.scores is None or len(self.scores) == 0:
            return None
        return float(np.max(self.scores))

    def visualize(
        self,
        draw_bbox: bool = True,
        draw_axes: bool = True,
        axis_length: float = 0.05,
        box_color: tuple[int, int, int] = (0, 255, 0),
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualizes estimated 6D poses by projecting 3D bounding boxes and coordinate axes.

        Args:
            draw_bbox: Whether to draw projected 3D bounding box wireframes.
            draw_axes: Whether to draw 3D coordinate axes.
            axis_length: Length of 3D axes in meters.
            box_color: RGB tuple for bounding box edges.
            save_path: Optional output path to write the annotated image.

        Returns:
            Annotated RGB image array of shape (H, W, 3).
        """
        vis_img = self.image.copy()

        intr = self.intrinsics[0] if self.intrinsics.ndim == 3 else self.intrinsics

        for i, pose in enumerate(self.poses):
            if draw_bbox and self.bbox_3d is not None:
                bbox = (
                    self.bbox_3d[i]
                    if self.bbox_3d.ndim == 3 and len(self.bbox_3d) == len(self.poses)
                    else self.bbox_3d
                )
                vis_img = draw_3d_box(
                    image=vis_img,
                    pose=pose,
                    intrinsics=intr,
                    bbox_corners_3d=bbox,
                    color=box_color,
                )
            if draw_axes and self.to_origin is not None:
                to_orig = (
                    self.to_origin[i]
                    if self.to_origin.ndim == 3 and len(self.to_origin) == len(self.poses)
                    else self.to_origin
                )
                vis_img = draw_3d_axis(
                    image=vis_img,
                    pose=pose,
                    intrinsics=intr,
                    axis_length=axis_length,
                    to_origin=to_orig,
                )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img

