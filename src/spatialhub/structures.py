from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from spatialhub.utils import draw_3d_axis, draw_projected_3d_box, visualize_masks, visualize_matches


@dataclass
class MatchResult:
    """
    A dataclass to hold the results of a feature matching operation.
    """
    image_a: str | Path | np.ndarray
    image_b: str | Path | np.ndarray
    keypoints_a: np.ndarray
    keypoints_b: np.ndarray
    confidence: np.ndarray

    def visualize(self, conf_thresh: float = 0.5, max_side: int = 800, top_k: int | None = None, save_path: str | Path | None = None):
        """
        Visualizes the top-k matches between two images using the provided keypoints and confidence scores.
        Only matches with confidence above the specified threshold will be displayed.
        """
        visualize_matches(
            self.image_a, 
            self.image_b, 
            self.keypoints_a, 
            self.keypoints_b, 
            self.confidence, 
            conf_thresh, 
            max_side,
            top_k,
            save_path
        )

@dataclass
class DepthPredictionResult:
    """
    Output of a depth estimation model.

    Attributes:
        depth: Depth maps of shape (N, H, W).
        conf: Optional confidence maps of shape (N, H, W).
        intrinsics: Optional camera intrinsics of shape (N, 3, 3).
        extrinsics: Optional camera extrinsics of shape (N, 4, 4).
        depth_type: Representation of the predicted depth.
    """

    image: np.ndarray
    depth: np.ndarray                  
    conf: np.ndarray | None = None     
    intrinsics: np.ndarray | None = None   
    extrinsics: np.ndarray | None = None   
    depth_type: Literal["metric", "relative", "inverse", "disparity"] = "metric"

@dataclass
class FeatureExtractionResult:
    """
    Result produced by feature extraction models.

    Supports both global image embeddings and dense per-pixel or per-patch feature representations.
    """

    images: np.ndarray
    features: np.ndarray
    embedding_type: Literal["global", "dense"] = "global"
    l2_normalized: bool = False

@dataclass
class SegmentationResult:
    """
    Result produced by instance segmentation / object detection / or zero-shot mask proposal models
    """
    image: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray | None = None
    class_names: list[str] | None = None

    def visualize_mask(self, save_path: str | Path | None = None):
        """
        Visualize Mask
        """

        visualize_masks(self.image, self.boxes, self.masks, self.scores, save_path=save_path)

@dataclass
class PoseEstimationResult:
    """
    Output of a 6D object pose estimation model (e.g. FoundationPose).

    Attributes:
        image: Original RGB input image of shape (H, W, 3).
        poses: Array of 4x4 object-to-camera transformation matrices of shape (N, 4, 4).
        intrinsics: Camera intrinsic matrix of shape (3, 3).
        scores: Optional confidence scores of shape (N,).
        labels: Optional class or model names for each detected pose of length N.
        bbox_3d: 3D bounding box corners of shape (N, 8, 3) or (8, 3) in canonical centered mesh space.
        to_origin: Mesh centering transformation matrix of shape (N, 4, 4) or (4, 4).
    """
    image: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray
    scores: np.ndarray | None = None
    labels: list[str] | None = None
    bbox_3d: np.ndarray | None = None
    to_origin: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Standardizes array dimensions for poses and scores."""
        if self.poses.ndim == 2:
            self.poses = self.poses[np.newaxis, ...]
        if self.scores is not None and self.scores.ndim == 0:
            self.scores = self.scores[np.newaxis, ...]
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]

    def __len__(self) -> int:
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
        """
        Visualizes the estimated 6D pose by projecting the 3D bounding box and coordinate axes.
        """
        vis_img = self.image.copy()

        for i, pose in enumerate(self.poses):
            if draw_bbox and self.bbox_3d is not None:
                bbox = (
                    self.bbox_3d[i]
                    if self.bbox_3d.ndim == 3 and len(self.bbox_3d) == len(self.poses)
                    else self.bbox_3d
                )
                vis_img = draw_projected_3d_box(
                    image=vis_img,
                    pose=pose,
                    intrinsics=self.intrinsics,
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
                    intrinsics=self.intrinsics,
                    axis_length=axis_length,
                    to_origin=to_orig,
                )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img

