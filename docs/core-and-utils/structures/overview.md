# Data Structures Overview

`spatialhub.structures` defines the standardized return contracts shared across all SpatialHub model adapters. Every adapter returns one of these dataclasses, providing consistent attribute names, array shapes, and helper methods across all perception tasks.

```python
from spatialhub.structures import (
    MatchResult,
    DepthPredictionResult,
    FeatureExtractionResult,
    SegmentationResult,
    PoseEstimationResult,
)
```

---

## Return Contracts Summary

| Data Structure | Perception Task | Primary Producing Adapters | Key Fields |
| :--- | :--- | :--- | :--- |
| [**`MatchResult`**](match_result.md) | Semi-dense Feature Matching | [`EfficientLoFTR`](../../models/eloftr.md) | `keypoints_a`, `keypoints_b`, `confidence` |
| [**`DepthPredictionResult`**](depth_prediction_result.md) | Monocular & Multi-View Depth | [`DepthAnything3`](../../models/depthanything3.md) | `depth`, `conf`, `intrinsics`, `depth_type` |
| [**`FeatureExtractionResult`**](feature_extraction_result.md) | Feature Extraction & Embeddings | [`DINOv2`](../../models/dinov2.md) | `features`, `embedding_type`, `l2_normalized` |
| [**`SegmentationResult`**](segmentation_result.md) | Instance Segmentation & AMG | [`FastSAM`](../../models/fastsam.md), [`SAM`](../../models/sam.md), [`CNOS`](../../models/cnos.md) | `boxes`, `masks`, `scores`, `class_ids` |
| [**`PoseEstimationResult`**](pose_estimation_result.md) | 6D Object Pose Estimation & Tracking | [`FoundationPose`](../../models/foundationpose.md) | `poses`, `best_pose`, `best_score`, `bbox_3d` |

---

## Semantic Type Aliases

Array fields are typed using `typing.Annotated` aliases to document shape constraints, numerical data types, and coordinate conventions:

| Type Alias | Target Shape | Data Type | Coordinate & Numerical Convention |
| :--- | :--- | :--- | :--- |
| `Intrinsics3x3` | `(3, 3)` or `(N, 3, 3)` | `float32` | Pinhole camera intrinsic matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` in pixel units. |
| `CameraToWorld` | `(4, 4)` or `(N, 4, 4)` | `float32` | Rigid homogeneous $SE(3)$ camera-to-world transformation matrix. |
| `ObjectToCamera` | `(4, 4)` or `(N, 4, 4)` | `float32` | Rigid homogeneous $SE(3)$ object-to-camera transformation matrix. |
| `RigidTransform` | `(4, 4)` or `(N, 4, 4)` | `float32` | Rigid homogeneous $SE(3)$ transformation matrix. |
| `BoxCorners3D` | `(8, 3)` or `(N, 8, 3)` | `float32` | 3D bounding box corners in canonical mesh coordinate space. |
| `Boxes2D` | `(N, 4)` | `float32` | 2D bounding boxes in `[x1, y1, x2, y2]` pixel coordinates. |
| `Keypoints2D` | `(N, 2)` | `float32` | 2D keypoint pixel coordinates `[x, y]`. |
| `DepthMap` | `(N, H, W)` or `(H, W)` | `float32` | Dense depth map in meters (or relative scale). |
| `ConfidenceMap` | `(N, H, W)` or `(H, W)` | `float32` | Per-pixel confidence values. |
| `BinaryMask` | `(N, H, W)` | `bool` | Binary instance segmentation masks. |
| `ImageRGB` | `(H, W, 3)` or `(N, H, W, 3)` | `uint8` | RGB channel order, pixel values in $[0, 255]$. |
| `Scores` | `(N,)` | `float32` | Confidence scores in $[0.0, 1.0]$. |
| `FeatureEmbedding` | `(N, D)` or `(N, H, W, D)` | `float32` | Dense or global feature embedding vectors. |

---

## Architectural Principles

- **Plug-and-Play Pipeline Modularity:** Standardized return contracts decouple perception models from downstream spatial algorithms. Any model producing a `MatchResult` or `DepthPredictionResult` can be substituted into downstream geometric pipelines without modifying downstream code.
- **Pure NumPy Array Contracts:** All coordinate arrays, spatial masks, depth maps, and feature vectors are stored as contiguous NumPy arrays (`float32`, `bool`, or `uint8`).
- **Fail-Fast Shape Validation:** Every dataclass validates array types, dimensions, and shape invariants in `__post_init__` upon instantiation.
- **Batching & Dimension Normalization:** Single-sample predictions (such as a 2D depth map `(H, W)` or single `(4, 4)` pose) are automatically promoted to standardized batch dimensions on instantiation.
- **Decoupled Visualization:** Each result dataclass provides a `.visualize()` or `.visualize_mask()` convenience method that delegates directly to stateless visualization routines in [`spatialhub.utils.viz`](../viz.md).
