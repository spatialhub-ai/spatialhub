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
| [**`FeatureExtractionResult`**](feature_extraction_result.md) | Feature Extraction & Embeddings | [`DINOV2`](../../models/dinov2.md) | `features`, `embedding_type`, `l2_normalized` |
| [**`SegmentationResult`**](segmentation_result.md) | Instance Segmentation & AMG | [`FastSAM`](../../models/fastsam.md), [`SAM`](../../models/sam.md), [`CNOS`](../../models/cnos.md) | `boxes`, `masks`, `scores`, `class_ids` |
| [**`PoseEstimationResult`**](pose_estimation_result.md) | 6D Object Pose Estimation & Tracking | [`FoundationPose`](../../models/foundationpose.md) | `poses`, `best_pose`, `best_score`, `bbox_3d` |

---

## Architectural Principles

- **Pure NumPy Array Contracts:** All coordinate arrays, spatial masks, depth maps, and feature vectors are returned as contiguous NumPy arrays (`float32`, `bool`, or `uint8`).
- **Decoupled Visualization:** Each result dataclass provides a `.visualize()` or `.visualize_mask()` convenience method that delegates directly to the stateless drawing routines in [`spatialhub.utils.viz`](../viz.md).
- **Batching and Shape Normalization:** Singular outputs (such as a single $(4, 4)$ pose matrix) are automatically expanded to standardized batch dimensions on instantiation.
