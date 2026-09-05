# FoundationPose Technical Reference

`spatialhub.models.foundationpose` provides an ONNX Runtime adapter for **FoundationPose**, a model-based 6D object pose estimation and tracking pipeline for novel and known objects using RGB-D observations.

---

## Supported Model Components

FoundationPose operates through two specialized ONNX neural network models:

| Component | ONNX File | Default Hugging Face Source | Description |
| :--- | :--- | :--- | :--- |
| **RefineNet** (`refine_net`) | `refine_net.onnx` | `SpatialHub/foundationpose` | Iteratively predicts egocentric $SE(3)$ transformation updates $(\Delta R, \Delta t)$ given rendered synthetic vs. observed RGB-D crops. |
| **ScoreNet** (`score_net`) | `score_net.onnx` | `SpatialHub/foundationpose` | Evaluates candidate pose hypotheses via pairwise tournament selection to rank and score estimated 6D poses. |

---

## Overview & Execution Pipeline

FoundationPose performs model-based 6D pose estimation through iterative pose refinement and candidate hypothesis evaluation.

### Iterative Pose Refinement (`PoseRefinePredictor`)

At each iteration $k$, the candidate pose $T^{(k)} \in SE(3)$ is used to render synthetic RGB and 3D coordinate templates $(A)$ using ModernGL. The real observed RGB-D crops $(B)$ and synthetic templates $(A)$ are fed into `RefineNet` to regress egocentric delta poses $(\Delta r, \Delta t)$:

$$T^{(k+1)} = \Delta T \cdot T^{(k)}$$

where $\Delta T = [ \mathbf{R}(\Delta r) \mid \Delta t ]$, converting the predicted rotation vector $\Delta r \in \mathbb{R}^3$ to a rotation matrix $\mathbf{R}(\Delta r)$.

### Tournament Hypothesis Scoring (`ScorePredictor`)

Candidate pose hypotheses are evaluated in batched tournament comparisons using `ScoreNet`. Pairwise score logits determine progressive candidate elimination until the global top-scoring pose is identified:

$$\text{score}^* = \arg\max_i S_i$$

### Headless ModernGL Rendering & Depth Filtering

- **Batched Rendering (`Renderer`):** Renders $N$ candidate viewpoints into multi-target G-Buffer attachments (RGBA + XYZ coordinates) in a single instanced draw call.
- **Depth Filtering (`DepthFilter`):** Applies GPU-accelerated morphological erosion and bilateral smoothing shader passes over observed depth maps to mitigate sensor noise and edge artifacts.

---

## SpatialHub Adapter API & Usage

### 6D Pose Registration

```python
import numpy as np
from spatialhub import FoundationPose

# Camera intrinsic matrix
K = np.array([
    [572.41,   0.0, 325.26],
    [  0.0, 573.57, 242.04],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

# Initialize FoundationPose adapter
with FoundationPose(
    mesh_file_path="cad_models/object.ply",
    camera_intrinsic=K,
    model_unit="mm",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
) as engine:
    # Estimate 6D pose from RGB, Depth, and 2D Segmentation Mask
    result = engine.register(
        rgb="scene_rgb.png",
        depth="scene_depth.png",  # or .npy / 16-bit PNG (mm)
        mask="object_mask.png",
        iteration=5,
    )

    print(f"Estimated Pose Matrix:\n{result.best_pose}")
    print(f"Confidence Score: {result.best_score}")

    # Visualize 3D bounding box and coordinate axes
    result.visualize(draw_bbox=True, draw_axes=True, save_path="pose_estimation.png")
```

### 6D Pose Tracking

```python
# Track pose in subsequent video frame
tracked_result = engine.track(
    rgb="frame_0002_rgb.png",
    depth="frame_0002_depth.png",
    previous_pose=result.best_pose,
    iteration=2,
)

tracked_result.visualize(save_path="tracked_pose.png")
```

---

## Returned Result Data Structure

Returns a [`PoseEstimationResult`](../../docs/core-and-utils/structures/pose_estimation_result.md) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `poses` | `np.ndarray` | `(N, 4, 4)` float32 | Estimated 4x4 object-to-camera transformation matrices. |
| `intrinsics` | `np.ndarray` | `(3, 3)` float32 | Camera intrinsic matrix. |
| `scores` | `np.ndarray \| None` | `(N,)` float32 | Confidence scores associated with candidate poses. |
| `labels` | `list[str] \| None` | Length `N` | Object model identifiers. |
| `bbox_3d` | `np.ndarray \| None` | `(N, 8, 3)` or `(8, 3)` float32 | 3D bounding box corners in canonical centered mesh space. |
| `to_origin` | `np.ndarray \| None` | `(N, 4, 4)` or `(4, 4)` float32 | Centering transform matrix for coordinate axis positioning. |
