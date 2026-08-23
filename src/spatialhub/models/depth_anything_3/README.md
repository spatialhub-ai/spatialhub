# Depth Anything 3 Technical Reference

`spatialhub.models.depth_anything_3` provides an ONNX Runtime adapter for **Depth Anything 3 (DA3)**, supporting monocular relative and metric depth estimation, multi-view camera pose alignment, and nested dual-model stitching.

---

## 1. Supported Model Presets & Series

`DepthAnything3Adapter` supports 4 categories of model presets via `model_name`:

### 🌟 DA3 Main Series (Monocular & Multi-View Geometry)
Flagship foundation models trained with a unified depth-ray representation for monocular depth, multi-view depth, and camera pose estimation:

| Preset Name (`model_name`) | ONNX File | Model Architecture | Description |
| :--- | :--- | :--- | :--- |
| `"da3_small"` | `da3_small.onnx` | DINO Small Backbone | Ultra-lightweight model for real-time mobile/edge devices. |
| `"da3_base"` (Default) | `da3_base.onnx` | DINO Base Backbone | Balanced accuracy and inference speed. |
| `"da3_large"` | `da3_large.onnx` | DINO Large Backbone | High-accuracy foundation model for 3D reconstruction. |
| `"da3_giant"` | `da3_giant.onnx` | DINO Giant Backbone | Flagship model with maximum visual geometry resolution. |

### 📐 DA3 Metric Series (Real-World Physical Scale)
Specialized models fine-tuned for physical scale estimation (depth output measured in meters):

| Preset Name (`model_name`) | ONNX File | Description |
| :--- | :--- | :--- |
| `"da3_metric_large"` | `da3_mono_large.onnx` | Predicts monocular depth in physical metric scale (meters). |

### 🔍 DA3 Monocular Series (High-Precision Relative Depth)
Dedicated models for high-quality relative monocular depth without disparity distortion:

| Preset Name (`model_name`) | ONNX File | Description |
| :--- | :--- | :--- |
| `"da3_mono_large"` | `da3_mono_large.onnx` | High-precision relative monocular depth estimation. |

### 🔗 DA3 Nested Dual-Model Series (Detail + Metric Scale)
Combines high-resolution geometric detail of any-view Giant with physical metric scale of Metric Large via least-squares scale-and-shift alignment (`align_nested_depth_np`):

```python
# Pass 2 models to create a nested dual-model pipeline
estimator = DepthAnything3(model_name=["da3_giant", "da3_metric_large"])
```

---

## 2. Overview & Mathematical Preprocessing

Depth Anything 3 processes input image batches of shape $(B, N, 3, H, W)$ where $B=1$ and $N$ represents the view count.

### Sky Masking & Horizon Suppression

Monocular depth estimates near the sky boundary undergo unconstrained dispersion. Sky mask prediction $M_{sky} \in [0, 1]^{H \times W}$ suppresses depth values where $M_{sky} > 0.5$, clamping near-infinite range predictions.

### Umeyama Sim(3) Trajectory Alignment

Multi-view poses $(R, t, s)$ are aligned to reference ground truth trajectories using Umeyama rigid similarity transformations minimizing squared point distance:

$$
\min_{R, t, s} \sum_{i=1}^N \| s R p_i + t - q_i \|^2
$$

---

## 3. ONNX Export Guide

The export environment for Depth Anything 3 resides beside its `pyproject.toml` file at `src/spatialhub/models/depth_anything_3/DepthAnything3`.

### Environment Setup

```bash
cd src/spatialhub/models/depth_anything_3/DepthAnything3
uv sync
```

### Running Export Script

```bash
uv run python export_onnx.py \
    --model-name depth-anything/DA3-BASE \
    --onnx-path weights/da3_base.onnx \
    --device cpu \
    --opset 18 \
    --views 2 \
    --height 504 \
    --width 504
```

---

## 4. SpatialHub Adapter API & Usage

```python
import cv2
from spatialhub import DepthAnything3

# Initialize adapter with DA3 Base preset
estimator = DepthAnything3(
    model_name="da3_base",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Estimate depth across image views
result = estimator.estimate_depth(images=["view1.png", "view2.png"])

# Render colorized depth map
colorized_depth = estimator.visualize(result.depth[0])
cv2.imwrite("depth_view1.png", cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR))
```

---

## 5. Returned Result Data Structure

Returns a [`DepthPredictionResult`](../../docs/structures/overview.md#2-depthpredictionresult) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(N, H, W, 3)` uint8 | Input RGB image batch. |
| `depth` | `np.ndarray` | `(N, H, W)` float32 | Predicted depth maps (in meters or relative scale). |
| `conf` | `np.ndarray \| None` | `(N, H, W)` float32 | Prediction confidence maps `[0.0, 1.0]`. |
| `intrinsics` | `np.ndarray \| None` | `(N, 3, 3)` float32 | Extracted or passed camera intrinsics. |
| `extrinsics` | `np.ndarray \| None` | `(N, 4, 4)` float32 | Estimated multi-view camera extrinsic matrices. |
| `depth_type` | `str` | N/A | Scale type (`"metric"` or `"relative"`). |
