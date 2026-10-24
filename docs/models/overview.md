# Model Adapters

SpatialHub provides ONNX Runtime inference adapters for deep vision architectures across spatial perception tasks. Each adapter standardizes input preprocessing and output decoding, returning typed geometric containers defined in [`spatialhub.structures`](../structures/poses_and_cameras.md).

---

## Adapter Catalog

| Model | Task | Primary Adapter Class | Output Structure |
| :--- | :--- | :--- | :--- |
| [**EfficientLoFTR**](eloftr.md) | Semi-dense Feature Matching | `EfficientLoFTR` | [`KeypointMatches`](../structures/perception_primitives.md#keypointmatches) |
| [**Depth Anything 3**](depthanything3.md) | Monocular & Metric Depth | `DepthAnything3` | [`DepthMap`](../structures/scene_and_bounds.md#depthmap) |
| [**DINOv2**](dinov2.md) | Vision Transformer Descriptors | `DINOv2` | [`FeatureEmbedding`](../structures/perception_primitives.md#featureembedding) |
| [**FastSAM**](fastsam.md) | Real-Time Proposal Segmentation | `FastSAM` | [`InstanceSegmentation`](../structures/perception_primitives.md#instancesegmentation) |
| [**SAM**](sam.md) | Automatic Mask Generation | `SAM` | [`InstanceSegmentation`](../structures/perception_primitives.md#instancesegmentation) |
| [**CNOS**](cnos.md) | Zero-Shot CAD Detection | `CNOS` | [`InstanceSegmentation`](../structures/perception_primitives.md#instancesegmentation) |
| [**FoundationPose**](foundationpose.md) | 6D Object Pose Estimation & Tracking | `FoundationPose` | [`ObjectPose6D`](../structures/poses_and_cameras.md#objectpose6d) |

---

## Execution Providers

Adapters configure hardware execution through the `providers` parameter, accepting ONNX Runtime execution provider names ordered by preference:

```python
# Hardware acceleration with CPU fallback
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Host CPU execution
providers = ["CPUExecutionProvider"]
```

When omitted, execution defaults to available host runtime providers.

---

## Resource Management

Adapters manage ONNX Runtime inference sessions and GPU memory buffers. All adapters support the Python context manager protocol (`with` statement) for deterministic teardown:

```python
with EfficientLoFTR(model_type="full") as matcher:
    matches = matcher.match(image_a, image_b)
# ONNX sessions and cached buffers released upon exit
```

For long-lived instances or pipeline loops, call `.close()` directly to release sessions:

```python
matcher = EfficientLoFTR(model_type="full")
matches = matcher.match(image_a, image_b)
matcher.close()
```

---

## Data Contracts

Every adapter maps raw neural network output tensors directly into typed dataclasses from `spatialhub.structures`:

- Pixel-space correspondences map to [`KeypointMatches`](../structures/perception_primitives.md#keypointmatches).
- Segmentation masks and bounding boxes map to [`InstanceSegmentation`](../structures/perception_primitives.md#instancesegmentation).
- Monocular and multi-view depth estimates map to [`DepthMap`](../structures/scene_and_bounds.md#depthmap).
- Dense and global semantic descriptors map to [`FeatureEmbedding`](../structures/perception_primitives.md#featureembedding).
- Rigid 3D camera-space poses map to [`ObjectPose6D`](../structures/poses_and_cameras.md#objectpose6d).

Downstream utilities in `spatialhub.utils` operate on these structures without format conversion.

---

## Custom Adapters

SpatialHub does not enforce an inheritance base class for model adapters. Custom models or inference routines can integrate into SpatialHub workflows by returning the corresponding dataclass from [`spatialhub.structures`](../structures/poses_and_cameras.md).
