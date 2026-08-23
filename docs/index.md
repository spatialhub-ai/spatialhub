# SpatialHub Technical Documentation

**SpatialHub** is a lightweight, PyTorch-free spatial computing and perception library designed for efficient inference, zero-configuration weight management, and unified Python API contracts built on **ONNX Runtime**.

---

## Key Technical Specifications

* **PyTorch-Free Inference Path:** Pure NumPy and OpenCV vector preprocessing and postprocessing. Inference engines execute exclusively on ONNX Runtime.
* **Unified Return Contracts:** Standardized dataclass outputs ([`MatchResult`](structures/overview.md#1-matchresult), [`DepthPredictionResult`](structures/overview.md#2-depthpredictionresult), [`FeatureExtractionResult`](structures/overview.md#3-featureextractionresult), [`SegmentationResult`](structures/overview.md#4-segmentationresult)).
* **Automatic Weight Management:** Downloads, verifies, and caches pretrained `.onnx` weight binaries from Hugging Face Hub.
* **Execution Provider Configuration:** Verifies target execution providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`) with runtime fallback warnings.

---

## Perception Models Summary

| Model | Task | Returned Result Class | Export Submodule |
| :--- | :--- | :--- | :--- |
| [**EfficientLoFTR**](models/eloftr.md) | Semi-dense Feature Matching | [`MatchResult`](structures/overview.md#1-matchresult) | `src/spatialhub/models/efficient_loftr` |
| [**Depth Anything 3**](models/depthanything3.md) | Monocular & Multi-View Depth | [`DepthPredictionResult`](structures/overview.md#2-depthpredictionresult) | `src/spatialhub/models/depth_anything_3/DepthAnything3` |
| [**DINOv2**](models/dinov2.md) | Image Feature Extraction | [`FeatureExtractionResult`](structures/overview.md#3-featureextractionresult) | `src/spatialhub/models/dinov2/DINOv2` |
| [**FastSAM**](models/fastsam.md) | Real-Time Proposal Segmentation | [`SegmentationResult`](structures/overview.md#4-segmentationresult) | `src/spatialhub/models/fastsam/FastSAM` |
| [**SAM**](models/sam.md) | Automatic Mask Generation (AMG) | [`SegmentationResult`](structures/overview.md#4-segmentationresult) | `src/spatialhub/models/sam/SAM` |
| [**CNOS**](models/cnos.md) | CAD Zero-Shot Object Detection | [`SegmentationResult`](structures/overview.md#4-segmentationresult) | `src/spatialhub/models/cnos/CNOS` |