# SpatialHub Technical Documentation

**SpatialHub** is a lightweight, PyTorch-free spatial computing and perception library designed for efficient inference, zero-configuration weight management, and unified Python API contracts built on **ONNX Runtime**.

---

## Key Technical Specifications

* **PyTorch-Free Inference Path:** Pure NumPy and OpenCV vector preprocessing and postprocessing. Inference engines execute exclusively on ONNX Runtime.
* **Unified Return Contracts:** Standardized dataclass outputs ([`MatchResult`](core-and-utils/structures/match_result.md), [`DepthPredictionResult`](core-and-utils/structures/depth_prediction_result.md), [`FeatureExtractionResult`](core-and-utils/structures/feature_extraction_result.md), [`SegmentationResult`](core-and-utils/structures/segmentation_result.md), [`PoseEstimationResult`](core-and-utils/structures/pose_estimation_result.md)).
* **Automatic Weight Management:** Downloads, verifies, and caches pretrained `.onnx` weight binaries from Hugging Face Hub.
* **Execution Provider Configuration:** Verifies target execution providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`) with runtime fallback warnings.

---

## Perception Models Summary

| Model | Task | Returned Result Class | Export Submodule |
| :--- | :--- | :--- | :--- |
| [**EfficientLoFTR**](models/eloftr.md) | Semi-dense Feature Matching | [`MatchResult`](core-and-utils/structures/match_result.md) | `src/spatialhub/models/efficient_loftr` |
| [**Depth Anything 3**](models/depthanything3.md) | Monocular & Multi-View Depth | [`DepthPredictionResult`](core-and-utils/structures/depth_prediction_result.md) | `src/spatialhub/models/depth_anything_3/DepthAnything3` |
| [**DINOv2**](models/dinov2.md) | Image Feature Extraction | [`FeatureExtractionResult`](core-and-utils/structures/feature_extraction_result.md) | `src/spatialhub/models/dinov2/DINOv2` |
| [**FastSAM**](models/fastsam.md) | Real-Time Proposal Segmentation | [`SegmentationResult`](core-and-utils/structures/segmentation_result.md) | `src/spatialhub/models/fastsam/FastSAM` |
| [**SAM**](models/sam.md) | Automatic Mask Generation (AMG) | [`SegmentationResult`](core-and-utils/structures/segmentation_result.md) | `src/spatialhub/models/sam/SAM` |
| [**CNOS**](models/cnos.md) | CAD Zero-Shot Object Detection | [`SegmentationResult`](core-and-utils/structures/segmentation_result.md) | `src/spatialhub/models/cnos/CNOS` |
| [**FoundationPose**](models/foundationpose.md) | Model-based 6D Object Pose & Tracking | [`PoseEstimationResult`](core-and-utils/structures/pose_estimation_result.md) | `src/spatialhub/models/foundationpose` |