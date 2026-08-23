# Models Overview

SpatialHub provides ONNX Runtime adapters for 6 computer vision model architectures across perception tasks. All adapters return standard Python dataclass contracts defined in [`spatialhub.structures`](../structures/overview.md).

---

## Perception Models Summary Table

| Model Architecture | Task | Primary Adapter Class | Available Model Options / Presets | Supported Sub-Adapters | Returned Result Contract |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientLoFTR** | Semi-dense Feature Matching | `EfficientLoFTR` | `"full"` (Default), `"opt"` | N/A | [`MatchResult`](../structures/overview.md#1-matchresult) |
| **Depth Anything 3** | Monocular & Multi-View Depth | `DepthAnything3` | `"da3_small"`, `"da3_base"` (Default), `"da3_large"`, `"da3_giant"`, `"da3_metric_large"`, `"da3_mono_large"`, `["da3_giant", "da3_metric_large"]` (Nested) | N/A | [`DepthPredictionResult`](../structures/overview.md#2-depthpredictionresult) |
| **DINOv2** | Image Feature Extraction | `DINOV2` | `"dinov2_vits14"`, `"dinov2_vitb14"`, `"dinov2_vitl14"` (Default), `"dinov2_vitg14"` | N/A | [`FeatureExtractionResult`](../structures/overview.md#3-featureextractionresult) |
| **FastSAM** | Real-Time Proposal Segmentation | `FastSAM` | `"FastSAM-x"` (Default), `"FastSAM-s"` | N/A | [`SegmentationResult`](../structures/overview.md#4-segmentationresult) |
| **SAM** | Automatic Mask Generation (AMG) | `SAM` | `"sam_vit_h"` (Default), `"sam_vit_l"`, `"sam_vit_b"` | N/A | [`SegmentationResult`](../structures/overview.md#4-segmentationresult) |
| **CNOS** | CAD Zero-Shot Object Segmentation | `CNOS` | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | **Segmentors:** `FastSAMAdapter`, `SAMAdapter`<br>**Descriptor:** `DINOv2Adapter` | [`SegmentationResult`](../structures/overview.md#4-segmentationresult) |

---

## Detailed Technical References

- [**EfficientLoFTR Documentation**](eloftr.md) — `"full"` and `"opt"` precision options, dimension alignment math, coordinate scaling, and matching API.
- [**Depth Anything 3 Documentation**](depthanything3.md) — Main foundation series (`small`, `base`, `large`, `giant`), Metric series, Monocular series, and Nested dual-model alignment.
- [**DINOv2 Documentation**](dinov2.md) — Vision Transformer backbones (`vits14`, `vitb14`, `vitl14`, `vitg14`), ImageNet normalization, and unit L2 normalization.
- [**FastSAM Documentation**](fastsam.md) — YOLOv8-Seg models (`FastSAM-x`, `FastSAM-s`), prototype mask decoding, NMS filtering, and proposal generation.
- [**SAM Documentation**](sam.md) — SAM variants (`sam_vit_h`, `sam_vit_l`, `sam_vit_b`), decoupled Image Encoder / Mask Decoder ONNX execution, point grid sampling, and AMG.
- [**CNOS Documentation**](cnos.md) — 3D CAD mesh template rendering, pluggable segmentors (`FastSAM`, `SAM`), DINOv2 descriptor matching, and zero-shot instance segmentation.
