# SpatialHub

[![PyPI version](https://badge.fury.io/py/spatialhub.svg)](https://badge.fury.io/py/spatialhub)
[![Documentation](https://img.shields.io/badge/docs-spatialhub--ai.github.io-blue)](https://spatialhub-ai.github.io/spatialhub/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A spatial AI and perception library providing unified ONNX Runtime inference adapters for computer vision and 3D spatial computing models.

---

## Motivation & Architecture

3D spatial vision systems (such as 6D object pose estimation, Visual SLAM, and 3D reconstruction) are multi-stage pipelines composed of representation learning, geometric matching, depth prediction, proposal segmentation, and pose optimization.

In practice, integrating multiple vision models into a unified pipeline presents practical challenges:
* **Interface Variations:** Models differ in coordinate conventions, dictionary formats, and output tensor dimensions.
* **Environment Conflicts:** Combining distinct model implementations often introduces dependency conflicts and heavy runtime footprints.

SpatialHub addresses this by providing standardized dataclass returns over a unified ONNX Runtime execution layer:

```text
Sensor & Asset Inputs                   Perception Adapters (ONNX)          Standardized Dataclasses
─────────────────────────────────      ─────────────────────────────       ─────────────────────────
Single RGB Image                  ───►  DINOv2 (Feature Extraction)   ───►  FeatureExtractionResult
Image Pair                        ───►  EfficientLoFTR (Matching)     ───►  MatchResult
RGB Images + Intrinsics (K)       ───►  Depth Anything 3 (Depth)      ───►  DepthPredictionResult
RGB Image + CAD Mesh (.ply)       ───►  FastSAM / SAM / CNOS (Masks)  ───►  SegmentationResult
RGB-D + Intrinsics (K) + CAD Mesh ───►  FoundationPose (6D Pose)      ───►  PoseEstimationResult
                                                                                    │
                                                                                    ▼
Downstream Systems (Visual SLAM, 3D Reconstruction, Robotics Manipulation)
```

Downstream spatial algorithms operate on these return types, allowing models to be interchanged without modifying downstream pipeline logic.

---

## Key Principles

- **Standardized Return Contracts:** Dataclass structures across model families ([`MatchResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/match_result/), [`DepthPredictionResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/depth_prediction_result/), [`FeatureExtractionResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/feature_extraction_result/), [`SegmentationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/segmentation_result/), [`PoseEstimationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/pose_estimation_result/)).
- **ONNX Runtime Execution:** Inference executes via ONNX Runtime with NumPy and OpenCV vector operations.
- **Model Weight Resolution:** Automatically fetches and caches `.onnx` weight files from Hugging Face Hub.
- **Hardware Acceleration:** Supports CPU, CUDA, and TensorRT execution providers.
- **Offscreen Rendering Utilities:** Headless G-buffer and batched atlas renderer for CAD mesh template matching and pose estimation.

---

## Supported Models

| Model Architecture | Task | Default Variant / Option | Returned Dataclass | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **FoundationPose** | 6D Object Pose Estimation & Tracking | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | [`PoseEstimationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/pose_estimation_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/foundationpose/) • [README](./src/spatialhub/models/foundationpose/README.md) |
| **EfficientLoFTR** | Semi-dense Feature Matching | `"full"` or `"opt"` | [`MatchResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/match_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/eloftr/) • [README](./src/spatialhub/models/efficient_loftr/README.md) |
| **Depth Anything 3** | Monocular & Multi-View Depth | `"da3_base"` (small/large/giant/metric/nested) | [`DepthPredictionResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/depth_prediction_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/depthanything3/) • [README](./src/spatialhub/models/depth_anything_3/README.md) |
| **DINOv2** | Image Feature Extraction | `"vitl14"` (vits14/vitb14/vitg14) | [`FeatureExtractionResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/feature_extraction_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/dinov2/) • [README](./src/spatialhub/models/dinov2/README.md) |
| **FastSAM** | Real-Time Proposal Segmentation | `"x"` or `"s"` | [`SegmentationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/segmentation_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/fastsam/) • [README](./src/spatialhub/models/fastsam/README.md) |
| **SAM** | Automatic Mask Generation (AMG) | `"vit_h"` (vit_l/vit_b) | [`SegmentationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/segmentation_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/sam/) • [README](./src/spatialhub/models/sam/README.md) |
| **CNOS** | CAD Zero-Shot Object Detection | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | [`SegmentationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/segmentation_result/) | [Docs](https://spatialhub-ai.github.io/spatialhub/models/cnos/) • [README](./src/spatialhub/models/cnos/README.md) |

---

## Installation

Requires **Python 3.12+**. Choose the installation for your hardware setup:

### Standard Installation

For CPU inference:
```bash
pip install "spatialhub[cpu]"
```

For NVIDIA GPU acceleration (CUDA / TensorRT):
```bash
pip install "spatialhub[gpu]"
```

### 3D CAD & Rendering Installation

For workflows requiring 3D CAD mesh loading and template rendering (e.g. FoundationPose, CNOS):

For CPU with rendering:
```bash
pip install "spatialhub[cpu,render]"
```

For GPU with rendering:
```bash
pip install "spatialhub[gpu,render]"
```

> [!NOTE]
> Do not install both `onnxruntime` and `onnxruntime-gpu` in the same Python environment as their binary namespaces conflict.

---

## Quickstart

```python
from spatialhub import FoundationPose, EfficientLoFTR, DepthAnything3, DINOv2, FastSAM, SAM, CNOS

# 6D Object Pose Estimation (FoundationPose)
est = FoundationPose(
    model_path="mesh.obj",
    model_unit="mm",
    scorer_weights="scorer.onnx",
    refiner_weights="refiner.onnx",
)
pose_res = est.estimate(rgb=rgb_img, depth=depth_img, K=cam_K, mask=obj_mask)
pose_res.visualize(draw_bbox=True, draw_axes=True, save_path="pose.png")

# Feature Matching (EfficientLoFTR)
matcher = EfficientLoFTR()
match_res = matcher.match("img1.jpg", "img2.jpg", max_dim=1024)
match_res.visualize(top_k=50, save_path="matches.png")

# Depth Estimation (Depth Anything 3)
estimator = DepthAnything3(model_name="da3_base")
depth_res = estimator.estimate_depth(images=["view1.png", "view2.png"])
depth_viz = estimator.visualize(depth_res.depth[0])

# Feature Extraction (DINOv2)
dino = DINOv2(model_variant="vitl14")
feat_res = dino.extract_features("image.png", l2_normalize=True)

# Proposal Segmentation (FastSAM / SAM)
fastsam = FastSAM(model_variant="x")
seg_res = fastsam.generate_masks("scene.png", conf_threshold=0.3)
seg_res.visualize_mask(save_path="fastsam_masks.png")

sam = SAM(model_variant="vit_b")
sam_res = sam.generate_masks("scene.png", points_per_side=16)
sam_res.visualize_mask(save_path="sam_masks.png")
```

---

## ONNX Export

Export scripts for generating `.onnx` models from source repositories are located under `tools/export/`:

```bash
uv run tools/export/export_efficient_loftr.py --checkpoint weights/eloftr_outdoor.ckpt --output-folder onnx_weight
```

See the [ONNX Export Guide](https://spatialhub-ai.github.io/spatialhub/core-and-utils/export/) for options and details.

---

## License

Core SpatialHub code is released under the [Apache 2.0 License](LICENSE). Pretrained model weights and submodule architectures maintain their respective original licenses.