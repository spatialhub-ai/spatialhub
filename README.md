# SpatialHub

[![PyPI version](https://badge.fury.io/py/spatialhub.svg)](https://badge.fury.io/py/spatialhub)
[![Documentation](https://img.shields.io/badge/docs-spatialhub--ai.github.io-blue)](https://spatialhub-ai.github.io/spatialhub/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance, zero-PyTorch spatial AI and perception library providing unified ONNX Runtime inference adapters for computer vision and 3D spatial computing models.

---

## Key Principles

- **Zero-PyTorch Inference:** Core runtime paths execute exclusively on **ONNX Runtime** with pure NumPy and OpenCV vector operations.
- **Unified Return Contracts:** Standardized dataclass outputs across all model families ([`MatchResult`](./docs/core-and-utils/structures/match_result.md), [`DepthPredictionResult`](./docs/core-and-utils/structures/depth_prediction_result.md), [`FeatureExtractionResult`](./docs/core-and-utils/structures/feature_extraction_result.md), [`SegmentationResult`](./docs/core-and-utils/structures/segmentation_result.md), [`PoseEstimationResult`](./docs/core-and-utils/structures/pose_estimation_result.md)).
- **Automatic Weight Management:** Downloads, verifies, and caches pretrained `.onnx` weight binaries seamlessly from Hugging Face Hub.
- **Hardware Acceleration:** Native support for CPU, CUDA, and TensorRT execution providers with runtime fallback verification.
- **ModernGL GPU Rendering:** Built-in headless offscreen G-buffer and batched atlas renderer for CAD model template matching and 6D pose estimation.

---

## Supported Models

| Model Architecture | Task | Default Variant / Option | Returned Dataclass | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| [**FoundationPose**](./docs/models/foundationpose.md) | 6D Object Pose Estimation & Tracking | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | [`PoseEstimationResult`](./docs/core-and-utils/structures/pose_estimation_result.md) | [Model Reference](./docs/models/foundationpose.md) • [Export Guide](./src/spatialhub/models/foundationpose/README.md) |
| [**EfficientLoFTR**](./docs/models/eloftr.md) | Semi-dense Feature Matching | `"full"` or `"opt"` | [`MatchResult`](./docs/core-and-utils/structures/match_result.md) | [Model Reference](./docs/models/eloftr.md) • [Export Guide](./src/spatialhub/models/efficient_loftr/README.md) |
| [**Depth Anything 3**](./docs/models/depthanything3.md) | Monocular & Multi-View Depth | `"da3_base"` (small/large/giant/metric/nested) | [`DepthPredictionResult`](./docs/core-and-utils/structures/depth_prediction_result.md) | [Model Reference](./docs/models/depthanything3.md) • [Export Guide](./src/spatialhub/models/depth_anything_3/README.md) |
| [**DINOv2**](./docs/models/dinov2.md) | Image Feature Extraction | `"dinov2_vitl14"` (vits14/vitb14/vitg14) | [`FeatureExtractionResult`](./docs/core-and-utils/structures/feature_extraction_result.md) | [Model Reference](./docs/models/dinov2.md) • [Export Guide](./src/spatialhub/models/dinov2/README.md) |
| [**FastSAM**](./docs/models/fastsam.md) | Real-Time Proposal Segmentation | `"FastSAM-x"` or `"FastSAM-s"` | [`SegmentationResult`](./docs/core-and-utils/structures/segmentation_result.md) | [Model Reference](./docs/models/fastsam.md) • [Export Guide](./src/spatialhub/models/fastsam/README.md) |
| [**SAM**](./docs/models/sam.md) | Automatic Mask Generation (AMG) | `"sam_vit_h"` (vit_l/vit_b) | [`SegmentationResult`](./docs/core-and-utils/structures/segmentation_result.md) | [Model Reference](./docs/models/sam.md) • [Export Guide](./src/spatialhub/models/sam/README.md) |
| [**CNOS**](./docs/models/cnos.md) | CAD Zero-Shot Object Detection | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | [`SegmentationResult`](./docs/core-and-utils/structures/segmentation_result.md) | [Model Reference](./docs/models/cnos.md) • [Export Guide](./src/spatialhub/models/cnos/README.md) |

---

## Installation

Requires **Python 3.12+**.

```bash
pip install spatialhub
```

For GPU acceleration (CUDA / TensorRT):

```bash
pip install "spatialhub[gpu]"
```

For 3D CAD mesh processing and ModernGL rendering:

```bash
pip install "spatialhub[render]"
```

---

## Quickstart

```python
from spatialhub import FoundationPose, EfficientLoFTR, DepthAnything3, DINOV2, FastSAM, SAM, CNOS

# 1. 6D Object Pose Estimation (FoundationPose)
est = FoundationPose(
    model_path="mesh.obj",
    model_unit="mm",
    scorer_weights="scorer.onnx",
    refiner_weights="refiner.onnx"
)
pose_res = est.estimate(rgb=rgb_img, depth=depth_img, K=cam_K, mask=obj_mask)
pose_res.visualize(draw_bbox=True, draw_axes=True, save_path="pose.png")

# 2. Feature Matching (EfficientLoFTR)
matcher = EfficientLoFTR()
match_res = matcher.match("img1.jpg", "img2.jpg", max_dim=1024)
match_res.visualize(top_k=50, save_path="matches.png")

# 3. Depth Estimation (Depth Anything 3)
estimator = DepthAnything3(model_name="da3_base")
depth_res = estimator.estimate_depth(images=["view1.png", "view2.png"])
depth_viz = estimator.visualize(depth_res.depth[0])

# 4. Feature Embeddings (DINOv2)
dino = DINOV2(model_variant="dinov2_vitl14")
feat_res = dino.extract_features("image.png", l2_normalize=True)

# 5. Proposal Segmentation (FastSAM)
fastsam = FastSAM(model_variant="FastSAM-x")
seg_res = fastsam.generate_masks("scene.png", conf_threshold=0.3)
seg_res.visualize_mask(save_path="fastsam_masks.png")
```

---

## Reproducible ONNX Export Workflow

Each model directory under `src/spatialhub/models/<model>/` contains an isolated environment configuration and `export_onnx.py` script to re-export custom ONNX graphs:

```bash
cd src/spatialhub/models/efficient_loftr
uv sync
uv run python export_onnx.py --checkpoint weights/model.ckpt --output-path weights/model.onnx
```

See the [Reproducible ONNX Export Guide](https://spatialhub-ai.github.io/spatialhub/onnx-export/overview/) for complete documentation.

---

## License

Core SpatialHub code is released under the [Apache 2.0 License](LICENSE). Pretrained model weights and submodule architectures maintain their respective original licenses.