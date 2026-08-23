# SpatialHub

[![PyPI version](https://badge.fury.io/py/spatialhub.svg)](https://badge.fury.io/py/spatialhub)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A lightweight spatial computing and perception library providing PyTorch-free ONNX Runtime inference adapters for computer vision models.

---

## Key Principles

- **Framework Decoupling:** Core runtime paths execute on **ONNX Runtime** without requiring PyTorch at inference time.
- **Pure NumPy & OpenCV Processing:** Preprocessing, resizing, dynamic padding, coordinate projections, and alignment solvers use pure NumPy and OpenCV vector operations.
- **Automatic Weight Management:** Automatically retrieves, verifies, and caches pretrained `.onnx` model weights from Hugging Face Hub.
- **Standardized API Contracts:** Unified dataclass return structures ([`MatchResult`](./docs/structures/overview.md#1-matchresult), [`DepthPredictionResult`](./docs/structures/overview.md#2-depthpredictionresult), [`FeatureExtractionResult`](./docs/structures/overview.md#3-featureextractionresult), [`SegmentationResult`](./docs/structures/overview.md#4-segmentationresult)).

---

## Supported Models & Technical References

| Model Architecture | Task | Default Variant / Option | Returned Dataclass | Documentation & Export Guide |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientLoFTR** | Semi-dense Feature Matching | `"full"` or `"opt"` | `MatchResult` | [Technical Reference & Export Guide](./src/spatialhub/models/efficient_loftr/README.md) |
| **Depth Anything 3** | Monocular & Multi-View Depth | `"da3_base"` (small/large/giant/metric/nested) | `DepthPredictionResult` | [Technical Reference & Export Guide](./src/spatialhub/models/depth_anything_3/README.md) |
| **DINOv2** | Image Feature Extraction | `"dinov2_vitl14"` (vits14/vitb14/vitg14) | `FeatureExtractionResult` | [Technical Reference & Export Guide](./src/spatialhub/models/dinov2/README.md) |
| **FastSAM** | Instance Proposal Segmentation | `"FastSAM-x"` or `"FastSAM-s"` | `SegmentationResult` | [Technical Reference & Export Guide](./src/spatialhub/models/fastsam/README.md) |
| **SAM** | Automatic Mask Generation (AMG) | `"sam_vit_h"` (vit_l/vit_b) | `SegmentationResult` | [Technical Reference & Export Guide](./src/spatialhub/models/sam/README.md) |
| **CNOS** | CAD Zero-Shot Object Detection | 3D CAD Mesh (`.ply`, `.obj`, `.stl`) | `SegmentationResult` | [Technical Reference & Export Guide](./src/spatialhub/models/cnos/README.md) |

---

## Installation

Requires **Python 3.12+**.

```bash
pip install spatialhub
```

For GPU acceleration (CUDA):

```bash
pip install "spatialhub[gpu]"
```

For 3D CAD mesh rendering support (Pyrender & Trimesh):

```bash
pip install "spatialhub[render]"
```

---

## Quickstart

```python
from spatialhub import EfficientLoFTR, DepthAnything3, DINOV2, FastSAM, SAM, CNOS

# 1. Feature Matching (EfficientLoFTR)
matcher = EfficientLoFTR()
match_res = matcher.match("img1.jpg", "img2.jpg", max_dim=1024)
match_res.visualize(top_k=50, save_path="matches.png")

# 2. Depth Estimation (Depth Anything 3)
estimator = DepthAnything3(model_name="da3_base")
depth_res = estimator.estimate_depth(images=["view1.png", "view2.png"])
depth_viz = estimator.visualize(depth_res.depth[0])

# 3. Feature Embeddings (DINOv2)
dino = DINOV2(model_variant="dinov2_vitl14")
feat_res = dino.extract_features("image.png", l2_normalize=True)

# 4. Proposal Segmentation (FastSAM)
fastsam = FastSAM(model_variant="FastSAM-x")
seg_res = fastsam.generate_masks("scene.png", conf_threshold=0.3)
seg_res.visualize_mask(save_path="fastsam_masks.png")
```

---

## Reproducible ONNX Export Workflow

Each model directory under `src/spatialhub/models/<model>/` contains its own `pyproject.toml` environment configuration and `export_onnx.py` script. To modify PyTorch source code or export custom ONNX graphs:

```bash
cd src/spatialhub/models/efficient_loftr
uv sync
uv run python export_onnx.py --checkpoint weights/model.ckpt --output-path weights/model.onnx
```

See the [Reproducible ONNX Export Guide](./docs/onnx-export/overview.md) for full instructions.

---

## License

Core SpatialHub code is released under the [Apache 2.0 License](LICENSE). Individual pretrained model weights and submodule architectures maintain their respective original licenses.