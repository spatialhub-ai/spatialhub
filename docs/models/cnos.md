# CNOS (CAD-based Novel Object Segmentation) Technical Reference

`spatialhub.models.cnos` provides an ONNX Runtime adapter for **CNOS**, executing CAD mesh template rendering, DINOv2 feature extraction, FastSAM/SAM proposal segmenting, and cosine similarity matching for zero-shot object detection.

---

## Supported Sub-Adapter Components

CNOS operates by coupling 2 pluggable sub-adapter pipelines with a 3D CAD mesh input file (`.ply`, `.obj`, `.stl`):

### Proposal Segmentor Sub-Adapters (`segmentor`)
Generates spatial object mask proposals across the scene image:

| Segmentor Adapter Class | Supported Variants | Target Performance |
| :--- | :--- | :--- |
| `FastSAMAdapter` (Default) | `"FastSAM-x"`, `"FastSAM-s"` | Real-time candidate box & mask proposal generation. |
| `SAMAdapter` | `"sam_vit_h"`, `"sam_vit_l"`, `"sam_vit_b"` | High-precision Automatic Mask Generation (AMG). |

### Feature Descriptor Sub-Adapter (`descriptor`)
Extracts L2-normalized feature embeddings from 2D rendered CAD templates and scene mask proposals:

| Descriptor Adapter Class | Supported Variants | Description |
| :--- | :--- | :--- |
| `DINOv2Adapter` (Default) | `"dinov2_vitl14"`, `"dinov2_vitb14"`, `"dinov2_vits14"` | Extracts L2-normalized CLS token embeddings for top-$k$ cosine similarity matching. |

---

## Overview & Mathematical Preprocessing

CNOS loads a 3D CAD mesh file (`.ply`, `.obj`, `.stl`), renders 2D template views across pre-computed camera poses, extracts DINOv2 feature embeddings for each template, and matches image segment proposals via cosine similarity.

### CAD Template Rendering

Using `TemplateRenderer`, $M$ template views $I_{rgba} \in \mathbb{U}^{H \times W \times 4}$ are rendered at fixed camera poses $P_m = [R_m \mid t_m]$.

### Bounding Box Cropping & Preprocessing

Each rendered template is cropped to its foreground bounding box $[x_1, y_1, x_2, y_2]$, square-padded, resized to $224 \times 224$, and normalized using ImageNet statistics.

### Cosine Similarity Matching

DINOv2 feature vectors $F_{scene} \in \mathbb{R}^{K \times D}$ for $K$ scene proposals and cached template features $F_{ref} \in \mathbb{R}^{M \times D}$ undergo cosine matrix multiplication:

$$
S = F_{scene} \cdot F_{ref}^T \quad \in \mathbb{R}^{K \times M}
$$

Top-$k$ semantic match score aggregation (CNOS defaults to $k=5$):

$$
\text{score}_i = \frac{1}{k} \sum_{j=1}^k \text{TopK}(S_{i, :}, k)_j
$$

---

## ONNX Export Guide

The export environments for CNOS submodules reside beside their `pyproject.toml` file at `src/spatialhub/models/cnos/CNOS`.

### Environment Setup

```bash
cd src/spatialhub/models/cnos/CNOS
uv sync
```

### Running Sub-Module Export Scripts

```bash
# Export DINOv2 descriptor sub-model
uv run python export_dinov2.py --model-name dinov2_vitl14 --output-folder ./pretrained

# Export FastSAM segmentor sub-model
uv run python export_fastsam.py --checkpoint FastSAM-x.pt --output-folder ./pretrained

# Export SAM segmentor sub-model
uv run python export_sam.py --model-type vit_h --out-encoder ./pretrained/sam_image_encoder.onnx --out-decoder ./pretrained/sam_mask_decoder.onnx
```

---

## SpatialHub Adapter API & Usage

### Usage with FastSAM Segmentor (Real-Time)

```python
from spatialhub import CNOS, DINOV2, FastSAM

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize sub-adapters
descriptor = DINOV2(model_variant="dinov2_vitl14", providers=providers)
segmentor = FastSAM(model_variant="FastSAM-x", providers=providers)

# Initialize CNOS with FastSAM segmentor
cnos = CNOS(
    model_path="cad_models/hope_object_01.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
    providers=providers,
)

result = cnos.inference("table_scene.png", num_max_dets=3, conf_threshold=0.15)
result.visualize_mask(save_path="cnos_fastsam_detection.png")
```

### Usage with SAM Segmentor (High-Precision AMG)

```python
from spatialhub import CNOS, DINOV2, SAM

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize sub-adapters
descriptor = DINOV2(model_variant="dinov2_vitl14", providers=providers)
segmentor = SAM(model_variant="sam_vit_h", providers=providers)

# Initialize CNOS with SAM segmentor
cnos = CNOS(
    model_path="cad_models/hope_object_01.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
    providers=providers,
)

result = cnos.inference("table_scene.png", num_max_dets=3, conf_threshold=0.15)
result.visualize_mask(save_path="cnos_sam_detection.png")
```

---

## Returned Result Data Structure

Returns a [`SegmentationResult`](../core-and-utils/structures/segmentation_result.md) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Matched bounding box coordinates `[x1, y1, x2, y2]`. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Matched binary segment masks. |
| `scores` | `np.ndarray` | `(N,)` float32 | Top-$k$ aggregated cosine similarity matching scores. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numerical class index array. |
| `class_names` | `list[str] \| None` | Length `N` | CAD object name strings. |
