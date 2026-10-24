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
| `DINOv2Adapter` (Default: `"vitb14"`) | `"dinov2_vits14"`, `"dinov2_vitb14"`, `"dinov2_vitl14"`, `"dinov2_vitg14"` | Extracts L2-normalized CLS token embeddings for top-$k$ cosine similarity matching. |

> [!NOTE]
> Coarser backbones (`"dinov2_vits14"`, `"dinov2_vitb14"`) prioritize global object silhouette and boundary completeness. Higher-capacity variants (`"dinov2_vitl14"`, `"dinov2_vitg14"`) attend sharply to local text, logos, and high-frequency patterns, which can rank textured sub-components higher than complete objects.

---

## Overview & Mathematical Preprocessing

CNOS loads a 3D CAD mesh file (`.ply`, `.obj`, `.stl`), renders 2D template views across pre-computed camera poses, extracts DINOv2 feature embeddings for each template, and matches image segment proposals via cosine similarity.

### CAD Template Rendering

Using `TemplateRenderer`, $M$ template views $I_{rgba} \in \mathbb{U}^{H \times W \times 4}$ are rendered at fixed camera poses $P_m = [R_m \mid t_m]$.

### Bounding Box Cropping & Preprocessing

Each rendered template is cropped to its foreground bounding box $[x1, y1, x2, y2]$, square-padded, resized to $224 \times 224$, and normalized using ImageNet statistics.

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

## Numerical Parity Verification

Numerical parity evaluates mathematical agreement between the original PyTorch reference models and the SpatialHub ONNX Runtime adapter on demo datasets using DINOv2 `vitl14` and FastSAM `s` on CUDA ($42$ template viewpoints, top-$k=5$).

| Dataset | Descriptor | Segmentor | Ref Feat MAE | Ref Feat Max Diff | Ref Cosine Sim | Scene Feat MAE | Scene Cosine Sim | Sim Matrix MAE | Top-1 Score Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hope_bba_sauce` | `vitl14` | `s` | `0.000088` | `0.004889` | **99.9969%** | `0.000065` | **99.9995%** | `0.000195` | `0.007040` |
| `needle_holder_tools` | `vitl14` | `s` | `0.000080` | `0.000824` | **99.9993%** | `0.000074` | **99.9945%** | `0.000211` | `0.034284` |

---

## Performance Benchmarks

Latency, throughput, and memory footprint measured across $1$ unmeasured warmup iteration and $5$ timed measurement iterations at $640 \times 480$ scene resolution (DINOv2 `vitl14` + FastSAM `s`, $42$ template viewpoints, $50$ proposals).

### CUDA Execution (`CUDAExecutionProvider`)

| Stage | Device | Latency Mean (ms) | Median (ms) | P95 (ms) | Throughput (FPS) | Peak RAM Delta | Peak VRAM Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Template Caching (42 Views)** | `CUDA` | 6369.10 +/- 582.42 | 6309.89 | 7368.29 | **0.2** | +46.5 MB | +389.3 MB |
| **Proposal Gen (FastSAM-s)** | `CUDA` | 118.69 +/- 15.55 | 112.53 | 148.23 | **8.4** | +1.1 MB | +36.0 MB |
| **Descriptor Extract (50 Crops)** | `CUDA` | 7068.42 +/- 374.15 | 7129.07 | 7530.17 | **0.1** | +9.3 MB | 0.0 MB |
| **Cosine Matching & Top-K** | `CUDA` | 2.00 +/- 0.28 | 1.96 | 2.60 | **500.0** | +0.1 MB | 0.0 MB |
| **End-to-End CNOS Pipeline** | `CUDA` | 6469.88 +/- 204.41 | 6433.79 | 6610.64 | **0.2** | +1.9 MB | 0.0 MB |

---

## Tooling & Verification Commands

### Performance Profile

Benchmark stage-by-stage latency, throughput, and memory footprint across execution providers:

```bash
uv run tools/benchmark/profile_cnos.py \
    --provider cuda \
    --descriptor-variant vitl14 \
    --segmentor-variant s \
    --num-viewpoints 42 \
    --warmup 1 \
    --iters 5 \
    --output-file .profile/profile_cnos.md
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--provider` | `str` | `"cuda"` | Target execution provider filter (`cuda`, `cpu`, or `all`). |
| `--descriptor-variant` | `str` | `"vitl14"` | DINOv2 model variant (`vits14`, `vitb14`, `vitl14`, `vitg14`). |
| `--segmentor-variant` | `str` | `"s"` | FastSAM model variant (`s` or `x`). |
| `--mesh-path` | `str` | `None` | Optional path to 3D CAD model file (.obj, .ply). |
| `--num-viewpoints` | `int` | `42` | Number of spherical template viewpoints to render. |
| `--warmup` | `int` | `1` | Number of unmeasured warmup iterations. |
| `--iters` | `int` | `5` | Number of timed measurement iterations. |
| `--output-file` | `str` | `None` | Optional path to save markdown benchmark report. |

### Parity Check

Evaluate numerical parity between PyTorch reference models and SpatialHub ONNX Runtime adapter on CUDA:

```bash
uv run tools/benchmark/parity_cnos.py \
    --descriptor-variant vitl14 \
    --segmentor-variant s \
    --output-file .profile/parity_cnos.md
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--descriptor-variant` | `str` | `"vitl14"` | DINOv2 model variant (`vits14`, `vitb14`, `vitl14`, `vitg14`). |
| `--segmentor-variant` | `str` | `"s"` | FastSAM model variant (`s` or `x`). |
| `--mesh-path` | `str` | `None` | Optional custom CAD mesh path. |
| `--image-path` | `str` | `None` | Optional custom scene image path. |
| `--num-viewpoints` | `int` | `42` | Number of spherical template viewpoints to render. |
| `--output-file` | `str` | `None` | Optional path to append Markdown parity report. |

---

## SpatialHub Adapter API & Usage

### Usage with FastSAM Segmentor (Real-Time)

```python
from spatialhub import CNOS, DINOv2, FastSAM
from spatialhub.utils import blend_masks, draw_boxes_2d, load_image, save_image

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize sub-adapters
descriptor = DINOv2(model_variant="dinov2_vitl14", providers=providers)
segmentor = FastSAM(model_variant="FastSAM-x", providers=providers)

# Initialize CNOS from 3D CAD mesh
cnos = CNOS.from_cad_model(
    model="cad_models/hope_object_01.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
)

image = load_image("table_scene.png")
result = cnos.inference(image, num_max_dets=3, conf_threshold=0.15)

vis = blend_masks(image, result.masks)
vis = draw_boxes_2d(vis, result.boxes, scores=result.scores, names=result.names)
save_image(vis, "cnos_fastsam_detection.png")
```

### Usage with SAM Segmentor (High-Precision AMG)

```python
from spatialhub import CNOS, DINOv2, SAM
from spatialhub.utils import blend_masks, draw_boxes_2d, load_image, save_image

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize sub-adapters
descriptor = DINOv2(model_variant="dinov2_vitl14", providers=providers)
segmentor = SAM(model_variant="sam_vit_h", providers=providers)

# Initialize CNOS from 3D CAD mesh
cnos = CNOS.from_cad_model(
    model="cad_models/hope_object_01.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
)

image = load_image("table_scene.png")
result = cnos.inference(image, num_max_dets=3, conf_threshold=0.15)

vis = blend_masks(image, result.masks)
vis = draw_boxes_2d(vis, result.boxes, scores=result.scores, names=result.names)
save_image(vis, "cnos_sam_detection.png")
```

---

## Returned Result Data Structure

Returns an [`InstanceSegmentation`](https://spatialhub-ai.github.io/spatialhub/structures/perception_primitives/#instancesegmentation) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Matched binary segment masks. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Matched bounding box coordinates in `[x1, y1, x2, y2]` pixel format. |
| `scores` | `np.ndarray` | `(N,)` float32 | Top-$k$ aggregated cosine similarity matching scores in `[0.0, 1.0]`. |
| `ids` | `np.ndarray | None` | `(N,)` int | Numerical class index array. |
| `names` | `list[str] | None` | Length `N` | CAD object name strings. |
