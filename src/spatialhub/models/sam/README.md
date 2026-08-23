# Segment Anything Model (SAM) Technical Reference

`spatialhub.models.sam` provides an ONNX Runtime adapter for **Segment Anything Model (SAM)**, running Automatic Mask Generation (AMG) via decoupled Image Encoder and Mask Decoder ONNX sessions.

---

## 1. Supported Model Variants

`SAMAdapter` supports 3 Vision Transformer backbone variants via `model_variant`:

| Model Variant (`model_variant`) | Encoder ONNX File | Decoder ONNX File | Description |
| :--- | :--- | :--- | :--- |
| `"sam_vit_h"` or `"vit_h"` (Default) | `sam_vit_h_encoder.onnx` | `sam_vit_h_decoder.onnx` | ViT-Huge backbone (highest mask quality & boundary precision). |
| `"sam_vit_l"` or `"vit_l"` | `sam_vit_l_encoder.onnx` | `sam_vit_l_decoder.onnx` | ViT-Large backbone (balanced memory footprint & speed). |
| `"sam_vit_b"` or `"vit_b"` | `sam_vit_b_encoder.onnx` | `sam_vit_b_decoder.onnx` | ViT-Base backbone (lightweight execution for fast inference). |

---

## 2. Overview & Mathematical Preprocessing

SAM processes images in two decoupled execution stages:
1. **Image Encoder ONNX Session:** Processes $(1, 3, 1024, 1024)$ input images and outputs $(1, 256, 64, 64)$ feature embedding maps.
2. **Mask Decoder ONNX Session:** Evaluates point coordinate grid prompts $(1, K, 2)$ over image embeddings to compute high-resolution binary spatial masks.

### Grid Point Sampling

Point prompts $(x_p, y_p)$ are sampled across a uniform spatial grid of density $G \times G$ (default $32 \times 32$):

$$
x_{p, i} = \frac{i + 0.5}{G} \times W, \qquad y_{p, j} = \frac{j + 0.5}{G} \times H
$$

### Mask-to-Box Extraction

Bounding boxes $[x_1, y_1, x_2, y_2]$ are extracted directly from non-zero indices of binary spatial masks $M$:

$$
x_1 = \min \{ x \mid M[y, x] = 1 \}, \qquad x_2 = \max \{ x \mid M[y, x] = 1 \}
$$

$$
y_1 = \min \{ y \mid M[y, x] = 1 \}, \qquad y_2 = \max \{ y \mid M[y, x] = 1 \}
$$

---

## 3. ONNX Export Guide

The export environment for SAM sits beside its `pyproject.toml` file at `src/spatialhub/models/sam/SAM`.

### Environment Setup

```bash
cd src/spatialhub/models/sam/SAM
uv sync
```

### Running Export Script

```bash
uv run python export_onnx.py \
    --model-type vit_h \
    --out-encoder ./onnx_model/sam_image_encoder.onnx \
    --out-decoder ./onnx_model/sam_mask_decoder.onnx \
    --opset 17
```

---

## 4. SpatialHub Adapter API & Usage

```python
from spatialhub import SAM

# Initialize SAM adapter with ViT-H encoder/decoder ONNX sessions
segmentor = SAM(
    model_variant="sam_vit_h",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Run Automatic Mask Generation (AMG) grid sampling
result = segmentor.generate_masks("landscape.jpg", points_per_side=32)

# Render colorized mask overlay
result.visualize_mask(save_path="sam_masks.png")
```

---

## 5. Returned Result Data Structure

Returns a [`SegmentationResult`](../../docs/structures/overview.md#4-segmentationresult) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Bounding box coordinates `[x1, y1, x2, y2]`. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Binary spatial segment masks. |
| `scores` | `np.ndarray` | `(N,)` float32 | Predicted IoU confidence scores `[0.0, 1.0]`. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numerical class index array. |
| `class_names` | `list[str] \| None` | Length `N` | Class label name list. |
