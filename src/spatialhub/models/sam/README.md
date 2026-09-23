# Segment Anything Model (SAM) Technical Reference

`spatialhub.models.sam` provides an ONNX Runtime adapter for **Segment Anything Model (SAM)**, running Automatic Mask Generation (AMG) via decoupled Image Encoder and Mask Decoder ONNX sessions.

---

## 1. Supported Model Variants

`SAMAdapter` supports 3 Vision Transformer backbone variants via `model_variant`:

| Model Variant (`model_variant`) | Encoder ONNX File | Decoder ONNX File | Description |
| :--- | :--- | :--- | :--- |
| `"vit_h"` (Default) | `vit_h_encoder.onnx` | `vit_h_decoder.onnx` | ViT-Huge backbone variant. |
| `"vit_l"` | `vit_l_encoder.onnx` | `vit_l_decoder.onnx` | ViT-Large backbone variant. |
| `"vit_b"` | `vit_b_encoder.onnx` | `vit_b_decoder.onnx` | ViT-Base backbone variant. |

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

Export Segment Anything (SAM) image encoder and mask decoder directly to ONNX format using the centralized export utility:

```bash
uv run tools/export/export_sam.py \
    --variant vit_h \
    --output-folder onnx_weight \
    --opset 17
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--variant` | `str` | `"vit_b"` | SAM model variant to export (`vit_b`, `vit_l`, `vit_h`, or `all`). |
| `--checkpoint` | `str` | `None` | Path to checkpoint file (`.pth`) (downloaded if omitted). |
| `--output-folder` | `str` | `onnx_weight` | Destination directory for exported `.onnx` model files. |
| `--opset` | `int` | `17` | ONNX Operator Set version. |
| `--return-single-mask` | `bool` | `True` | Output single best mask proposal. |

---

## 4. SpatialHub Adapter API & Usage

```python
from spatialhub import SAM

# Initialize SAM adapter with ViT-H encoder/decoder ONNX sessions
segmentor = SAM(
    model_variant="vit_h",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Run Automatic Mask Generation (AMG) grid sampling
result = segmentor.generate_masks("landscape.jpg", points_per_side=32)

# Render colorized mask overlay
result.visualize_mask(save_path="sam_masks.png")
```

---

## 5. Returned Result Data Structure

Returns a [`SegmentationResult`](https://spatialhub-ai.github.io/spatialhub/core-and-utils/structures/segmentation_result/) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Bounding box coordinates `[x1, y1, x2, y2]`. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Binary spatial segment masks. |
| `scores` | `np.ndarray` | `(N,)` float32 | Predicted IoU confidence scores `[0.0, 1.0]`. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numerical class index array. |
| `class_names` | `list[str] \| None` | Length `N` | Class label name list. |
