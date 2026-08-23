# FastSAM Technical Reference

`spatialhub.models.fastsam` provides an ONNX Runtime adapter for **FastSAM (YOLOv8-Seg)**, performing real-time instance segmentation and mask proposal generation.

---

## 1. Supported Model Variants

`FastSAMAdapter` supports 2 YOLOv8-Seg model variants via `model_variant`:

| Model Variant (`model_variant`) | ONNX File | Description | Target Performance |
| :--- | :--- | :--- | :--- |
| `"FastSAM-x"` or `"x"` (Default) | `FastSAM-x.onnx` | Extra Large YOLOv8-Seg backbone. | Highest segmentation proposal quality and fine mask boundary precision. |
| `"FastSAM-s"` or `"s"` | `FastSAM-s.onnx` | Small YOLOv8-Seg backbone. | Lightweight real-time proposal generation for edge/mobile devices. |

---

## 2. Overview & Mathematical Preprocessing

FastSAM decodes bounding boxes, confidence scores, and prototype mask coefficient matrices from a YOLOv8-Seg ONNX graph output tensor of shape $(1, 37, 8400)$ and prototype tensor of shape $(1, 32, 160, 160)$.

### Prototype Mask Synthesis

Binary spatial segment masks $M \in \mathbb{B}^{H \times W}$ are computed by matrix-multiplying mask coefficient vectors $C \in \mathbb{R}^{N \times 32}$ with spatial prototype tensor $P \in \mathbb{R}^{32 \times 160 \times 160}$, applying sigmoid activation, and cropping to candidate bounding boxes:

$$
M_{raw} = \sigma\left( C \cdot P \right) = \frac{1}{1 + e^{-(C \cdot P)}}
$$

$$
M_{binary} = M_{raw} > 0.5
$$

### Non-Maximum Suppression (NMS)

Vectorized NMS filters candidate bounding boxes $B$ based on Intersection over Union (IoU) ratio:

$$
\text{IoU}(B_i, B_j) = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i \cup B_j)}
$$

---

## 3. ONNX Export Guide

The export environment for FastSAM sits beside its `pyproject.toml` file at `src/spatialhub/models/fastsam/FastSAM`.

### Environment Setup

```bash
cd src/spatialhub/models/fastsam/FastSAM
uv sync
```

### Running Export Script

```bash
uv run python export_onnx.py \
    --checkpoint FastSAM-x.pt \
    --output-folder ./onnx_model \
    --imgsz 640 \
    --opset 17 \
    --dynamic
```

---

## 4. SpatialHub Adapter API & Usage

```python
from spatialhub import FastSAM

# Initialize FastSAM adapter
segmentor = FastSAM(
    model_variant="FastSAM-x",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Generate segment proposals
result = segmentor.generate_masks("scene.png", conf_threshold=0.3, iou_threshold=0.7)

# Render colorized segment overlay image
result.visualize_mask(save_path="fastsam_masks.png")
```

---

## 5. Returned Result Data Structure

Returns a [`SegmentationResult`](../structures/overview.md#4-segmentationresult) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Bounding box coordinates `[x1, y1, x2, y2]`. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Binary spatial segment masks. |
| `scores` | `np.ndarray` | `(N,)` float32 | Detection confidence scores `[0.0, 1.0]`. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numerical class index array. |
| `class_names` | `list[str] \| None` | Length `N` | Class label name list. |
