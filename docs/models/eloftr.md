# EfficientLoFTR Technical Reference

`spatialhub.models.efficient_loftr` provides an ONNX Runtime adapter for **EfficientLoFTR**, a semi-dense local feature matching model using sparse transformers.

---

## Supported Model Variants

The `EfficientLoFTRAdapter` accepts two model precision variants via `model_type`:

| Model Variant (`model_type`) | ONNX File | Description | Target Use Case |
| :--- | :--- | :--- | :--- |
| `"full"` (Default) | `eloftr_outdoor_full.onnx` | Full precision semi-dense feature matching model. | Maximum matching precision & robust keypoint coverage. |
| `"opt"` | `eloftr_outdoor_opt.onnx` | Optimized/quantized lightweight model variant. | High throughput, lower memory footprint, and real-time edge processing. |

---

## Overview & Mathematical Preprocessing

EfficientLoFTR matches coarse-to-fine keypoints across image pairs without requiring PyTorch during inference. Preprocessing scales inputs dynamically to spatial dimensions divisible by 32, pads image pairs to matching spatial shapes, and projects keypoints back to original coordinate spaces.

### Dimension Alignment

Sparse transformer feature maps downsample spatial dimensions by a factor of 32. Input spatial dimensions $(W_{curr}, H_{curr})$ are scaled to nearest lower multiples of 32:

$$
W_{new} = \max\left(32,\ \left\lfloor\frac{W_{curr}}{32}\right\rfloor \times 32\right)
$$

$$
H_{new} = \max\left(32,\ \left\lfloor\frac{H_{curr}}{32}\right\rfloor \times 32\right)
$$

### Coordinate Projection

Scale factors $S = [S_x, S_y]$ project raw ONNX output coordinates $P_{raw} = (x_{raw}, y_{raw})$ back to original image coordinates $P_{orig}$:

$$
S_x = \frac{W_{orig}}{W_{new}}, \qquad S_y = \frac{H_{orig}}{H_{new}}
$$

$$
P_{orig} = (x_{raw} \cdot S_x,\; y_{raw} \cdot S_y)
$$

### Boundary Filtering

Tensors are zero-padded to $\max(H_a, H_b) \times \max(W_a, W_b)$. Matches inside padded regions are filtered out using boundary mask $V$:

$$
V = \{ (P_0, P_1) \mid x_0 < W_a \land y_0 < H_a \land x_1 < W_b \land y_1 < H_b \}
$$

---

## ONNX Export Guide

The export environment for EfficientLoFTR sits beside its `pyproject.toml` file at `src/spatialhub/models/efficient_loftr`.

### Environment Setup

```bash
cd src/spatialhub/models/efficient_loftr
uv sync
```

### Running Export Script

```bash
uv run python export_onnx.py \
    --checkpoint weights/eloftr_outdoor.ckpt \
    --output-path weights/eloftr_outdoor.onnx \
    --opset 17 \
    --device cpu
```

---

## SpatialHub Adapter API & Usage

```python
from spatialhub import EfficientLoFTR

# Initialize with 'opt' variant and CUDA acceleration
matcher = EfficientLoFTR(
    model_type="opt",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Execute matching between two images
result = matcher.match("image_a.jpg", "image_b.jpg", max_dim=1024)

# Render side-by-side visualization
result.visualize(top_k=50, save_path="matches.png")
```

---

## Returned Result Data Structure

Returns a [`MatchResult`](../core-and-utils/structures/match_result.md) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image_a` | `str \| Path \| np.ndarray` | Input | First image reference or NumPy array. |
| `image_b` | `str \| Path \| np.ndarray` | Input | Second image reference or NumPy array. |
| `keypoints_a` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_a`. |
| `keypoints_b` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_b`. |
| `confidence` | `np.ndarray` | `(N,)` float32 | Match confidence scores `[0.0, 1.0]`. |
