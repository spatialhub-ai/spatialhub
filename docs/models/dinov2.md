# DINOv2 Technical Reference

`spatialhub.models.dinov2` provides an ONNX Runtime adapter for **DINOv2**, extracting global L2-normalized CLS token feature embeddings from images.

---

## Supported Model Variants

`DINOv2Adapter` supports 4 Vision Transformer backbone variants via `model_variant`:

| Model Variant (`model_variant`) | ONNX File | Output Dimension ($D$) | Description |
| :--- | :--- | :--- | :--- |
| `"dinov2_vits14"` or `"vits14"` | `dinov2_vits14.onnx` | $D = 384$ | ViT-Small/14 backbone (fastest runtime, minimal memory). |
| `"dinov2_vitb14"` or `"vitb14"` | `dinov2_vitb14.onnx` | $D = 768$ | ViT-Base/14 backbone (balanced descriptor performance). |
| `"dinov2_vitl14"` or `"vitl14"` (Default) | `dinov2_vitl14.onnx` | $D = 1024$ | ViT-Large/14 backbone (high semantic discriminability). |
| `"dinov2_vitg14"` or `"vitg14"` | `dinov2_vitg14.onnx` | $D = 1536$ | ViT-Giant/14 backbone (flagship descriptor accuracy). |

---

## Overview & Mathematical Preprocessing

DINOv2 accepts RGB image inputs, crops/resizes them to a fixed spatial resolution of $224 \times 224$, normalizes channels using ImageNet mean and standard deviation, and outputs feature vector embeddings.

### ImageNet Normalization

For each image channel $c \in \{R, G, B\}$, pixel values $x \in [0, 1]$ are normalized:

$$
x_{norm} = \frac{x - \mu_c}{\sigma_c}
$$

Where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$.

### L2 Feature Normalization

Extracted CLS token feature vectors $v \in \mathbb{R}^D$ are normalized to unit L2 length:

$$
v_{norm} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^D v_i^2}}
$$

---

## ONNX Export Guide

The export environment for DINOv2 sits beside its `pyproject.toml` file at `src/spatialhub/models/dinov2/DINOv2`.

### Environment Setup

```bash
cd src/spatialhub/models/dinov2/DINOv2
uv sync
```

### Running Export Script

```bash
uv run python export_onnx.py \
    --model-name dinov2_vitl14 \
    --output-folder ./onnx_model \
    --opset 17
```

---

## SpatialHub Adapter API & Usage

```python
from spatialhub import DINOV2

# Initialize DINOv2 adapter with ViT-L/14 backbone (1024-dim)
extractor = DINOV2(
    model_variant="dinov2_vitl14",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Extract L2-normalized feature embedding vector
result = extractor.extract_features("object.png", l2_normalize=True)

print("Features shape:", result.features.shape)  # (1, 1024)
print("Is L2 normalized:", result.l2_normalized) # True
```

---

## Returned Result Data Structure

Returns a [`FeatureExtractionResult`](../core-and-utils/structures/feature_extraction_result.md) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `images` | `np.ndarray` | `(N, 224, 224, 3)` uint8 | Preprocessed RGB image batch. |
| `features` | `np.ndarray` | `(N, D)` float32 | Extracted feature embedding vectors. |
| `embedding_type` | `str` | N/A | Feature scope (`"global"`). |
| `l2_normalized` | `bool` | N/A | Flag indicating if vectors are unit L2-normalized. |
