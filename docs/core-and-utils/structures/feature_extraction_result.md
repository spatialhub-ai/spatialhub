# FeatureExtractionResult

`spatialhub.structures.FeatureExtractionResult` contains dense patch tokens or global CLS embedding vectors produced by feature extraction adapters such as [`DINOv2Adapter`](../../models/dinov2.md).

```python
from spatialhub.structures import FeatureExtractionResult
```

---

## Fields

| Field | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `images` | `np.ndarray | list | str | Path` | `(N, H, W, 3)`, `(H, W, 3)`, or path | Preprocessed input image batch or paths. |
| `features` | `FeatureEmbedding` | `(N, D)` or `(N, H, W, D)` float32 | Extracted feature embedding tensor ($D=1024$ for ViT-L/14). |
| `embedding_type` | `Literal["global", "dense"]` | `"global"` | Embedding resolution: `"global"` (single CLS token per image) or `"dense"` (patch tokens). |
| `l2_normalized` | `bool` | `False` | Boolean indicating whether embedding vectors satisfy $\lVert v \rVert_2 = 1.0$. |

---

## Usage Example

```python
from spatialhub import DINOv2

extractor = DINOv2(model_variant="dinov2_vitl14")
result = extractor.extract_features(["frame_a.png", "frame_b.png"])

print(f"Features shape: {result.features.shape}")  # (2, 1024)
print(f"L2 Normalized: {result.l2_normalized}")    # True

# Compute pairwise cosine similarity
similarity = result.features[0] @ result.features[1].T
print(f"Cosine Similarity: {similarity:.4f}")
```
