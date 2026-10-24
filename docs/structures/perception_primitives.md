# Perception Primitives

The `spatialhub.structures` module provides standardized dataclasses for 2D spatial detections, dense embeddings, and two-view geometric correspondences emitted by vision adapters.

## `InstanceSegmentation`

Container for detected 2D object instances, segmentation masks, and bounding proposals.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `masks` | `np.ndarray` | `(N, H, W)` | `bool` or `uint8` | Binary instance masks. |
| `boxes` | `np.ndarray` | `(N, 4)` | `float32` | Bounding box coordinates in $[x_1, y_1, x_2, y_2]$ pixel format. |
| `scores` | `np.ndarray` | `(N,)` | `float32` | Detection confidence scores in $[0.0, 1.0]$. |
| `ids` | `np.ndarray | None` | `(N,)` | `int64` | Optional class or instance identifier integers. |
| `names` | `list[str] | None` | length $N$ | `str` | Optional class label names. |

### Validation and Slicing

- When initialized with unbatched 2D masks `(H, W)` or a single box `(4,)`, dimensions are automatically expanded to leading batch size $N=1$.
- Slicing along the instance dimension synchronizes all present attributes (`masks`, `boxes`, `scores`, `ids`, `names`):

```python
import numpy as np
from spatialhub.structures import InstanceSegmentation

instances = InstanceSegmentation(
    masks=np.zeros((10, 480, 640), dtype=bool),
    boxes=np.zeros((10, 4), dtype=np.float32),
    scores=np.linspace(0.95, 0.45, 10, dtype=np.float32),
    names=[f"class_{i}" for i in range(10)],
)

# filter by confidence threshold
high_conf = instances[instances.scores > 0.7]
print("Remaining count:", len(high_conf))
print("Filtered masks shape:", high_conf.masks.shape)
print("Filtered boxes shape:", high_conf.boxes.shape)
```

---

## `KeypointMatches`

Represents pairwise 2D point correspondences between two camera views (e.g. from local feature matchers such as `EfficientLoFTR`).

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `keypoints_a` | `np.ndarray` | `(N, 2)` | `float32` | Matched keypoints in View A in $(x, y)$ pixel coordinates. |
| `keypoints_b` | `np.ndarray` | `(N, 2)` | `float32` | Matched keypoints in View B in $(x, y)$ pixel coordinates. |
| `confidence` | `np.ndarray` | `(N,)` | `float32` | Match confidence scores in $[0.0, 1.0]$. |

### Correspondence Slicing

Array lengths across `keypoints_a`, `keypoints_b`, and `confidence` are strictly verified for equality in `__post_init__`. Slicing preserves correspondences:

```python
import numpy as np
from spatialhub.structures import KeypointMatches

matches = KeypointMatches(
    keypoints_a=np.random.rand(500, 2).astype(np.float32),
    keypoints_b=np.random.rand(500, 2).astype(np.float32),
    confidence=np.random.rand(500).astype(np.float32),
)

# extract top-50 matches
top_matches = matches[np.argsort(matches.confidence)[-50:]]
print("Top match count:", len(top_matches))
```

---

## `FeatureEmbedding`

Represents global image representations or dense spatial feature tensors (e.g. from `DINOv2`).

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `features` | `np.ndarray` | `(N, D)` or `(N, H, W, D)` | `float32` | Floating-point embedding tensor. |
| `embedding_type` | `Literal["global", "dense"]` | scalar | `str` | Representation format (`"global"` or `"dense"`). |
| `l2_normalized` | `bool` | scalar | `bool` | Whether embeddings are unit L2-normalized along channel axis $D$. |

### Dimension Constraints

- For `embedding_type="global"`, `features.ndim` must equal 2: `(N, D)`.
- For `embedding_type="dense"`, `features.ndim` must equal 4: `(N, H, W, D)`.

```python
import numpy as np
from spatialhub.structures import FeatureEmbedding

dense_feats = FeatureEmbedding(
    features=np.random.randn(1, 32, 32, 768).astype(np.float32),
    embedding_type="dense",
    l2_normalized=True,
)
print("Feature map tensor shape:", dense_feats.features.shape)
```

---

## Related Utilities

- [`mask_to_bbox`](../utilities/image.md#mask_to_bbox): Computes $[x_1, y_1, x_2, y_2]$ bounding boxes from binary masks.
- [`nms_boxes`](../utilities/image.md#nms_boxes): Applies non-maximum suppression on bounding boxes and scores.
- [`blend_masks`](../utilities/visualization.md#blend_masks): Overlays instance masks on RGB images with alpha transparency.
- [`draw_boxes_2d`](../utilities/visualization.md#draw_boxes_2d): Annotates 2D bounding boxes and confidence labels on images.
- [`visualize_matched_keypoints`](../utilities/visualization.md#visualize_matched_keypoints): Renders dual-panel correspondence lines between image pairs.
- [`visualize_dense_features`](../utilities/visualization.md#visualize_dense_features): Projects high-dimensional feature embeddings to RGB via PCA.
- [`visualize_global_features`](../utilities/visualization.md#visualize_global_features): Visualizes pairwise cosine similarity matrices across global vectors.
