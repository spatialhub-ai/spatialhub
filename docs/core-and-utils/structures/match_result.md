# MatchResult

`spatialhub.structures.MatchResult` represents 2D keypoint correspondences and match confidence scores produced by feature matching adapters such as [`EfficientLoFTRAdapter`](../../models/eloftr.md).

```python
from spatialhub.structures import MatchResult
```

---

## Fields

| Field | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image_a` | `str \| Path \| np.ndarray` | `(H, W, C)` or path | First input image — path or in-memory array. |
| `image_b` | `str \| Path \| np.ndarray` | `(H, W, C)` or path | Second input image — path or in-memory array. |
| `keypoints_a` | `np.ndarray` | `(N, 2)` float32 | Matched keypoint pixel coordinates `[x, y]` in `image_a`. |
| `keypoints_b` | `np.ndarray` | `(N, 2)` float32 | Matched keypoint pixel coordinates `[x, y]` in `image_b`. |
| `confidence` | `np.ndarray` | `(N,)` float32 | Per-match confidence scores in `[0.0, 1.0]`. |

---

## Methods

### `visualize`

Renders a side-by-side keypoint correspondence canvas. Calls [`visualize_matches`](../viz.md#visualize_matches) internally.

```python
vis = result.visualize(conf_thresh=0.5, max_side=800, top_k=50, save_path="matches.png")
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `conf_thresh` | `float` | `0.5` | Minimum confidence score threshold to draw a match. |
| `max_side` | `int` | `800` | Caps the longest image dimension in the output visualization canvas. |
| `top_k` | `int \| None` | `None` | Limit display to top $k$ matches by confidence score. |
| `save_path` | `str \| Path \| None` | `None` | Optional disk path to write the annotated image. |

#### Return Value
* **`np.ndarray`**: uint8 BGR visualization canvas of shape `(H, W_a + W_b, 3)`.
