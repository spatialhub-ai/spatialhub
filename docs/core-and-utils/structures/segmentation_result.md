# SegmentationResult

`spatialhub.structures.SegmentationResult` represents spatial binary masks, bounding boxes, and confidence scores produced by segmentation, proposal, and zero-shot detection models such as [`FastSAMAdapter`](../../models/fastsam.md), [`SAMAdapter`](../../models/sam.md), and [`CNOSAdapter`](../../models/cnos.md).

```python
from spatialhub.structures import SegmentationResult
```

---

## Fields

| Field | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Bounding boxes in `[x1, y1, x2, y2]` pixel coordinate format. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Binary spatial masks, one per candidate detection. |
| `scores` | `np.ndarray` | `(N,)` float32 | Detection or match confidence scores. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numeric class indices. `None` for class-agnostic proposals. |
| `class_names` | `list[str] \| None` | Length `N` | String class or CAD object labels. |

---

## Methods

### `visualize_mask`

Renders colorized semi-transparent mask overlays, bounding box outlines, and confidence scores. Calls [`visualize_masks`](../viz.md#visualize_masks) internally.

```python
vis = result.visualize_mask(save_path="segmentation_overlay.png", alpha=0.4)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `save_path` | `str \| Path \| None` | `None` | Optional disk path to write the annotated image. |
| `alpha` | `float` | `0.4` | Mask blend transparency ($0.0 \dots 1.0$). |

#### Return Value
* **`np.ndarray`**: Annotated uint8 RGB image array of shape `(H, W, 3)`.
