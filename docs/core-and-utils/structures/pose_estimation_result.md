# PoseEstimationResult

`spatialhub.structures.PoseEstimationResult` encapsulates estimated 6D object poses ($SE(3)$ transformation matrices) and confidence scores produced by 6D pose estimators and trackers such as [`FoundationPoseAdapter`](../../models/foundationpose.md).

```python
from spatialhub.structures import PoseEstimationResult
```

---

## Fields

| Field | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `poses` | `np.ndarray` | `(N, 4, 4)` float32 | Estimated 4x4 object-to-camera transformation matrices $[R \mid t]$. |
| `intrinsics` | `np.ndarray` | `(3, 3)` float32 | Camera intrinsic matrix $K$. |
| `scores` | `np.ndarray \| None` | `(N,)` float32 | Pose confidence scores. |
| `labels` | `list[str] \| None` | Length `N` | Object class or model name identifiers. |
| `bbox_3d` | `np.ndarray \| None` | `(N, 8, 3)` or `(8, 3)` float32 | 3D bounding box corners in canonical centered CAD space. |
| `to_origin` | `np.ndarray \| None` | `(N, 4, 4)` or `(4, 4)` float32 | Centering transform from raw CAD origin to centered frame. |

---

## Properties

| Property | Returns | Description |
| :--- | :--- | :--- |
| `best_pose` | `np.ndarray (4, 4)` | The pose with the highest confidence score. If `scores` is `None`, returns `poses[0]`. |
| `best_score` | `float \| None` | The highest confidence value, or `None` if no scores are available. |

---

## Methods

### `visualize`

Projects the 3D bounding box wireframe and Cartesian XYZ coordinate axes onto the input image. Calls [`draw_projected_3d_box`](../viz.md#draw_projected_3d_box) and [`draw_3d_axis`](../viz.md#draw_3d_axis) internally.

```python
vis = result.visualize(draw_bbox=True, draw_axes=True, axis_length=0.05, save_path="pose.png")
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `draw_bbox` | `bool` | `True` | Draw the projected 3D bounding box wireframe. Requires `bbox_3d`. |
| `draw_axes` | `bool` | `True` | Draw XYZ coordinate axes. Requires `to_origin`. |
| `axis_length` | `float` | `0.05` | Axis length in meters. |
| `box_color` | `tuple[int, int, int]` | `(0, 255, 0)` | RGB color for the bounding box wireframe. |
| `save_path` | `str \| Path \| None` | `None` | Optional disk path to write the annotated image. |

#### Return Value
* **`np.ndarray`**: Annotated uint8 RGB image array of shape `(H, W, 3)`.
