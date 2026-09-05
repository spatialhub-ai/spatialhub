# Visualization Utilities

`spatialhub.utils.viz` provides drawing and annotation functions for keypoint correspondences, segmentation masks, projected 3D bounding boxes, and coordinate frame axes.

```python
from spatialhub.utils import (
    visualize_matches,
    visualize_masks,
    draw_projected_3d_box,
    draw_3d_axis,
)
```

---

## Architectural Binding with `spatialhub.structures`

The visualization functions in this module are stateless, pure NumPy/OpenCV drawing routines. They do not import or depend on `spatialhub.structures`.

Instead, the return contracts in [`spatialhub.structures`](structures/overview.md) invoke these visualization functions within their `.visualize()` or `.visualize_mask()` convenience methods:

| Data Structure Method | Underlying Visualization Function |
| :--- | :--- |
| [`MatchResult.visualize()`](structures/match_result.md) | `visualize_matches` |
| [`SegmentationResult.visualize_mask()`](structures/segmentation_result.md) | `visualize_masks` |
| [`PoseEstimationResult.visualize()`](structures/pose_estimation_result.md) | `draw_projected_3d_box` and `draw_3d_axis` |

---

## `visualize_matches`

Draws matched keypoint correspondences across two images placed side by side.

```python
vis = visualize_matches(img0, img1, mkpts0, mkpts1, mconf=conf, top_k=100)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `img0_input` | `str \| Path \| np.ndarray` | required | First image path or array. |
| `img1_input` | `str \| Path \| np.ndarray` | required | Second image path or array. |
| `mkpts0` | `np.ndarray (N, 2)` | required | Keypoints in first image `[x, y]`. |
| `mkpts1` | `np.ndarray (N, 2)` | required | Keypoints in second image `[x, y]`. |
| `mconf` | `np.ndarray (N,) \| None` | `None` | Match confidence scores. |
| `conf_thresh` | `float` | `0.5` | Minimum confidence threshold to render. |
| `max_side` | `int` | `800` | Maximum spatial dimension for output canvas. |
| `top_k` | `int \| None` | `None` | Limit to top $k$ matches by confidence. |
| `save_path` | `str \| Path \| None` | `None` | Path to save output image file. |

### Return Value
* **`np.ndarray`**: uint8 BGR canvas of shape `(H, W0 + W1, 3)`.

---

## `visualize_masks`

Renders colorized instance segmentation masks, contour edges, bounding boxes, and score labels using golden-angle hue distribution.

```python
vis = visualize_masks(image, boxes, masks, scores, save_path="out.png")
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray (H, W, 3)` uint8 | required | Input RGB image. |
| `boxes` | `np.ndarray (N, 4)` | required | Bounding boxes in `[x1, y1, x2, y2]` format. |
| `masks` | `np.ndarray (N, H, W)` bool | required | Binary masks per detection. |
| `scores` | `np.ndarray (N,)` | required | Confidence scores. |
| `save_path` | `str \| Path \| None` | `None` | Path to save output image file. |
| `alpha` | `float` | `0.4` | Transparency blend factor ($0.0 \dots 1.0$). |

### Return Value
* **`np.ndarray`**: Annotated uint8 RGB image array.

---

## `draw_projected_3d_box`

Transforms canonical 3D bounding box corners into camera coordinates using `pose`, projects them to pixel space with `intrinsics`, and draws wireframe box edges.

```python
vis = draw_projected_3d_box(image, pose, K, bbox_corners_3d)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray (H, W, 3)` uint8 | required | Base RGB image. |
| `pose` | `np.ndarray (4, 4)` | required | 4x4 object-to-camera transform. |
| `intrinsics` | `np.ndarray (3, 3)` | required | 3x3 camera intrinsic matrix. |
| `bbox_corners_3d` | `np.ndarray (8, 3)` | required | Canonical 3D corner coordinates. |
| `color` | `tuple[int, int, int]` | `(0, 255, 0)` | Line color in RGB. |
| `thickness` | `int` | `2` | Line thickness in pixels. |

### Return Value
* **`np.ndarray`**: Annotated uint8 RGB image array.

---

## `draw_3d_axis`

Draws 3D Cartesian coordinate axes (+X Red, +Y Green, +Z Blue) projected at the object's origin.

```python
vis = draw_3d_axis(image, pose, K, to_origin=mesh_offset_matrix)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray (H, W, 3)` uint8 | required | Base RGB image. |
| `pose` | `np.ndarray (4, 4)` | required | 4x4 object-to-camera transform. |
| `intrinsics` | `np.ndarray (3, 3)` | required | 3x3 camera intrinsic matrix. |
| `to_origin` | `np.ndarray (4, 4) \| None` | `None` | Centering offset transform matrix. |
| `axis_length` | `float` | `0.05` | Length of axes in meters. |
| `thickness` | `int` | `2` | Line thickness in pixels. |

### Return Value
* **`np.ndarray`**: Annotated uint8 RGB image array.
