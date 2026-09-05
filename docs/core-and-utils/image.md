# Image Preprocessing

`spatialhub.utils.image` provides image I/O and preprocessing operations used across model adapters.

```python
from spatialhub.utils import (
    load_image,
    normalize_image,
    extract_foreground_bbox,
    square_crop_and_resize,
    non_max_suppression,
)
```

---

## `load_image`

Reads an image from disk or accepts an in-memory array, converting it to the requested channel format (`cv2.IMREAD_UNCHANGED`).

```python
img_rgb  = load_image("frame.png")                    # (H, W, 3) uint8
img_gray = load_image("frame.png", color_mode="GRAY") # (H, W)    uint8
img_rgba = load_image("frame.png", color_mode="RGBA") # (H, W, 4) uint8
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image_input` | `str \| Path \| np.ndarray` | required | File path or in-memory array. |
| `color_mode` | `"RGB" \| "RGBA" \| "GRAY"` | `"RGB"` | Target channel format. |

### Return Value
* **`np.ndarray`**: Grayscale output has shape `(H, W)`; RGB/RGBA has shape `(H, W, C)`.

---

## `normalize_image`

Casts an RGB image to float32, applies per-channel normalization `(pixel - mean) / std`, and optionally transposes the layout from `(H, W, C)` to `(C, H, W)`.

```python
normed = normalize_image(img_rgb, to_chw=True)  # (3, H, W) float32
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | required | RGB array of shape `(H, W, 3)`. |
| `mean` | `np.ndarray` | ImageNet mean | Per-channel mean values (default: `[0.485, 0.456, 0.406]`). |
| `std` | `np.ndarray` | ImageNet std | Per-channel std values (default: `[0.229, 0.224, 0.225]`). |
| `to_chw` | `bool` | `True` | Transpose output to `(C, H, W)` layout. |

### Return Value
* **`np.ndarray`**: Normalized float32 tensor of shape `(C, H, W)` or `(H, W, C)`.

---

## `extract_foreground_bbox`

Finds the tight bounding box `(x_min, y_min, x_max, y_max)` enclosing non-background pixels.

```python
bbox = extract_foreground_bbox(rgba_template)  # e.g. (45, 30, 210, 195)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | required | `(H, W, 3)` or `(H, W, 4)` uint8 array. |

### Return Value
* **`tuple[int, int, int, int]`**: `(x_min, y_min, x_max, y_max)` coordinates.

---

## `square_crop_and_resize`

Crops to `bbox`, pads the shorter dimension symmetrically with zeros to create a square canvas, and optionally resizes to `target_size x target_size`.

```python
square = square_crop_and_resize(img_rgba, bbox, target_size=224)  # (224, 224, C)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | required | Source image array `(H, W, C)`. |
| `bbox` | `tuple[int, int, int, int]` | required | Region to crop: `(x_min, y_min, x_max, y_max)`. |
| `target_size` | `int \| None` | `None` | Square target dimension. If `None`, returns unresized square canvas. |

### Return Value
* **`np.ndarray`**: Padded and resized square array.

---

## `non_max_suppression`

Greedy Non-Maximum Suppression (NMS) for bounding box filtering based on Intersection-over-Union (IoU) overlap.

```python
keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.7)
filtered_boxes = boxes[keep_indices]
```

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `boxes` | `np.ndarray (N, 4)` float32 | Bounding boxes in `[x1, y1, x2, y2]` format. |
| `scores` | `np.ndarray (N,)` float32 | Confidence scores. |
| `iou_threshold` | `float` | Overlap suppression threshold. |

### Return Value
* **`list[int]`**: Kept box indices sorted by descending confidence score.
