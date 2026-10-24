# Images

The image utilities provide deterministic routines for image disk I/O, color space transformations, depth map rescaling, image standardization, bounding box derivation, symmetric aspect-ratio padding, and non-maximum suppression (NMS).

## Functions

### `load_image`

Reads an image file from disk and converts it to the requested color ordering.

```python
def load_image(
    path: str | Path,
    color_mode: Literal["RGB", "RGBA", "GRAY", "BGR", "BGRA"] | None = "RGB",
) -> np.ndarray
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `path` | `str | Path` | File system path to the image file. |
| `color_mode` | `str | None` | Target color mode (`"RGB"`, `"RGBA"`, `"GRAY"`, `"BGR"`, `"BGRA"`, or `None` for uncompressed raw). Default: `"RGB"`. |

#### Outputs

- Returns `np.ndarray` of shape `(H, W, C)` or `(H, W)`, dtype `uint8`.

#### Exceptions

- `FileNotFoundError`: If the file cannot be opened.
- `ValueError`: If the color mode or channel count is unsupported.

#### Example

```python
from spatialhub.utils import load_image

image = load_image("scene.png", color_mode="RGB")
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
```

---

### `save_image`

Writes an image array to disk, creating parent directories when necessary.

```python
def save_image(
    image: np.ndarray,
    path: str | Path,
    color_mode: Literal["RGB", "RGBA", "GRAY", "BGR", "BGRA"] | None = "RGB",
) -> Path
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W)`, `(H, W, 1)`, `(H, W, 3)`, or `(H, W, 4)` | `uint8` or `bool` | Image array to serialize. |
| `path` | `str | Path` | - | Destination file path. |
| `color_mode` | `str | None` | - | Color order of the input array. Converted automatically to disk formats (`BGR`/`BGRA`/`GRAY`). Default: `"RGB"`. |

#### Outputs

- Returns `Path` pointing to the written file.

#### Example

```python
import numpy as np
from spatialhub.utils import save_image

rgb_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
out_path = save_image(rgb_canvas, "outputs/frame.png", color_mode="RGB")
print("Saved to:", out_path)
```

---

### `save_video`

Writes a sequence of frames or a batched 4D array to an encoded video file on disk.

```python
def save_video(
    frames: np.ndarray | Sequence[np.ndarray],
    path: str | Path,
    fps: float | int = 30,
    color_mode: Literal["RGB", "BGR"] = "RGB",
    fourcc: str | None = None,
) -> Path
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `frames` | `(T, H, W, 3)` or `Sequence[(H, W, 3)]` | `uint8` | Sequence of video frames or batched 4D array. |
| `path` | `str | Path` | - | Destination video file path (`.mp4`, `.avi`). |
| `fps` | `float | int` | - | Playback frame rate in frames per second (default: `30`). |
| `color_mode` | `str` | - | Color channel order of input frames (`"RGB"` or `"BGR"`). Default: `"RGB"`. |
| `fourcc` | `str | None` | - | Optional 4-character codec code (e.g. `'mp4v'`, `'avc1'`). Defaults to `'mp4v'` for `.mp4`. |

#### Outputs

- Returns `Path` pointing to the written video file.

#### Example

```python
import numpy as np
from spatialhub.utils import save_video

# write 60 animated frames to MP4
frames = [np.full((480, 640, 3), i * 4, dtype=np.uint8) for i in range(60)]
out_video = save_video(frames, "outputs/tracking.mp4", fps=30)
print("Video saved to:", out_video)
```

---

### `to_color`

Converts an image array between color spaces (`RGB`, `RGBA`, `BGR`, `BGRA`, `GRAY`). Supports unbatched 2D and 3D arrays as well as batched arrays with arbitrary leading dimensions `(..., H, W)` or `(..., H, W, C)`.

```python
def to_color(
    image: np.ndarray,
    dst: str = "RGB",
    src: str | None = None,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W)`, `(..., H, W)`, `(H, W, C)`, or `(..., H, W, C)` | `uint8` or `float` | Input image array. |
| `dst` | `str` | `str` | Target color mode (`"RGB"`, `"RGBA"`, `"BGR"`, `"BGRA"`, `"GRAY"`). Default: `"RGB"`. |
| `src` | `str | None` | `str | None` | Source color mode. When `None`, inferred from channel count (1 channel or 2D: `GRAY`, 3 channels: `RGB`, 4 channels: `RGBA`). |

#### Outputs

- Returns converted `np.ndarray` of shape `(..., H, W)` for `GRAY` or `(..., H, W, C_out)` for multichannel formats.

#### Exceptions

- `TypeError`: If `image` is not a `np.ndarray`.
- `ValueError`: If target or source color modes are unsupported or image dimensions are invalid.

#### Example

```python
import numpy as np
from spatialhub.utils import to_color

# Convert BGR image to RGB
bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)
rgb_image = to_color(bgr_image, dst="RGB", src="BGR")

print("Output shape:", rgb_image.shape)
print("Output dtype:", rgb_image.dtype)
```

---

### `scale_depth`

Scales depth values from arbitrary units (such as millimeters or centimeters) to metric meters.

```python
def scale_depth(
    depth: np.ndarray,
    depth_unit: Literal["m", "cm", "mm"] | float | int = "m",
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `depth` | `(H, W)` or `(..., H, W)` | `np.floating` or `np.integer` | Depth array. |
| `depth_unit` | `"m" | "cm" | "mm" | float | int` | - | Unit scale or multiplier (`"m"`: 1.0, `"cm"`: 0.01, `"mm"`: 0.001, or numeric factor to meters). Default: `"m"`. |

#### Outputs

- Returns `np.ndarray` of shape `(H, W)` or `(..., H, W)`, dtype `float32`, in metric meters.

#### Example

```python
import numpy as np
from spatialhub.utils import scale_depth

# convert raw 16-bit uint16 depth in millimeters to metric meters
raw_depth_mm = np.full((480, 640), 1500, dtype=np.uint16)
depth_meters = scale_depth(raw_depth_mm, depth_unit="mm")

print("Depth in meters:", depth_meters[240, 320])
```

---

### `normalize_image`

Standardizes an image using per-channel mean and standard deviation values.

```python
def normalize_image(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    to_chw: bool = True,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W, 3)` or `(..., H, W, 3)` | `uint8` or `float32` | RGB image array. |
| `mean` | `(3,)` or `(1, 1, 3)` | `float32` | Channel-wise means. Default: ImageNet mean `[0.485, 0.456, 0.406]`. |
| `std` | `(3,)` or `(1, 1, 3)` | `float32` | Channel-wise standard deviations. Default: ImageNet std `[0.229, 0.224, 0.225]`. |
| `to_chw` | `bool` | - | If `True`, transposes trailing dimensions from `(..., H, W, C)` to `(..., C, H, W)`. Default: `True`. |

#### Outputs

- Returns standardized `np.ndarray` of shape `(..., 3, H, W)` (if `to_chw=True`) or `(..., H, W, 3)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import normalize_image

rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
tensor = normalize_image(rgb, to_chw=True)

print("Normalized tensor shape:", tensor.shape)
print("Normalized tensor dtype:", tensor.dtype)
```

---

### `mask_to_bbox`

Extracts the bounding box enclosing non-zero pixels from a binary mask.

```python
def mask_to_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `mask` | `(H, W)` or `(H, W, 1)` | `bool`, `uint8`, or `integer` | Binary segmentation mask. |

#### Outputs

- Returns a bounding box tuple `(x_min, y_min, x_max, y_max)` in pixel coordinates. If the mask is empty, returns `(0, 0, W, H)`.

#### Example

```python
import numpy as np
from spatialhub.utils import mask_to_bbox

mask = np.zeros((480, 640), dtype=bool)
mask[100:250, 150:350] = True

x1, y1, x2, y2 = mask_to_bbox(mask)
print(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
```

---

### `pad_to_square`

Symmetrically pads an image along its shorter dimension to create a square canvas.

```python
def pad_to_square(
    image: np.ndarray,
    mode: str = "constant",
    pad_value: float | int = 0,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W)`, `(..., H, W)`, `(H, W, C)`, or `(..., H, W, C)` | any numeric | Input image array. |
| `mode` | `str` | - | NumPy padding mode. Default: `"constant"`. |
| `pad_value` | `float | int` | - | Value used to fill padded margins. Default: `0`. |

#### Outputs

- Returns square `np.ndarray` of shape `(..., max(H, W), max(H, W), C)` or `(..., max(H, W), max(H, W))`.

#### Example

```python
import numpy as np
from spatialhub.utils import pad_to_square

img = np.zeros((480, 640, 3), dtype=np.uint8)
padded = pad_to_square(img, pad_value=0)

print("Padded shape:", padded.shape)
```

---

### `nms_boxes`

Performs Non-Maximum Suppression (NMS) over 2D bounding boxes based on Intersection-over-Union (IoU) overlap.

```python
def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `boxes` | `(N, 4)` | `float32` | Bounding box coordinates in `[x1, y1, x2, y2]` format. |
| `scores` | `(N,)` or `(N, 1)` | `float32` | Confidence scores. |
| `iou_threshold` | `float` | - | IoU threshold in `[0.0, 1.0]`. Default: `0.5`. |

#### Outputs

- Returns a `list[int]` containing the indices of surviving bounding boxes, ordered by descending confidence.

#### Example

```python
import numpy as np
from spatialhub.utils import nms_boxes

boxes = np.array([
    [10.0, 10.0, 50.0, 50.0],
    [12.0, 11.0, 49.0, 52.0],
    [100.0, 100.0, 160.0, 160.0],
], dtype=np.float32)

scores = np.array([0.90, 0.75, 0.85], dtype=np.float32)

surviving_indices = nms_boxes(boxes, scores, iou_threshold=0.5)
print("Surviving box indices:", surviving_indices)
```
