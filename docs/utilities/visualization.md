# Visualization

The `spatialhub.utils.visualization` module provides lightweight rendering and annotation routines for debugging geometric predictions, including projected 3D wireframe boxes, 3D coordinate axes, instance masks, depth maps, keypoint correspondences, and PCA feature maps.

## Functions

### `draw_3d_box`

Projects 3D bounding box corners into 2D camera coordinates and draws wireframe edges on an image.

```python
def draw_3d_box(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    bbox_corners_3d: np.ndarray,
    color: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W, 3)` | `uint8` | Source RGB image canvas. |
| `pose` | `(4, 4)` or `(N, 4, 4)` | `float32` | Rigid transformation matrix $T_{c\leftarrow o}$ mapping object to camera space. |
| `intrinsics` | `(3, 3)` | `float32` | Pinhole camera intrinsic calibration matrix $K$. |
| `bbox_corners_3d` | `(8, 3)` or `(N, 8, 3)` | `float32` | 8 canonical bounding box corner vertices in the object coordinate frame. |
| `color` | `(3,)` tuple or list of tuples | `int` | RGB color tuple or list of colors per box. Default: `(0, 255, 0)`. |
| `thickness` | `int` | - | Stroke width in pixels. Default: `2`. |

#### Outputs

- Returns annotated RGB image array of shape `(H, W, 3)`, dtype `uint8`. Boxes behind the optical center ($Z \le 0$) are skipped.

#### Example

```python
import numpy as np
from spatialhub.utils import draw_3d_box, extents_to_corners

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
pose = np.eye(4, dtype=np.float32)
pose[:3, 3] = [0.0, 0.0, 1.2]

intrinsics = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

corners = extents_to_corners([0.15, 0.20, 0.10])
annotated = draw_3d_box(canvas, pose, intrinsics, corners, color=(0, 255, 0))
```

---

### `draw_3d_axis`

Draws 3D orthogonal coordinate axes at the target object origin ($X$: Red, $Y$: Green, $Z$: Blue).

```python
def draw_3d_axis(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    axis_length: float = 0.05,
    thickness: int = 2,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W, 3)` | `uint8` | Source RGB image canvas. |
| `pose` | `(4, 4)` or `(N, 4, 4)` | `float32` | Rigid object-to-camera transformation matrix $T_{c\leftarrow o}$. |
| `intrinsics` | `(3, 3)` | `float32` | Pinhole camera intrinsic calibration matrix $K$. |
| `axis_length` | `float` | - | Axis vector length in meters. Default: `0.05`. |
| `thickness` | `int` | - | Stroke width in pixels. Default: `2`. |

#### Outputs

- Returns annotated RGB image array of shape `(H, W, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import draw_3d_axis

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
pose = np.eye(4, dtype=np.float32)
pose[:3, 3] = [0.0, 0.0, 1.0]

intrinsics = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

annotated = draw_3d_axis(canvas, pose, intrinsics, axis_length=0.1)
```

---

### `blend_masks`

Overlays colored binary instance segmentation masks onto an image canvas with alpha transparency.

```python
def blend_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.4,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    draw_border: bool = False,
    border_thickness: int = 2,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W, 3)` | `uint8` | Source RGB image. |
| `masks` | `(H, W)` or `(N, H, W)` | `bool` or `uint8` | Binary instance masks. |
| `alpha` | `float` | - | Overlay transparency factor in `[0.0, 1.0]`. Default: `0.4`. |
| `colors` | `(3,)` tuple or list of tuples | `int` | RGB color for masks. If `None`, distinct palette colors are auto-generated. |
| `draw_border` | `bool` | - | If `True`, draws solid boundary contours around masks. Default: `False`. |
| `border_thickness` | `int` | - | Contour stroke width in pixels. Default: `2`. |

#### Outputs

- Returns blended RGB image array of shape `(H, W, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import blend_masks

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=bool)
mask[100:300, 200:400] = True

blended = blend_masks(canvas, mask, alpha=0.5, colors=(255, 0, 0))
```

---

### `draw_boxes_2d`

Draws 2D bounding boxes and optional confidence or class labels onto an image canvas.

```python
def draw_boxes_2d(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray | None = None,
    names: list[str] | None = None,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image` | `(H, W, 3)` | `uint8` | Source RGB image. |
| `boxes` | `(4,)` or `(N, 4)` | `float32` | Bounding box coordinates in `[x1, y1, x2, y2]`. |
| `scores` | `(N,)` or `None` | `float32` | Optional confidence scores. |
| `names` | `list[str]` or `None` | `str` | Optional class label names. |
| `colors` | `(3,)` tuple or list of tuples | `int` | Optional box colors. |
| `thickness` | `int` | - | Stroke width in pixels. Default: `2`. |

#### Outputs

- Returns annotated RGB image array of shape `(H, W, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import draw_boxes_2d

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
boxes = np.array([[50.0, 50.0, 200.0, 250.0]], dtype=np.float32)
scores = np.array([0.95], dtype=np.float32)

annotated = draw_boxes_2d(canvas, boxes, scores=scores, names=["mug"])
```

---

### `visualize_depth`

Colormaps a 2D depth map into a 3-channel RGB image canvas.

```python
def visualize_depth(
    depth: np.ndarray,
    colormap: str | int = "inferno",
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `depth` | `(H, W)` or `(1, H, W)` | `float32` | Metric depth map in meters. |
| `colormap` | `str | int` | - | Colormap name (`"inferno"`, `"turbo"`, `"plasma"`, `"viridis"`, `"magma"`, `"bone"`) or OpenCV integer ID. Default: `"inferno"`. |
| `min_depth` | `float | None` | - | Optional lower depth bound. If `None`, uses 2nd percentile of valid pixels. |
| `max_depth` | `float | None` | - | Optional upper depth bound. If `None`, uses 98th percentile of valid pixels. |

#### Outputs

- Returns colormapped RGB image array of shape `(H, W, 3)`, dtype `uint8`. Invalid pixels ($Z \le 0$ or non-finite) are rendered black.

#### Example

```python
import numpy as np
from spatialhub.utils import visualize_depth

depth = np.random.uniform(0.5, 3.0, (480, 640)).astype(np.float32)
depth_rgb = visualize_depth(depth, colormap="turbo")

print("Visualized depth shape:", depth_rgb.shape)
print("Visualized depth dtype:", depth_rgb.dtype)
```

---

### `visualize_matched_keypoints`

Renders side-by-side keypoint correspondences between two camera views with connecting lines.

```python
def visualize_matched_keypoints(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.5,
    max_side: int = 800,
    top_k: int | None = None,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `image_a` | `(H_a, W_a, 3)` | `uint8` | First RGB image view. |
| `image_b` | `(H_b, W_b, 3)` | `uint8` | Second RGB image view. |
| `keypoints_a` | `(N, 2)` | `float32` | Matched keypoints in view A in `(x, y)` coordinates. |
| `keypoints_b` | `(N, 2)` | `float32` | Matched keypoints in view B in `(x, y)` coordinates. |
| `confidence` | `(N,)` or `None` | `float32` | Optional match confidence scores in `[0.0, 1.0]`. |
| `min_confidence` | `float` | - | Minimum score threshold for display. Default: `0.5`. |
| `max_side` | `int` | - | Canvas scale limit in pixels. Default: `800`. |
| `top_k` | `int | None` | - | Optional limit on the number of highest-confidence matches. |

#### Outputs

- Returns side-by-side composite RGB image of shape `(H, W_total, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import visualize_matched_keypoints

img_a = np.zeros((480, 640, 3), dtype=np.uint8)
img_b = np.zeros((480, 640, 3), dtype=np.uint8)

kpts_a = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
kpts_b = np.array([[110.0, 145.0], [210.0, 240.0]], dtype=np.float32)
conf = np.array([0.92, 0.88], dtype=np.float32)

matched_vis = visualize_matched_keypoints(img_a, img_b, kpts_a, kpts_b, confidence=conf)
print("Side-by-side canvas shape:", matched_vis.shape)
```

---

### `visualize_dense_features`

Projects high-dimensional dense feature maps onto their top 3 principal components (PCA) and scales them into an RGB image.

```python
def visualize_dense_features(
    features: np.ndarray,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `features` | `(H, W, D)` or `(1, H, W, D)` | `float32` | Dense feature tensor with embedding dimension $D$. |

#### Outputs

- Returns PCA-projected RGB image of shape `(H, W, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import visualize_dense_features

features = np.random.randn(60, 80, 384).astype(np.float32)
feat_rgb = visualize_dense_features(features)

print("PCA feature map shape:", feat_rgb.shape)
print("PCA feature map dtype:", feat_rgb.dtype)
```

---

### `visualize_global_features`

Computes and visualizes the pairwise cosine similarity matrix across global feature vectors.

```python
def visualize_global_features(
    features: np.ndarray,
    colormap: str | int = "viridis",
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `features` | `(N, D)` | `float32` | Array of $N$ feature vectors of dimension $D$. |
| `colormap` | `str | int` | - | Colormap name or OpenCV colormap integer ID. Default: `"viridis"`. |

#### Outputs

- Returns pairwise cosine similarity matrix canvas of shape `(N, N, 3)`, dtype `uint8`.

#### Example

```python
import numpy as np
from spatialhub.utils import visualize_global_features

vectors = np.random.randn(16, 768).astype(np.float32)
sim_canvas = visualize_global_features(vectors, colormap="viridis")

print("Similarity canvas shape:", sim_canvas.shape)
```
