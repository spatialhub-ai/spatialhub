# Scene and Depth

The `spatialhub.structures` module provides standardized dataclasses for 3D oriented bounding boxes, metric depth maps, and aligned multimodal raster buffers.

## `BoundingBox3D`

Represents 3D oriented bounding box vertices in a 3D coordinate frame.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `corners` | `np.ndarray` | `(N, 8, 3)` | `float32` | Eight 3D corner coordinates per bounding box. |

### Corner Vertex Topology

Corners follow the canonical 8-vertex parallel-face topology defined in [`extents_to_corners`](../utilities/transform.md#extents_to_corners) (vertices `0..3` form the near $-Z$ face; `4..7` form the opposing far $+Z$ face).

### Example

```python
import numpy as np
from spatialhub.structures import BoundingBox3D
from spatialhub.utils import extents_to_corners

# compute 8 corners from 3D bounding extents (dx, dy, dz)
corners = extents_to_corners([0.15, 0.20, 0.30])
bbox = BoundingBox3D(corners=corners)

print("Bounding box corners shape:", bbox.corners.shape)  # (1, 8, 3)
```

---

## `DepthMap`

Represents estimated dense depth fields in camera space.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `depth` | `np.ndarray` | `(N, H, W)` | `float32` | Metric depth field array in meters. |
| `confidence` | `np.ndarray | None` | `(N, H, W)` | `float32` | Optional per-pixel confidence or uncertainty map. |
| `depth_type` | `Literal["metric", "relative", "inverse", "disparity"]` | scalar | `str` | Scale representation (default: `"metric"`). |
| `intrinsics` | `CameraIntrinsics | np.ndarray | None` | `(N, 3, 3)` | `float32` | Optional camera intrinsic calibration matrices. |
| `extrinsics` | `CameraExtrinsics | np.ndarray | None` | `(N, 4, 4)` | `float32` | Optional camera extrinsic transformation matrices in $\mathrm{SE}(3)$. |

### Usage and Slicing

When slicing along the leading dimension, depth fields, confidence maps, and attached camera matrices are sliced synchronously:

```python
import numpy as np
from spatialhub.structures import DepthMap

depth_map = DepthMap(
    depth=np.ones((2, 480, 640), dtype=np.float32) * 1.5,
    depth_type="metric",
)

first_frame = depth_map[0]
print("Standardized batch shape:", first_frame.depth.shape)  # (1, 480, 640)
```

---

## `ImageBuffers`

Represents multimodal, pixel-aligned 2D raster buffers on the image plane. Used primarily as the output structure for offscreen ModernGL render passes.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `rgba` | `np.ndarray | None` | `(N, H, W, 4)` | `uint8` | Color images with alpha channel. |
| `depth` | `np.ndarray | None` | `(N, H, W)` | `float32` | Metric depth maps. |
| `normal` | `np.ndarray | None` | `(N, H, W, 3)` | `float32` | Surface normal vector fields in $[-1, 1]$. |
| `mask` | `np.ndarray | None` | `(N, H, W)` | `uint8` or `bool` | Binary foreground masks. |

`__post_init__` enforces that all non-null buffers share identical batch size $N$ and spatial dimensions $(H, W)$.

### Example

```python
import numpy as np
from spatialhub.structures import ImageBuffers

buffers = ImageBuffers(
    rgba=np.zeros((1, 256, 256, 4), dtype=np.uint8),
    depth=np.zeros((1, 256, 256), dtype=np.float32),
)
print("RGBA buffer shape:", buffers.rgba.shape)  # (1, 256, 256, 4)
```

---

## Related Utilities

- [`extents_to_corners`](../utilities/transform.md#extents_to_corners): Converts 3D bounding extents into 8-corner canonical vertices.
- [`scale_depth`](../utilities/image.md#scale_depth): Scales metric depth maps or converts units to meters.
- [`unproject_depth_map`](../utilities/camera.md#unproject_depth_map): Unprojects 2D depth fields into 3D Cartesian point clouds.
- [`draw_3d_box`](../utilities/visualization.md#draw_3d_box): Projects and draws 3D bounding box edges onto 2D images.
- [`visualize_depth`](../utilities/visualization.md#visualize_depth): Normalizes float depth maps and applies colormaps for inspection.
- [`compute_obb`](../utilities/mesh.md#compute_obb): Computes minimum-volume oriented bounding boxes from 3D CAD meshes.
