# Cameras

The `spatialhub.utils.camera` module implements pinhole camera operations, depth unprojection to 3D Cartesian coordinates, 3D point projection to 2D image coordinates, OpenGL projection matrix generation, and spherical camera viewpoint sampling.

## Functions

### `scale_intrinsics`

Scales camera calibration intrinsic parameters according to source and target image resolutions.

```python
def scale_intrinsics(
    K: np.ndarray,
    orig_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `K` | `(3, 3)` or `(..., 3, 3)` | `float32` | Pinhole camera intrinsic calibration matrix $K$. |
| `orig_size` | `tuple[int, int]` | `int` | Original image dimensions as `(width, height)`. |
| `new_size` | `tuple[int, int]` | `int` | Target image dimensions as `(width, height)`. |

#### Outputs

- Returns scaled intrinsic matrix of shape `(..., 3, 3)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import scale_intrinsics

k_orig = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# scale resolution from 640x480 down to 320x240
k_scaled = scale_intrinsics(k_orig, orig_size=(640, 480), new_size=(320, 240))
print("Scaled principal point:", k_scaled[0, 2], k_scaled[1, 2])
```

---

### `unproject_points`

Unprojects discrete 2D pixel coordinates and corresponding depth values into 3D Cartesian coordinates in the camera reference frame.

```python
def unproject_points(
    points_2d: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray
```

#### Mathematical Formulation

For each 2D coordinate $(u, v)$ with depth $Z$:

$$X = \frac{u - c_x}{f_x} \cdot Z, \quad Y = \frac{v - c_y}{f_y} \cdot Z, \quad Z = Z$$

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `points_2d` | `(N, 2)` or `(..., N, 2)` | `float32` | 2D pixel coordinates $(u, v)$. |
| `depth` | `(N,)` or `(..., N)` | `float32` | Measured depth values along the optical axis in meters. |
| `intrinsics` | `(3, 3)` or `(..., 3, 3)` | `float32` | Pinhole camera calibration matrix $K$. |

#### Outputs

- Returns 3D point coordinate array of shape `(N, 3)` or `(..., N, 3)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import unproject_points

points_2d = np.array([[320.0, 240.0], [420.0, 290.0]], dtype=np.float32)
depth = np.array([2.0, 2.5], dtype=np.float32)
intrinsics = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

points_3d = unproject_points(points_2d, depth, intrinsics)
print("3D points:", points_3d)
```

---

### `unproject_depth_map`

Unprojects a 2D depth map into a 3D Cartesian point cloud in the camera reference frame.

```python
def unproject_depth_map(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray
```

#### Mathematical Formulation

For each pixel coordinate $(u, v)$ with measured depth $Z > 0$:

$$X = \frac{u - c_x}{f_x} \cdot Z, \quad Y = \frac{v - c_y}{f_y} \cdot Z, \quad Z = Z$$

Pixels with $Z \le 0$ or non-finite values are assigned `0.0`.

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `depth` | `(H, W)` or `(N, H, W)` | `float32` | Metric depth map in meters. |
| `intrinsics` | `(3, 3)` or `(N, 3, 3)` | `float32` | Pinhole camera calibration matrix $K$. |

#### Outputs

- Returns 3D point coordinates array of shape `(H, W, 3)` or `(N, H, W, 3)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import unproject_depth_map

depth = np.ones((480, 640), dtype=np.float32) * 1.5
intrinsics = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

points_3d = unproject_depth_map(depth, intrinsics)
print("Point cloud shape:", points_3d.shape)
print("Center optical depth Z:", points_3d[240, 320, 2])
```

---

### `project_points_to_image`

Projects 3D points onto the camera image plane, optionally transforming them from an object frame first.

```python
def project_points_to_image(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `points_3d` | `(M, 3)` or `(N, M, 3)` | `float32` | 3D Cartesian coordinates. |
| `intrinsics` | `(3, 3)` or `(N, 3, 3)` | `float32` | Pinhole camera calibration matrix $K$. |
| `pose` | `(4, 4)` or `(N, 4, 4)` | `float32` | Optional rigid transformation $T_{c\leftarrow o}$ mapping points to camera space. |

#### Outputs

- Returns tuple `(points_2d, valid_mask)`:
  - `points_2d`: Projected pixel coordinates `(x, y)` of shape `(M, 2)` or `(N, M, 2)`, dtype `float32`.
  - `valid_mask`: Boolean array of shape `(M,)` or `(N, M)` indicating whether the point lies strictly in front of the camera ($Z > 0$).

#### Example

```python
import numpy as np
from spatialhub.utils import project_points_to_image

points_obj = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
k = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

pose = np.eye(4, dtype=np.float32)
pose[:3, 3] = [0.0, 0.0, 1.0]

points_2d, valid = project_points_to_image(points_obj, k, pose=pose)
print("Projected pixel coords:\n", points_2d)
print("Valid depth mask:", valid)
```

---

### `intrinsics_to_projection_matrix`

Converts a pinhole intrinsic calibration matrix into an OpenGL standard projection matrix.

```python
def intrinsics_to_projection_matrix(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `intrinsics` | `(3, 3)` or `(..., 3, 3)` | `float32` | OpenCV camera intrinsic matrix. |
| `width` | `int` | - | Image viewport width in pixels. |
| `height` | `int` | - | Image viewport height in pixels. |
| `znear` | `float` | - | Near clipping plane in meters. Default: `0.01`. |
| `zfar` | `float` | - | Far clipping plane in meters. Default: `100.0`. |

#### Outputs

- Returns $4 \times 4$ OpenGL projection matrix of shape `(..., 4, 4)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import intrinsics_to_projection_matrix

k = np.array([
    [500.0, 0.0, 320.0],
    [0.0, 500.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

proj_mat = intrinsics_to_projection_matrix(k, width=640, height=480, znear=0.01, zfar=10.0)
print("Projection matrix shape:", proj_mat.shape)
```

---

### `sample_sphere_points`

Samples quasi-uniform points on a 2-sphere using a spherical Fibonacci lattice.

```python
def sample_sphere_points(
    num_points: int = 100,
    radius: float = 1.0,
) -> np.ndarray
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `num_points` | `int` | Number of sample vertices. Default: `100`. |
| `radius` | `float` | Sphere radius in meters. Default: `1.0`. |

#### Outputs

- Returns Cartesian point coordinates of shape `(num_points, 3)`, dtype `float32`.

#### Example

```python
from spatialhub.utils import sample_sphere_points

pts = sample_sphere_points(num_points=128, radius=1.5)
print("Sphere points shape:", pts.shape)
```

---

### `sample_sphere_poses`

Generates camera-to-world extrinsic poses on a sphere looking toward a central target.

```python
def sample_sphere_poses(
    num_views: int = 100,
    radius: float = 1.0,
    look_at: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    up_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Description |
| :--- | :--- | :--- |
| `num_views` | `int` | Number of camera views. Default: `100`. |
| `radius` | `float` | Orbit distance from the camera center to `look_at`. Default: `1.0`. |
| `look_at` | `(3,)` or 3-tuple | Target focus point coordinates. Default: `(0, 0, 0)`. |
| `up_vector` | `(3,)` or 3-tuple | World reference up direction. Default: `(0, 0, 1)`. |

#### Outputs

- Returns camera-to-world rigid transformation matrices of shape `(num_views, 4, 4)`, dtype `float32`.

#### Example

```python
from spatialhub.utils import sample_sphere_poses

poses = sample_sphere_poses(num_views=64, radius=0.8)
print("Extrinsic poses shape:", poses.shape)
```

---

### `sample_icosphere_poses`

Generates camera viewpoints by recursively subdividing an icosahedron surface.

```python
def sample_icosphere_poses(
    subdivisions: int = 2,
    radius: float = 1.0,
    look_at: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    up_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> np.ndarray
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `subdivisions` | `int` | Number of recursive triangle subdivisions ($\ge 0$). Resulting view count is $V = 10 \cdot 4^{\text{subdivisions}} + 2$. Default: `2` (162 views). |
| `radius` | `float` | Orbit distance in meters. Default: `1.0`. |
| `look_at` | `(3,)` or 3-tuple | Target focus point coordinates. Default: `(0, 0, 0)`. |
| `up_vector` | `(3,)` or 3-tuple | World reference up direction. Default: `(0, 0, 1)`. |

#### Outputs

- Returns camera-to-world matrices of shape `(V, 4, 4)`, dtype `float32`.

#### Example

```python
from spatialhub.utils import sample_icosphere_poses

# subdivisions=1 generates 42 views; subdivisions=2 generates 162 views
poses = sample_icosphere_poses(subdivisions=1, radius=1.0)
print("Generated view count:", len(poses))
```
