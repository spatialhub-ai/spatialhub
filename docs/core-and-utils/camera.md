# Camera Geometry

`spatialhub.utils.camera` provides functions for camera intrinsic scaling, 2D-to-3D depth reprojection, OpenGL projection matrices, and OpenCV/OpenGL coordinate space conversions.

```python
from spatialhub.utils import (
    scale_camera_intrinsics,
    reproject_depth_to_3d,
    reproject_depth_to_3d_batch,
    create_perspective_projection_matrix,
    convert_opencv_to_opengl_pose,
)
```

---

## Coordinate Space Conventions

| Convention | +X Axis | +Y Axis | +Z Axis |
| :--- | :--- | :--- | :--- |
| **OpenCV** | Right | Down | Forward (Optical Axis into Scene) |
| **OpenGL** | Right | Up | Backward (Camera views along -Z) |

Bridging transformation:

$$
T_{gl \leftarrow cv} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

---

## `scale_camera_intrinsics`

Scales camera focal lengths $(f_x, f_y)$ and principal points $(c_x, c_y)$ proportionally when image dimensions change.

```python
K_scaled = scale_camera_intrinsics(K, orig_size=(640, 480), new_size=(320, 240))
```

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `K` | `np.ndarray (3, 3)` | Camera intrinsic matrix. |
| `orig_size` | `tuple[int, int]` | Original `(width, height)`. |
| `new_size` | `tuple[int, int]` | New `(width, height)`. |

### Return Value
* **`np.ndarray`**: Scaled float32 intrinsic matrix of shape `(3, 3)`.

---

## `reproject_depth_to_3d`

Reprojects a 2D depth map (in meters) into a 3D camera-space XYZ coordinate map:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = \text{depth}[v, u]$$

```python
xyz_map = reproject_depth_to_3d(depth, K)  # (H, W, 3)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `depth` | `np.ndarray (H, W)` | required | Depth map in meters. |
| `K` | `np.ndarray (3, 3)` | required | Camera intrinsic matrix. |
| `uvs` | `np.ndarray (N, 2) \| None` | `None` | Optional pixel coordinates. If `None`, evaluates all pixels. |
| `z_min` | `float` | `0.001` | Minimum valid depth in meters. |
| `z_max` | `float` | `inf` | Maximum valid depth in meters. |

### Return Value
* **`np.ndarray`**: Float32 XYZ coordinate map array `(H, W, 3)`.

---

## `reproject_depth_to_3d_batch`

Batched vectorized version of `reproject_depth_to_3d` operating on `(B, H, W)` depth arrays and `(B, 3, 3)` camera matrices.

```python
xyz_maps = reproject_depth_to_3d_batch(depths, Ks)  # (B, H, W, 3)
```

### Parameters
* **`depths`** (`np.ndarray` of shape `(B, H, W)`): Batched depth maps in meters.
* **`Ks`** (`np.ndarray` of shape `(B, 3, 3)`): Batched camera intrinsic matrices.

### Return Value
* **`np.ndarray`**: Float32 XYZ maps of shape `(B, H, W, 3)`.

---

## `create_perspective_projection_matrix`

Converts a 3x3 OpenCV camera intrinsic matrix $K$ into a 4x4 OpenGL clip-space perspective projection matrix.

```python
proj = create_perspective_projection_matrix(K, height=480, width=640, znear=0.001, zfar=100.0)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `K` | `np.ndarray (3, 3)` | required | OpenCV camera intrinsic matrix. |
| `height` | `int` | required | Frame height in pixels. |
| `width` | `int` | required | Frame width in pixels. |
| `znear` | `float` | `0.001` | Near clipping plane distance in meters. |
| `zfar` | `float` | `100.0` | Far clipping plane distance in meters. |
| `window_coords` | `str` | `"y_down"` | `"y_down"` (OpenCV top-left) or `"y_up"` (OpenGL bottom-left). |

### Return Value
* **`np.ndarray`**: 4x4 OpenGL projection matrix as float32.

---

## `convert_opencv_to_opengl_pose`

Transforms 4x4 rigid camera poses between OpenCV (+Z forward) and OpenGL (-Z forward) camera coordinate spaces using $T_{gl \leftarrow cv}$.

```python
pose_gl = convert_opencv_to_opengl_pose(pose_cv)          # (4, 4)
poses_gl = convert_opencv_to_opengl_pose(poses_cv_batch)  # (B, 4, 4)
```

### Parameters
* **`pose_cv`** (`np.ndarray`): Rigid pose matrix of shape `(4, 4)` or `(B, 4, 4)`.

### Return Value
* **`np.ndarray`**: Transformed pose matrix in OpenGL coordinates.
