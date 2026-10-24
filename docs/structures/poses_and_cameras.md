# Poses and Cameras

The `spatialhub.structures` module provides standardized dataclasses for camera calibration parameters and 3D rigid transformations in $\mathrm{SE}(3)$. All spatial vectors and coordinate axes follow the OpenCV camera frame ($+X$ right, $+Y$ down, $+Z$ forward along the optical axis).

## `CameraIntrinsics`

Represents camera intrinsic calibration matrices mapping 3D camera-frame coordinates to 2D homogeneous pixel coordinates.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `matrix` | `np.ndarray` | `(N, 3, 3)` | `float32` | Intrinsic pinhole camera calibration matrices $K$. |

### Behavior

- Accepts unbatched `(3, 3)` or batched `(N, 3, 3)` arrays. Unbatched matrices are automatically expanded to leading batch size $N=1$.
- Slicing along the leading dimension returns a new `CameraIntrinsics` instance. Integer indexing preserves the leading batch dimension `(1, 3, 3)`.

### Example

```python
import numpy as np
from spatialhub.structures import CameraIntrinsics

k_matrix = np.array([
    [525.0,   0.0, 320.0],
    [  0.0, 525.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float32)

intrinsics = CameraIntrinsics(matrix=k_matrix)
print("Standardized batch shape:", intrinsics.matrix.shape)  # (1, 3, 3)
```

---

## `RigidTransform3D`

Represents homogeneous rigid transformation matrices in $\mathrm{SE}(3)$ mapping 3D coordinates between reference frames:

$$
T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathrm{SE}(3), \quad R \in \mathrm{SO}(3), \quad \mathbf{t} \in \mathbb{R}^3
$$

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `matrix` | `np.ndarray` | `(N, 4, 4)` | `float32` | Homogeneous transformation matrices in $\mathrm{SE}(3)$. |

### Behavior

- Accepts `(4, 4)` or `(N, 4, 4)` arrays. Unbatched matrices are automatically expanded to `(1, 4, 4)`.
- Slicing along the leading dimension returns a new `RigidTransform3D` instance.

### Example

```python
import numpy as np
from spatialhub.structures import RigidTransform3D

t_matrix = np.eye(4, dtype=np.float32)
t_matrix[:3, 3] = [0.1, -0.2, 1.5]

transform = RigidTransform3D(matrix=t_matrix)
print("Batch size:", len(transform))
print("Translation vector:", transform.matrix[0, :3, 3])
```

---

## `CameraExtrinsics`

Subclass of `RigidTransform3D` representing camera extrinsic poses in $\mathrm{SE}(3)$. Defines coordinate transformations mapping between the camera space and an external world reference frame.

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `matrix` | `np.ndarray` | `(N, 4, 4)` | `float32` | Homogeneous camera-to-world or world-to-camera matrices. |

### Example

```python
import numpy as np
from spatialhub.structures import CameraExtrinsics

pose_matrix = np.eye(4, dtype=np.float32)
extrinsics = CameraExtrinsics(matrix=pose_matrix)

first_view = extrinsics[0]
print("Extrinsics batch shape:", first_view.matrix.shape)  # (1, 4, 4)
```

---

## `ObjectPose6D`

Represents estimated 6D rigid object poses in $\mathrm{SE}(3)$ mapping coordinates from an object CAD frame into the camera frame:

$$
\begin{bmatrix} \mathbf{P}_{\text{cam}} \\ 1 \end{bmatrix} = T_{\text{cam} \leftarrow \text{obj}} \begin{bmatrix} \mathbf{P}_{\text{obj}} \\ 1 \end{bmatrix}
$$

### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `pose` | `np.ndarray` | `(N, 4, 4)` | `float32` | Rigid transformation matrices in $\mathrm{SE}(3)$. |
| `scores` | `np.ndarray | None` | `(N,)` | `float32` | Optional pose estimation confidence scores. |
| `ids` | `np.ndarray | None` | `(N,)` | `int64` | Optional object or class identifier indices. |
| `names` | `list[str] | None` | length $N$ | `str` | Optional CAD model or category names. |

### Behavior

- Standardizes unbatched `(4, 4)` matrices to `(1, 4, 4)`.
- Validates that non-null `scores`, `ids`, and `names` match the pose batch size $N$.
- Slicing along the instance dimension synchronizes `pose`, `scores`, `ids`, and `names`.

### Example

```python
import numpy as np
from spatialhub.structures import ObjectPose6D

poses = ObjectPose6D(
    pose=np.eye(4, dtype=np.float32)[np.newaxis, ...],
    scores=np.array([0.96], dtype=np.float32),
    ids=np.array([1], dtype=np.int64),
    names=["drill"],
)

print("Pose count:", len(poses))
print("Detected object name:", poses.names[0])
```

---

## Related Utilities

- [`invert_transform`](../utilities/transform.md#invert_transform): Computes the analytical inverse of an $\mathrm{SE}(3)$ matrix.
- [`transform_points`](../utilities/transform.md#transform_points): Transforms 3D point arrays using an $\mathrm{SE}(3)$ matrix.
- [`opencv_to_opengl`](../utilities/transform.md#opencv_to_opengl): Converts transformation matrices between OpenCV and OpenGL coordinate frames.
- [`scale_intrinsics`](../utilities/camera.md#scale_intrinsics): Adjusts focal lengths and optical centers for resized views.
- [`reproject_depth_to_3d`](../utilities/camera.md#reproject_depth_to_3d): Unprojects 2D depth fields into 3D Cartesian point clouds.
- [`project_points_to_image`](../utilities/camera.md#project_points_to_image): Projects 3D points onto the camera image plane.
- [`draw_3d_axis`](../utilities/visualization.md#draw_3d_axis): Renders 3D coordinate axes representing object poses onto images.
