# Transforms

The `spatialhub.utils.transform` module implements closed-form operations on rigid 3D spatial transformations in SE(3), affine point transformations, bounding box vertex construction, and camera coordinate frame conversions.

## Functions

### `invert_transform`

Computes the analytical inverse of a $4 \times 4$ rigid homogeneous transformation matrix in SE(3).

```python
def invert_transform(
    transform: np.ndarray,
) -> np.ndarray
```

#### Mathematical Formulation

For a rigid transformation matrix:

$$T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$

the inverse is computed exploiting the orthogonality of $R \in \text{SO}(3)$ ($R^{-1} = R^T$):

$$T^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `transform` | `(4, 4)` or `(..., 4, 4)` | `float32` | Rigid transformation matrix mapping from reference frame A to B. |

#### Outputs

- Returns inverted matrix of identical shape `(..., 4, 4)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import invert_transform

# define transformation matrix T_b_from_a
t_b_a = np.eye(4, dtype=np.float32)
t_b_a[:3, 3] = [0.1, 0.2, 0.5]

# compute inverse transformation T_a_from_b
t_a_b = invert_transform(t_b_a)

identity_check = t_b_a @ t_a_b
print("Transformation inverse computed correctly:", np.allclose(identity_check, np.eye(4)))
```

---

### `transform_points`

Applies a $4 \times 4$ rigid transformation matrix to a set of 3D point coordinates.

```python
def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `points` | `(N, 3)` or `(..., N, 3)` | `float32` | 3D Cartesian coordinates in the source frame. |
| `transform` | `(4, 4)` or `(..., 4, 4)` | `float32` | Homogeneous transformation matrix mapping source to target frame. |

#### Outputs

- Returns transformed point coordinates matching broadcasted shape `(..., N, 3)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import transform_points

points_local = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.2, 0.0],
], dtype=np.float32)

t_world_local = np.eye(4, dtype=np.float32)
t_world_local[:3, 3] = [1.0, 2.0, 3.0]

points_world = transform_points(points_local, t_world_local)
print("Transformed points:\n", points_world)
```

---

### `extents_to_corners`

Generates 8 canonical bounding box corner vertices centered at the coordinate origin `(0, 0, 0)`.

```python
def extents_to_corners(
    extents: np.ndarray | list[float],
) -> np.ndarray
```

#### Corner Vertex Topology

Canonical 8-corner vertices where $x = dx/2$, $y = dy/2$, and $z = dz/2$:

```text
         4 [-x, -y, +z] ------------------- 5 [+x, -y, +z]
             /|                                /|
            / |                               / |
           /  |                              /  |
 0 [-x, -y, -z] ------------------- 1 [+x, -y, -z]
        |     |                           |     |
        | 7 [-x, +y, +z] -----------------|--- 6 [+x, +y, +z]
        |    /                            |    /
        |   /                             |   /
        |  /                              |  /
 3 [-x, +y, -z] ------------------- 2 [+x, +y, -z]
```

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `extents` | `(3,)` or sequence of 3 floats | `float32` | Bounding box dimensions `(dx, dy, dz)`. |

#### Outputs

- Returns `np.ndarray` of shape `(8, 3)`, dtype `float32`.

#### Example

```python
from spatialhub.utils import extents_to_corners

corners = extents_to_corners([0.10, 0.20, 0.30])
print("Box corners shape:", corners.shape)
print("First corner coordinates:", corners[0])
```

---

### `opencv_to_opengl`

Converts a $4 \times 4$ rigid transformation between the OpenCV camera coordinate frame and the OpenGL coordinate frame.

```python
def opencv_to_opengl(
    transform: np.ndarray,
) -> np.ndarray
```

#### Frame Conventions

- **OpenCV frame**: $+X$ right, $+Y$ down, $+Z$ forward along optical axis.
- **OpenGL frame**: $+X$ right, $+Y$ up, $-Z$ forward along optical axis.

The conversion applies the reflection involution matrix $C = \operatorname{diag}(1, -1, -1, 1)$, inverting the directions of the $Y$ and $Z$ axes:

$$T_{\text{OpenGL}} = C \cdot T_{\text{OpenCV}}$$

Because $C = C^{-1}$, the transformation is self-inverting.

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `transform` | `(4, 4)` or `(..., 4, 4)` | `float32` | Homogeneous transformation matrix in OpenCV frame. |

#### Outputs

- Returns converted matrix of identical shape `(..., 4, 4)`, dtype `float32`.

#### Example

```python
import numpy as np
from spatialhub.utils import opencv_to_opengl

pose_opencv = np.eye(4, dtype=np.float32)
pose_opengl = opencv_to_opengl(pose_opencv)

print("OpenGL diagonal elements:", np.diag(pose_opengl))
```
