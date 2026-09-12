# 3D Mesh Processing

`spatialhub.utils.mesh` provides functions for reading, scaling, centering, and analyzing 3D CAD meshes (`.ply`, `.obj`, `.stl`, `.off`) using `trimesh`, as well as spherical viewpoint pose generation utilities.

```python
from spatialhub.utils import (
    load_mesh,
    read_mesh,
    scale_mesh,
    center_mesh,
    compute_mesh_diameter,
    compute_obb,
    sample_sphere_poses,
    look_at,
    invert_transform,
)
```

> [!NOTE]
> Mesh utilities require optional rendering dependencies.
> Install via `uv sync --extra render` or `pip install "spatialhub[render]"`.

---

## Modular vs. Composite Pipeline

```
read_mesh  ->  scale_mesh  ->  center_mesh
          \________________________/
                 load_mesh
```

---

## `read_mesh`

Parses file paths, `Path`, `trimesh.Trimesh`, or `trimesh.Scene` instances into a single concatenated `trimesh.Trimesh`.

```python
mesh = read_mesh("models/obj_000001.ply")
```

### Parameters
* **`mesh_input`** (`str | Path | trimesh.Trimesh | trimesh.Scene`): Input CAD geometry source.

### Return Value
* **`trimesh.Trimesh`**: Concatenated single mesh.

---

## `scale_mesh`

Scales mesh vertices from CAD source units into metric meters:

$$V_{\text{meters}} = V_{\text{raw}} \cdot s$$

```python
mesh_m = scale_mesh(mesh_raw, model_unit="mm")
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | required | Source mesh. |
| `model_unit` | `str \| float` | `"m"` | Unit: `"m"` ($1.0$), `"cm"` ($0.01$), `"mm"` ($0.001$), or numeric scale factor. |

### Return Value
* **`trimesh.Trimesh`**: Mesh with vertices in meters.

---

## `center_mesh`

Translates the mesh so that its bounding box centroid moves to $(0, 0, 0)$:

$$V_{\text{centered}} = V_{\text{meters}} - c_{\text{bbox}}$$

```python
mesh_centered, offset = center_mesh(mesh_m)
```

### Parameters
* **`mesh`** (`trimesh.Trimesh`): Input mesh in meters.

### Return Value
* **`tuple[trimesh.Trimesh, np.ndarray]`**: `(centered_mesh, translation_offset_applied)`.

---

## `load_mesh`

Composite pipeline combining `read_mesh`, `scale_mesh`, and `center_mesh` in a single call.

```python
mesh = load_mesh("models/obj_000001.ply", model_unit="mm", center=True)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mesh_input` | `str \| Path \| trimesh.Trimesh \| trimesh.Scene` | required | Source geometry. |
| `model_unit` | `str \| float` | `"m"` | Source coordinate unit system. |
| `center` | `bool` | `True` | Whether to translate bounding box centroid to origin. |

### Return Value
* **`trimesh.Trimesh`**: Metric, optionally centered mesh.

---

## `compute_mesh_diameter`

Calculates the maximum 3D Euclidean distance across the mesh surface by sampling convex hull vertices.

```python
diameter = compute_mesh_diameter(mesh)  # e.g. 0.142 meters
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | required | Input mesh in meters. |
| `n_sample` | `int` | `10000` | Number of convex hull sample points. |

### Return Value
* **`float`**: Mesh diameter in meters.

---

## `compute_obb`

Computes oriented bounding box (OBB) corners, dimensions, and transformation matrix.

```python
corners, extents, to_origin = compute_obb(mesh)
```

### Parameters
* **`mesh`** (`trimesh.Trimesh`): Input mesh in meters.

### Return Value
* **`tuple[np.ndarray, np.ndarray, np.ndarray]`**: `(corners_in_mesh, extents, to_origin)`.

---

## `sample_sphere_poses`

Generates evenly distributed camera or object poses around a sphere using either Fibonacci spiral or subdivided icosphere distribution.

```python
poses = sample_sphere_poses(
    num_viewpoints=42,
    radius=0.4,
    sampling_method="fibonacci",
    pose_type="object_pose",
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_viewpoints` | `int` | `42` | Target number of viewpoints. |
| `radius` | `float` | `1.0` | Distance from model origin in meters. |
| `sampling_method` | `str` | `"fibonacci"` | Sampling distribution: `"fibonacci"` or `"icosphere"`. |
| `subdivisions` | `int \| None` | `None` | Optional fixed icosphere subdivision level. |
| `pose_type` | `str` | `"object_pose"` | `"object_pose"` (w2c) or `"camera_pose"` (c2w). |

### Return Value
* **`np.ndarray`**: Array of shape `(N, 4, 4)` containing 4x4 transformation matrices in float32.

---

## `look_at`

Computes a 4x4 camera-to-world transformation matrix pointing from `cam_location` towards `target_point` in OpenCV camera coordinates (+X right, +Y down, +Z forward).

```python
c2w = look_at(cam_location=np.array([0.0, 0.0, 1.0]), target_point=np.array([0.0, 0.0, 0.0]))
```

---

## `invert_transform`

Computes the analytical inverse of 4x4 rigid homogeneous transformation matrices (single `(4, 4)` or batched `(B, 4, 4)`).

```python
w2c = invert_transform(c2w)
```

---

## `to_single_mesh`

Helper function that flattens multi-geometry `trimesh.Scene` structures into a single `trimesh.Trimesh`.

---

## `prepare_mesh_arrays`

Extracts contiguous vertex, normal, index, UV, and texture buffer arrays from a CAD mesh into a `MeshArrays` data container.

```python
mesh_arrays = prepare_mesh_arrays(mesh, max_tex_size=2048, flip_uv=True)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | required | Input CAD mesh. |
| `max_tex_size` | `int \| None` | `None` | Optional maximum texture dimension limit in pixels. |
| `flip_uv` | `bool` | `True` | Invert vertical UV coordinates ($1 - V$) for OpenGL shader convention. |

### Return Value
* **`MeshArrays`**: Container dataclass holding contiguous NumPy arrays (`pos`, `faces`, `vnormals`, `tex`, `uv`, `vertex_color`).
