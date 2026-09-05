# 3D Mesh Processing

`spatialhub.utils.mesh` provides functions for reading, scaling, centering, and analyzing 3D CAD meshes (`.ply`, `.obj`, `.stl`, `.off`) using `trimesh`.

```python
from spatialhub.utils import load_mesh, read_mesh, scale_mesh, center_mesh, compute_mesh_diameter
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

## `to_single_mesh`

Helper function that flattens multi-geometry `trimesh.Scene` structures into a single `trimesh.Trimesh`.
