# Meshes

The `spatialhub.utils.mesh` module provides utilities for loading CAD meshes, centering geometry, scaling units to meters, calculating 3D bounding diameters, computing minimum oriented bounding boxes (OBB), and preparing contiguous GPU-ready vertex and texture buffers.

!!! NOTE
    Mesh operations require the optional `render` extra: `uv sync --extra render` or `pip install 'spatialhub[render]'`.

## Data Containers

### `CADModel`

Contiguous C-ordered vertex, face, normal, UV, and texture buffers formatted for GPU upload.

#### Attributes

| Attribute | Type | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| `vertices` | `np.ndarray` | `(V, 3)` | `float32` | 3D vertex coordinates in meters. |
| `faces` | `np.ndarray` | `(F, 3)` | `int32` | Triangular face element indices. |
| `normals` | `np.ndarray` | `(V, 3)` | `float32` | Vertex normal unit vectors. |
| `texture` | `np.ndarray | None` | `(1, H, W, 3)` | `float32` | Normalized RGB texture map in $[0.0, 1.0]$. |
| `uvs` | `np.ndarray | None` | `(V, 2)` | `float32` | Normalized UV texture coordinates in $[0.0, 1.0]$. |
| `vertex_colors` | `np.ndarray | None` | `(V, 3)` | `float32` | Normalized RGB vertex colors in $[0.0, 1.0]$. |

#### Example

```python
import numpy as np
from spatialhub.utils import CADModel, prepare_cad_model, load_mesh, scale_mesh, center_mesh

# load, scale to meters, and center mesh
mesh = load_mesh("assets/sample_object.obj")
scale_mesh(mesh, model_unit="mm")
mesh, _ = center_mesh(mesh)

# construct CADModel with continuous buffers
cad_model = prepare_cad_model(mesh)

print("Vertices shape:", cad_model.vertices.shape)
print("Faces shape:   ", cad_model.faces.shape)
```

---

## Functions

### `load_mesh`

Loads a 3D CAD mesh file from disk.

```python
def load_mesh(
    path: str | Path,
) -> trimesh.Trimesh
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `path` | `str | Path` | Path to 3D asset (`.obj`, `.ply`, `.stl`, `.glb`, etc.). |

#### Outputs

- Returns loaded `trimesh.Trimesh` instance.

#### Example

```python
from spatialhub.utils import load_mesh

mesh = load_mesh("assets/sample_object.obj")
print("Vertex count:", len(mesh.vertices))
print("Face count:", len(mesh.faces))
```

---

### `scale_mesh`

Scales mesh vertices in-place from a designated unit system into metric meters.

```python
def scale_mesh(
    mesh: trimesh.Trimesh,
    model_unit: str | float = "m",
) -> trimesh.Trimesh
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry modified in-place. |
| `model_unit` | `str | float` | Unit identifier (`"m"`: 1.0, `"cm"`: 0.01, `"mm"`: 0.001) or custom numeric scale multiplier. Default: `"m"`. |

#### Outputs

- Returns the scaled `trimesh.Trimesh` instance.

#### Example

```python
from spatialhub.utils import load_mesh, scale_mesh

mesh = load_mesh("assets/sample_object.obj")
# convert mesh modeled in millimeters to meters
scale_mesh(mesh, model_unit="mm")
```

---

### `center_mesh`

Translates a mesh in-place so that its bounding box centroid sits at the origin `(0, 0, 0)`.

```python
def center_mesh(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry modified in-place. |

#### Outputs

- Returns tuple `(centered_mesh, translation_applied)` where `translation_applied` is a `(3,)` array in `float32`.

#### Example

```python
from spatialhub.utils import load_mesh, center_mesh

mesh = load_mesh("assets/sample_object.obj")
mesh, offset = center_mesh(mesh)
print("Centroid offset applied:", offset)
```

---

### `compute_mesh_diameter`

Computes the 3D bounding diameter (maximum pairwise vertex distance) of a mesh.

```python
def compute_mesh_diameter(
    mesh: trimesh.Trimesh,
    num_dirs: int = 256,
) -> float
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry. |
| `num_dirs` | `int` | Number of directional probe vectors sampled on the sphere. Default: `256`. |

#### Outputs

- Returns bounding diameter as a `float` in meters.

#### Example

```python
from spatialhub.utils import load_mesh, compute_mesh_diameter

mesh = load_mesh("assets/sample_object.obj")
diameter = compute_mesh_diameter(mesh)
print("Mesh diameter (m):", diameter)
```

---

### `compute_obb`

Computes the minimum-volume oriented bounding box (OBB) for a mesh.

```python
def compute_obb(
    mesh: trimesh.Trimesh,
) -> BoundingBox3D
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry. |

#### Outputs

- Returns [`BoundingBox3D`](../structures/scene_and_bounds.md#boundingbox3d) containing canonical corner coordinates in the mesh frame of shape `(1, 8, 3)`.

#### Example

```python
from spatialhub.utils import load_mesh, compute_obb

mesh = load_mesh("assets/sample_object.obj")
obb = compute_obb(mesh)

print("OBB corners shape:", obb.corners.shape)  # (1, 8, 3)
```

---

### `prepare_cad_model`

Extracts GPU-ready, C-contiguous vertex attributes, face indices, UV coordinates, and texture buffers from a CAD mesh into a `CADModel` container.

```python
def prepare_cad_model(
    mesh: trimesh.Trimesh,
    max_tex_size: int | None = None,
    flip_uv: bool = True,
) -> CADModel
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry. |
| `max_tex_size` | `int | None` | Optional upper bound on texture map spatial dimension in pixels. Default: `None`. |
| `flip_uv` | `bool` | Inverts vertical UV coordinates ($1.0 - V$) for the OpenGL convention. Default: `True`. |

#### Outputs

- Returns [`CADModel`](#cadmodel) holding contiguous `float32` and `int32` arrays (`vertices`, `faces`, `normals`, and either `texture`/`uvs` or `vertex_colors`).

#### Example

```python
from spatialhub.utils import load_mesh, prepare_cad_model

mesh = load_mesh("assets/sample_object.obj")
cad_model = prepare_cad_model(mesh, max_tex_size=1024)

print("Vertices: ", cad_model.vertices.shape, cad_model.vertices.dtype)
print("Faces:    ", cad_model.faces.shape, cad_model.faces.dtype)
print("Normals:  ", cad_model.normals.shape, cad_model.normals.dtype)
```
