# Meshes

The `spatialhub.utils.mesh` module provides utilities for loading CAD meshes, centering geometry, scaling units to meters, calculating 3D bounding diameters, computing minimum oriented bounding boxes (OBB), and preparing contiguous GPU-ready vertex and texture buffers.

!!! NOTE
    Mesh operations require the optional `render` extra: `uv sync --extra render` or `pip install 'spatialhub[render]'`.

## Data Containers

### `MeshBuffers`

Contiguous C-ordered vertex, face, normal, UV, and texture buffers formatted for GPU upload.

```python
@dataclass
class MeshBuffers:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    texture: np.ndarray | None = None
    uvs: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None
```

| Field | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `vertices` | `(V, 3)` | `float32` | 3D vertex coordinates. |
| `faces` | `(F, 3)` | `int32` | Triangular element indices. |
| `normals` | `(V, 3)` | `float32` | Vertex normal unit vectors. |
| `texture` | `(1, H, W, 3)` or `None` | `float32` | Normalized RGB texture map in `[0.0, 1.0]`. |
| `uvs` | `(V, 2)` or `None` | `float32` | Normalized UV texture coordinates. |
| `vertex_colors` | `(V, 3)` or `None` | `float32` | Normalized RGB vertex colors in `[0.0, 1.0]`. |

---

### `OrientedBoundingBox`

Minimum-volume oriented bounding box (OBB) representation.

```python
@dataclass
class OrientedBoundingBox:
    corners: np.ndarray
    extents: np.ndarray
    to_origin: np.ndarray | None = None
```

| Field | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `corners` | `(8, 3)` | `float32` | 8 box corner vertices in the mesh coordinate frame. |
| `extents` | `(3,)` | `float32` | Box side dimensions `(dx, dy, dz)` along principal axes. |
| `to_origin` | `(4, 4)` or `None` | `float32` | Transformation matrix mapping mesh coordinates to the origin-centered canonical frame. |

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
) -> OrientedBoundingBox
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry. |

#### Outputs

- Returns [`OrientedBoundingBox`](#orientedboundingbox) containing corner coordinates in the mesh frame, dimension extents, and the transformation matrix to canonical origin space.

#### Example

```python
from spatialhub.utils import load_mesh, compute_obb

mesh = load_mesh("assets/sample_object.obj")
obb = compute_obb(mesh)

print("OBB extents (dx, dy, dz):", obb.extents)
print("OBB corners shape:", obb.corners.shape)
```

---

### `prepare_mesh_buffers`

Extracts GPU-ready, C-contiguous vertex attributes, face indices, UV coordinates, and texture buffers from a CAD mesh.

```python
def prepare_mesh_buffers(
    mesh: trimesh.Trimesh,
    max_tex_size: int | None = None,
    flip_uv: bool = True,
) -> MeshBuffers
```

#### Inputs

| Argument | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Mesh geometry. |
| `max_tex_size` | `int | None` | Optional upper bound on texture map spatial dimension in pixels. Default: `None`. |
| `flip_uv` | `bool` | Inverts vertical UV coordinates ($1.0 - V$) for the OpenGL convention. Default: `True`. |

#### Outputs

- Returns [`MeshBuffers`](#meshbuffers) holding contiguous `float32` and `int32` arrays (`vertices`, `faces`, `normals`, and either `texture`/`uvs` or `vertex_colors`).

#### Example

```python
from spatialhub.utils import load_mesh, prepare_mesh_buffers

mesh = load_mesh("assets/sample_object.obj")
buffers = prepare_mesh_buffers(mesh, max_tex_size=1024)

print("Vertices:", buffers.vertices.shape, buffers.vertices.dtype)
print("Faces:   ", buffers.faces.shape, buffers.faces.dtype)
print("Normals: ", buffers.normals.shape, buffers.normals.dtype)
```
