# 3D Template Renderer

`spatialhub.utils.renderer.TemplateRenderer` renders 2D RGBA templates and metric depth maps from 3D CAD meshes (`.ply`, `.obj`, `.stl`, `.off`) using `trimesh` and offscreen `pyrender` (OpenGL/EGL).

> [!NOTE]
> Rendering features require optional dependencies.
> Install via `uv sync --extra render` or `pip install "spatialhub[render]"`.

```python
from spatialhub.utils import TemplateRenderer
```

---

## Constructor

```python
TemplateRenderer(
    model_path: str | Path | trimesh.Trimesh,
    model_unit: str | float = "m",
    ambient_light: tuple[float, float, float, float] | None = (1.0, 1.0, 1.0, 1.0),
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    light_intensity: float = 1.0,
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str \| Path \| trimesh.Trimesh` | required | Target CAD mesh file path or pre-loaded `trimesh.Trimesh`. |
| `model_unit` | `str \| float` | `"m"` | Coordinate unit of raw CAD vertices (`"m"`, `"cm"`, `"mm"`, or float multiplier). |
| `ambient_light` | `tuple[float, float, float, float] \| None` | `(1.0, 1.0, 1.0, 1.0)` | Ambient scene lighting RGBA. |
| `light_color` | `tuple[float, float, float]` | `(1.0, 1.0, 1.0)` | Spot light RGB color. |
| `light_intensity` | `float` | `1.0` | Spot light intensity multiplier. |
| `bg_color` | `tuple[float, float, float, float]` | `(0.0, 0.0, 0.0, 0.0)` | Background clear color. |

---

## Unit Scaling & Coordinate Handling

### `model_unit` Scaling & Centering
Raw vertices $V_{\text{raw}}$ are scaled to metric meters using scale factor $s_{\text{model}}$:

$$V_{\text{meters}} = V_{\text{raw}} \cdot s_{\text{model}}, \quad V_{\text{centered}} = V_{\text{meters}} - c_{\text{bbox}}$$

### `pose_unit` & `radius` Translation Scaling
Loaded transformation matrices $T_i$ have translation vectors $t = T_i[:3, 3]$ scaled:

$$t_{\text{scaled}} = t_{\text{raw}} \cdot s_{\text{pose}} \cdot \text{radius}$$

### Coordinate System Axis Conversion ($T_{gl \leftarrow cv}$)
Applies OpenGL $\leftrightarrow$ OpenCV axis conversion automatically:

$$
T_{gl \leftarrow cv} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

---

## `render_templates`

Renders templates across specified or Fibonacci-generated viewpoints.

```python
templates = renderer.render_templates(
    width=640,
    height=480,
    intrinsics=[572.41, 573.57, 325.26, 242.04],
    num_viewpoints=42,
    radius=0.4,
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` | `int` | required | Rendered output width in pixels. |
| `height` | `int` | required | Rendered output height in pixels. |
| `intrinsics` | `np.ndarray \| list[float]` | required | 3x3 matrix or 4-element `[fx, fy, cx, cy]`. |
| `poses` | `np.ndarray \| str \| Path \| None` | `None` | `(N, 4, 4)` array or path to `.npy`. If `None`, generates $N$ viewpoints on a Fibonacci sphere. |
| `pose_unit` | `str \| float` | `"mm"` | Unit of translation vector in `poses`. |
| `pose_type` | `"object_pose" \| "camera_pose" \| None` | `None` | Reference frame interpretation. Defaults to `"object_pose"`. |
| `num_viewpoints` | `int` | `42` | Number of viewpoints on Fibonacci sphere when `poses=None`. |
| `radius` | `float` | `0.4` | Viewpoint distance from model centroid in meters. |

### Return Value
* **`list[dict[str, np.ndarray]]`**: List of dictionaries with `"rgba"` (`(H, W, 4)` uint8) and `"depth"` (`(H, W)` float32 meters).

---

## `save`

Saves rendered RGBA images (`000000_rgba.png`) and float32 metric depth arrays (`000000_depth.npy`) to disk.

```python
rgba_paths, depth_paths = renderer.save(templates, output_dir="output", save_depth=True)
```

### Parameters
* **`results`** (`list[dict]`): Output from `render_templates()`.
* **`output_dir`** (`str | Path`): Destination directory.
* **`save_scene`** (`bool`): Save RGBA images (default: `True`).
* **`save_depth`** (`bool`): Save depth maps (default: `False`).

### Return Value
* **`tuple[list[Path], list[Path]]`**: Saved file paths `(rgba_paths, depth_paths)`.
