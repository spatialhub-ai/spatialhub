# Utilities API Reference

The `spatialhub.utils` module provides unified image loading, ImageNet normalization, foreground cropping, non-maximum suppression (NMS), 3D template rendering, and drawing engines.

---

## 1. Image Preprocessing

### `load_image(image_input, color_mode="RGB")`
Reads an image from disk or accepts an in-memory NumPy array, converting channels to the requested format.

```python
from spatialhub.utils import load_image

img_rgb = load_image("image.jpg", color_mode="RGB")    # (H, W, 3)
img_gray = load_image("image.jpg", color_mode="GRAY")  # (H, W)
```

- **`image_input`** (`str | Path | np.ndarray`): File path string, Path object, or NumPy array.
- **`color_mode`** (`"RGB" | "RGBA" | "GRAY"`): Target color format.

### `normalize_image(image, mean=IMAGENET_MEAN, std=IMAGENET_STD, to_chw=True)`
Applies channel-wise ImageNet normalization and optionally transposes array layout from `(H, W, C)` to `(C, H, W)`.

```python
from spatialhub.utils import normalize_image

normed_chw = normalize_image(img_float32, to_chw=True)  # (3, H, W)
```

### `extract_foreground_bbox(image_rgba)`
Extracts the tight foreground bounding box `[x1, y1, x2, y2]` from an RGBA image based on non-zero alpha channel pixels.

### `square_crop_and_resize(image, bbox, target_size=224)`
Crops an image to a bounding box, applies equal square padding around the cropped region, and resizes the square canvas to `target_size x target_size`.

---

## 2. Postprocessing & Filtering

### `non_max_suppression(boxes, scores, iou_threshold=0.7)`
Performs vectorized Non-Maximum Suppression (NMS) over candidate bounding boxes based on IoU overlap ratios.

```python
from spatialhub.utils import non_max_suppression

keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.7)
filtered_boxes = boxes[keep_indices]
```

- **`boxes`** (`np.ndarray` of shape `[N, 4]`): Bounding box coordinates `[x1, y1, x2, y2]`.
- **`scores`** (`np.ndarray` of shape `[N]`): Confidence scores.
- **`iou_threshold`** (`float`): Maximum allowable IoU overlap threshold before suppression.

---

## 3. 3D CAD Template Renderer (`TemplateRenderer`)

The `TemplateRenderer` class renders 2D RGBA template views and metric depth maps from 3D CAD meshes (`.ply`, `.obj`, `.stl`, `.off`) using `trimesh` and offscreen `pyrender` (EGL/OpenGL).

```python
from spatialhub.utils import TemplateRenderer

renderer = TemplateRenderer(
    model_path="cad_models/obj_000001.ply",
    model_unit="mm",
    ambient_light=(1.0, 1.0, 1.0, 1.0),
    light_color=(1.0, 1.0, 1.0),
    light_intensity=1.0,
    bg_color=(0.0, 0.0, 0.0, 0.0),
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str \| Path \| trimesh.Trimesh` | Required | Target 3D CAD mesh file path or pre-loaded `trimesh.Trimesh` instance. |
| `model_unit` | `str \| float` | `"m"` | Unit of vertex coordinates in the raw CAD mesh file (`"m"`, `"cm"`, `"mm"`, or a numeric scale factor). |
| `ambient_light` | `tuple[float, float, float, float] \| None` | `(1.0, 1.0, 1.0, 1.0)` | Ambient scene lighting intensity in RGBA float format. |
| `light_color` | `tuple[float, float, float]` | `(1.0, 1.0, 1.0)` | Spot light color in RGB float format. |
| `light_intensity` | `float` | `1.0` | Spot light intensity multiplier. |
| `bg_color` | `tuple[float, float, float, float]` | `(0.0, 0.0, 0.0, 0.0)` | Background clear color (transparent by default). |

---

### Technical Unit Scaling & Transformations

Internal rendering calculations in OpenGL/Pyrender require all spatial coordinates (vertices, translations, and depth maps) to be represented in **meters ($m$)**. `TemplateRenderer` normalizes mesh vertex coordinates and camera pose translation vectors to a shared metric space:

#### 1. `model_unit` Scaling Behavior (`load_mesh`)
The raw vertex array $V \in \mathbb{R}^{V_{count} \times 3}$ of the CAD mesh is scaled by a scale factor $s_{model}$:

$$
s_{model} = \begin{cases} 
1.0 & \text{if } \text{model\_unit} = \text{"m"} \\
0.01 & \text{if } \text{model\_unit} = \text{"cm"} \\
0.001 & \text{if } \text{model\_unit} = \text{"mm"} \\
\text{numeric\_value} & \text{if } \text{isinstance(model\_unit, float)}
\end{cases}
$$

$$
V_{meters} = V_{raw} \cdot s_{model}
$$

After scaling to meters, the mesh bounding box centroid $c_{bbox} \in \mathbb{R}^3$ is translated to origin $(0, 0, 0)$:

$$
V_{centered} = V_{meters} - c_{bbox}
$$

#### 2. `pose_unit` & `radius` Scaling Behavior (`_get_view_attribute`)
Loaded 4x4 transformation matrices $T_i \in \mathbb{R}^{4 \times 4}$ have their 3D translation column $t = T_i[:3, 3]$ converted to meters and scaled by $r$:

$$
s_{pose} = \begin{cases} 
1.0 & \text{if } \text{pose\_unit} = \text{"m"} \\
0.01 & \text{if } \text{pose\_unit} = \text{"cm"} \\
0.001 & \text{if } \text{pose\_unit} = \text{"mm"} \\
\text{numeric\_value} & \text{if } \text{isinstance(pose\_unit, float)}
\end{cases}
$$

$$
t_{scaled} = t_{raw} \cdot s_{pose} \cdot \text{radius}
$$

#### 3. Coordinate System Axis Conversion ($T_{gl\_cv}$)
Pyrender operates in OpenGL camera coordinates ($+X$ right, $+Y$ up, $-Z$ forward). Standard vision models (OpenCV / COLMAP) operate in $+Z$ forward coordinates ($+X$ right, $+Y$ down, $+Z$ forward). `TemplateRenderer` applies $T_{gl\_cv}$ to bridge coordinate conventions:

$$
T_{gl\_cv} = \begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 
\end{bmatrix}
$$

---

### Rendering Method: `render_templates()`

```python
templates = renderer.render_templates(
    width=640,
    height=480,
    intrinsics=[572.41, 573.57, 325.26, 242.04],
    poses=poses_array,           # (N, 4, 4) NumPy array or path to .npy file
    pose_unit="mm",
    pose_type="object_pose",
    num_viewpoints=42,
    radius=0.4,
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` | `int` | Required | Output rendered image width in pixels. |
| `height` | `int` | Required | Output rendered image height in pixels. |
| `intrinsics` | `np.ndarray \| list[float]` | Required | Camera intrinsic matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` or 4-element sequence `[fx, fy, cx, cy]`. |
| `poses` | `np.ndarray \| str \| Path \| None` | `None` | Array of shape `(N, 4, 4)` containing 4x4 transformation matrices. If `None`, generates $N$ viewpoints across a Fibonacci sphere. |
| `pose_unit` | `str \| float` | `"mm"` | Unit of translation vector in `poses` (`"m"`, `"cm"`, `"mm"`). |
| `pose_type` | `"object_pose" \| "camera_pose" \| None` | `None` | Pose reference frame interpretation: <br>• `"object_pose"`: Camera is fixed at origin; object moves via $T$.<br>• `"camera_pose"`: Object is fixed at origin $(0,0,0)$; camera moves via $T \cdot T_{gl\_cv}$. |
| `num_viewpoints` | `int` | `42` | Number of viewpoints to generate on Fibonacci sphere if `poses=None`. |
| `radius` | `float` | `0.4` | Spherical viewpoint radius distance from object centroid in meters. |

#### Return Value
List of $N$ dictionaries containing:
- `"rgba"` (`np.ndarray` of shape `(H, W, 4)` uint8): 32-bit RGBA rendered image.
- `"depth"` (`np.ndarray` of shape `(H, W)` float32): Metric depth map in meters ($m$).

---

### File Output Helper: `save()`

```python
rgba_paths, depth_paths = renderer.save(
    results=templates,
    output_dir="output_templates",
    save_scene=True,
    save_depth=True,
)
```

- **`save_scene`** (`bool`): Writes 32-bit RGBA `.png` images (`000000_rgba.png`).
- **`save_depth`** (`bool`): Writes float32 metric depth `.npy` arrays (`000000_depth.npy`).

---

## 4. Visualization Utilities

### `visualize_matches(img_a, img_b, mkpts0, mkpts1, mconf, conf_thresh=0.5, max_side=800, top_k=None, save_path=None)`
Renders side-by-side keypoint correspondences connected by confidence-colored lines.

### `visualize_masks(image, boxes, masks, scores, save_path=None)`
Renders colorized segment mask overlays with bounding box outlines on top of the original image.
