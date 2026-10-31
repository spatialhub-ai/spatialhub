# Quickstart

This guide steps through loading an input image, estimating depth, scaling camera parameters, aligning predictions to metric scale, unprojecting to a 3D point cloud, and extracting instance segmentation masks.

The full standalone script is available in the repository at [`examples/quickstart.py`](https://github.com/spatialhub-ai/spatialhub/blob/main/examples/quickstart.py).

---

## Setup

Clone the repository and install project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/spatialhub-ai/spatialhub.git
cd spatialhub
uv sync
```

For manual virtual environment management or GPU extras, see [Installation](installation.md).

---

## Depth Estimation and Reprojection

### Load Data

Load the input RGB frame and pinhole calibration matrix $K$. Ground-truth depth maps stored as 16-bit integer millimeters are normalized into metric meters ($d_{\text{meters}} = d_{\text{mm}} / 1000.0$) using [`scale_depth`](../utilities/image.md#scale_depth):

```python title="examples/quickstart.py" linenums="36"
--8<-- "examples/quickstart.py:load_assets"
```

---

### Estimate Depth

Monocular depth estimation is executed with [`DepthAnything3`](../models/depthanything3.md). The forward pass produces a typed [`DepthMap`](../structures/scene_and_bounds.md#depthmap) container:

```python title="examples/quickstart.py" linenums="46"
--8<-- "examples/quickstart.py:depth_inference"
```

Output:

```text
Predicted depth shape:  (504, 504)
Ground-truth shape:     (1024, 1024)
Depth scale mode:       metric
```

The adapter processes images with a Vision Transformer backbone using patch size $P = 14 \times 14$ and target resolution `process_res = 504` ($504 / 14 = 36$ tokens). The raw output is returned at the model's native processing resolution $(504, 504)$ without applying internal upsampling.

---

### Scale Intrinsics

The calibration matrix $K$ is defined in pixel coordinates for the original $1024 \times 1024$ sensor. Evaluating $K$ directly against a $(504, 504)$ array would place the principal point $(512.0, 512.0)$ outside the frame boundaries. [`scale_intrinsics`](../utilities/camera.md#scale_intrinsics) scales focal lengths and principal point coordinates proportionally ($s = 504 / 1024 \approx 0.4922$):

```python title="examples/quickstart.py" linenums="59"
--8<-- "examples/quickstart.py:scale_intrinsics"
```

This maps the optical center to $(252.0, 252.0)$, aligning pinhole geometry with the native $(504, 504)$ prediction array. When unprojecting directly at model resolution, `pred_intrinsics` is used; when unprojecting after resizing back to $(1024, 1024)$, the original $K$ is used.

---

### Align Scale and Evaluate Error

In synthetic indoor CAD models, windows look out onto an environment background map without interior geometry, producing $0.0\,\text{m}$ ground-truth depth. The validity mask `gt_valid = (gt_depth > 0) & np.isfinite(gt_depth)` excludes these void regions.
 
Although `da3metric_large` outputs metric depth estimates, zero-shot predictions on uncalibrated indoor camera setups can exhibit an affine scale offset. The prediction is registered to metric ground truth via median ratio scaling:

$$s = \frac{\operatorname{median}(d_{\text{gt}})}{\operatorname{median}(\hat{d}_{\text{pred}})}, \quad d_{\text{metric}} = s \cdot \hat{d}_{\text{pred}}$$

```python title="examples/quickstart.py" linenums="70"
--8<-- "examples/quickstart.py:metric_alignment"
```

Output:

```text
Estimated scale factor: 0.8588
Mean absolute error:    0.0975 m
```

<div class="panel-viewer-card">
  <div class="panel-viewer-toolbar">
    <div class="panel-btn-group-vertical">
      <button id="pbtn_rgb" class="panel-btn" onclick="switchPanel('rgb')">RGB Frame</button>
      <button id="pbtn_gt" class="panel-btn" onclick="switchPanel('gt')">Ground Truth</button>
      <button id="pbtn_pred" class="panel-btn active" onclick="switchPanel('pred')">Predicted Depth</button>
      <button id="pbtn_error" class="panel-btn" onclick="switchPanel('error')">Absolute Error</button>
    </div>
  </div>
  <div class="panel-viewer-body">
    <img id="panelViewerImg" src="../assets/panel_depth_pred.jpg" alt="Predicted Metric Depth Map" loading="lazy" decoding="async" />
  </div>
</div>

<script>
  function switchPanel(mode) {
    const panels = {
      rgb: '../assets/panel_rgb.jpg',
      gt: '../assets/panel_depth_gt.jpg',
      pred: '../assets/panel_depth_pred.jpg',
      error: '../assets/panel_depth_error.jpg'
    };
    const img = document.getElementById('panelViewerImg');
    if (img && panels[mode]) {
      img.src = panels[mode];
    }
    ['rgb', 'gt', 'pred', 'error'].forEach(function(k) {
      const btn = document.getElementById('pbtn_' + k);
      if (btn) btn.classList.toggle('active', k === mode);
    });
  }
</script>

Depth maps are colormapped from $0.0\,\text{m}$ (dark purple) to $4.5\,\text{m}$ (bright yellow). The absolute error heatmap spans $0.0\,\text{m}$ (blue, zero residual) through $0.25\,\text{m}$ (green) to $\ge 0.5\,\text{m}$ (red). Zero-depth window pixels in the ground truth are masked out during error calculation.

---

### Reproject to 3D

Depth fields are unprojected into 3D camera space via [`unproject_depth_map`](../utilities/camera.md#unproject_depth_map) using camera intrinsics $K$:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$

Non-zero 3D coordinates are paired with RGB pixel values and exported to standard `.ply` format:

```python title="examples/quickstart.py" linenums="96"
--8<-- "examples/quickstart.py:point_cloud_reprojection"
```

Output:

```text
Extracted point cloud vertices: 1048576
Point cloud saved to: examples/output/room_point_cloud_pred.ply
```

<iframe src="../assets/point_cloud_viewer.html" width="100%" height="520px" style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px; margin: 1rem 0;" title="Interactive 3D Point Cloud Viewer"></iframe>

---

## Instance Segmentation

### Segment and Slice Masks

Class-agnostic instance mask proposal is executed using [`FastSAM`](../models/fastsam.md). Adapters return candidates sorted by descending confidence score. Slicing along the leading batch dimension (`instances[:12]`) extracts the top 12 highest-confidence detections to isolate major scene structures while preserving typed metadata.

Masks and bounding boxes are composited onto the source image using [`blend_masks`](../utilities/visualization.md#blend_masks) and [`draw_boxes_2d`](../utilities/visualization.md#draw_boxes_2d):

```python title="examples/quickstart.py" linenums="113"
--8<-- "examples/quickstart.py:instance_segmentation"
```

Output:

```text
Total detected instances: 50
Top instance masks shape: (12, 1024, 1024)
Top instance boxes shape: (12, 4)
Top instance scores:      [0.9118806  0.8701272  0.86241454 0.85187453 0.8433275  0.8344767
 0.82899237 0.8113675  0.7925509  0.7879998  0.7801984  0.7556777 ]
Segmentation overlay saved to: examples/output/segmentation.jpg
```

<div class="panel-viewer-card">
  <div class="panel-viewer-toolbar">
    <div class="panel-btn-group-vertical">
      <button id="sbtn_rgb" class="panel-btn" onclick="switchSeg('rgb')">RGB Frame</button>
      <button id="sbtn_seg" class="panel-btn active" onclick="switchSeg('seg')">Predicted Instances</button>
    </div>
  </div>
  <div class="panel-viewer-body">
    <img id="segViewerImg" src="../assets/segmentation.jpg" alt="Instance Segmentation Overlay" loading="lazy" decoding="async" />
  </div>
</div>

<script>
  function switchSeg(mode) {
    const panels = {
      rgb: '../assets/panel_rgb.jpg',
      seg: '../assets/segmentation.jpg'
    };
    const img = document.getElementById('segViewerImg');
    if (img && panels[mode]) {
      img.src = panels[mode];
    }
    ['rgb', 'seg'].forEach(function(k) {
      const btn = document.getElementById('sbtn_' + k);
      if (btn) btn.classList.toggle('active', k === mode);
    });
  }
</script>

---

## Execution

The full pipeline is available in [`examples/quickstart.py`](https://github.com/spatialhub-ai/spatialhub/blob/main/examples/quickstart.py). Run it from the repository root:

```bash
uv run examples/quickstart.py
```

Generated evaluation artifacts and 3D geometry are written to `examples/output/`:

- `depth_comparison.jpg`: 4-panel canvas displaying the input RGB frame, ground truth, predicted metric depth, and absolute spatial error map.
- `room_point_cloud_pred.ply`: Colormapped 3D point cloud ($1,048,576$ vertices) in standard PLY format (compatible with MeshLab, CloudCompare, or the WebGL viewer above).
- `segmentation.jpg`: Multi-instance segmentation mask overlay with 2D bounding boxes and confidence scores.

To export the individual labeled panels with calibrated colorbars into `examples/output/panels/`:

```bash
uv run examples/quickstart.py --save-panels
```

---

## Next Steps

- Explore [Poses & Cameras](../structures/poses_and_cameras.md) for standardized perception containers, camera calibrations, and SE(3) poses.
- Review [Model Adapters](../models/overview.md) for supported vision models and output contract specifications.
- Explore [Rendering](../rendering/headless.md) for headless offscreen rasterization and CAD template generation.
