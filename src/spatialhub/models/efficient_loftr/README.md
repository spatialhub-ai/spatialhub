# EfficientLoFTR Technical Reference

`spatialhub.models.efficient_loftr` provides an ONNX Runtime adapter for **EfficientLoFTR**, a semi-dense local feature matching model using sparse transformers.

---

## Supported Model Variants

The `EfficientLoFTRAdapter` accepts two model precision variants via `model_type`:

| Model Variant (`model_type`) | ONNX File | Description | Target Use Case |
| :--- | :--- | :--- | :--- |
| `"full"` (Default) | `eloftr_outdoor_full.onnx` | Full precision semi-dense feature matching model. | Maximum matching precision & robust keypoint coverage. |
| `"opt"` | `eloftr_outdoor_opt.onnx` | Optimized/reparameterized lightweight model variant. | High throughput, lower memory footprint, and edge processing. |

---

## Preprocessing & Post-processing Flow

EfficientLoFTR matches coarse-to-fine keypoints across image pairs without requiring PyTorch during inference.

```mermaid
graph TD
    A["Image Pair (A, B): Paths / Arrays"] --> B["Aspect Ratio Downscaling (max_dim)"]
    B --> C["Multiple-of-32 Dimension Alignment"]
    C --> D["Grayscale Conversion & Normalization"]
    D --> E["Bottom-Right Zero Padding to Shared (W_pad, H_pad)"]
    E --> F["ONNX Runtime Forward Pass"]
    F --> G["Filter Matches Outside Unpadded Boundaries"]
    G --> H["Project Coordinates to Native Resolution"]
    H --> I["MatchResult"]
```

### Maximum Dimension Scaling

Given an input image with native dimensions $(W_{\text{orig}}, H_{\text{orig}})$ and optional maximum dimension limit $D_{\max}$ (`max_dim`), dimensions are scaled preserving aspect ratio:

$$
(W_{\text{scaled}}, H_{\text{scaled}}) = \begin{cases} 
\left(\left\lfloor W_{\text{orig}} \cdot \frac{D_{\max}}{\max(W_{\text{orig}}, H_{\text{orig}})} \right\rfloor,\; \left\lfloor H_{\text{orig}} \cdot \frac{D_{\max}}{\max(W_{\text{orig}}, H_{\text{orig}})} \right\rfloor\right) & \text{if } D_{\max} \text{ and } \max(W_{\text{orig}}, H_{\text{orig}}) > D_{\max} \\[8pt]
(W_{\text{orig}}, H_{\text{orig}}) & \text{otherwise}
\end{cases}
$$

### Multiple-of-32 Alignment & Normalization

Spatial dimensions are aligned to the nearest lower multiples of 32 required by sparse transformer downsampling, and pixel values are normalized to $[0, 1]$ float32 tensors of shape $(1, 1, H, W)$:

$$
W = \max\left(32,\ \left\lfloor\frac{W_{\text{scaled}}}{32}\right\rfloor \times 32\right), \qquad H = \max\left(32,\ \left\lfloor\frac{H_{\text{scaled}}}{32}\right\rfloor \times 32\right)
$$

### Bottom-Right Batch Padding

For an image pair $(A, B)$ with individual aligned dimensions $(W_a, H_a)$ and $(W_b, H_b)$, both tensors are bottom-right zero-padded to shared maximum spatial dimensions $(W_{\text{pad}}, H_{\text{pad}})$:

$$
W_{\text{pad}} = \max(W_a, W_b), \qquad H_{\text{pad}} = \max(H_a, H_b)
$$

### Post-processing: Boundary Filtering & Coordinate Projection

Matches $(P_0, P_1)$ detected inside bottom-right zero-padded regions are filtered out using boundary mask $V$:

$$
V = \{ (P_0, P_1) \mid x_0 < W_a \land y_0 < H_a \land x_1 < W_b \land y_1 < H_b \}
$$

Valid raw coordinates $P_{\text{raw}} = (x_{\text{raw}}, y_{\text{raw}})$ are projected back to native image dimensions $(W_{\text{orig}}, H_{\text{orig}})$:

$$
P_{\text{orig}} = \left(x_{\text{raw}} \cdot \frac{W_{\text{orig}}}{W},\; y_{\text{raw}} \cdot \frac{H_{\text{orig}}}{H}\right)
$$

---

## Numerical Parity Verification

Numerical parity verifies mathematical equivalence between the original PyTorch implementation and the exported ONNX Runtime CUDA graph.

In sparse transformer models, keypoint extraction relies on top-k sorting and parallel score reduction. Due to non-associative floating-point operations across parallel GPU threads, PyTorch and ONNX Runtime may produce keypoints in slightly different array order. SpatialHub embeds every matched point pair into a 4D spatial vector $(x_0, y_0, x_1, y_1)$ and resolves correspondences using KD-tree (`scipy.spatial.cKDTree`) nearest-neighbor queries.

Parity was evaluated on 30 image pairs from the [MegaDepth-1500](https://huggingface.co/SpatialHub/efficient-loftr-onnx/tree/main/0015_pairs) outdoor evaluation dataset (`max_dim` denotes the maximum spatial dimension scaling threshold):

<table>
  <thead>
    <tr>
      <th>Variant</th>
      <th>Max Dim</th>
      <th>Pairs</th>
      <th>Mean Matches (PyTorch / ONNX)</th>
      <th>Match Ratio</th>
      <th>Keypoint MAE (px)</th>
      <th>Keypoint Max Diff (px)</th>
      <th>Confidence MAE</th>
      <th>Confidence Max Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>full</code></td>
      <td><code>640</code></td>
      <td>30</td>
      <td>1091.4 / 1091.4</td>
      <td>99.9%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000534</td>
      <td>0.005604</td>
    </tr>
    <tr>
      <td><code>768</code></td>
      <td>30</td>
      <td>1552.0 / 1552.0</td>
      <td>100.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000543</td>
      <td>0.005379</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>30</td>
      <td>1850.8 / 1850.6</td>
      <td>100.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000488</td>
      <td>0.005781</td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>opt</code></td>
      <td><code>640</code></td>
      <td>30</td>
      <td>1150.2 / 1149.9</td>
      <td>100.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.003513</td>
      <td>0.023308</td>
    </tr>
    <tr>
      <td><code>768</code></td>
      <td>30</td>
      <td>1638.7 / 1638.5</td>
      <td>100.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.003573</td>
      <td>0.030832</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>30</td>
      <td>1956.6 / 1956.6</td>
      <td>100.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.003477</td>
      <td>0.034790</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
> Keypoint coordinate error is $0.0000\text{ px}$ MAE and Max Diff across all tested configurations. Minor confidence variance ($\sim 10^{-4}$ in `full`, $\sim 10^{-3}$ in `opt`) stems from floating-point reduction order in the attention layers. Evaluated up to `max_dim=832`.

To verify parity on a custom dataset organized into two folders with matching filenames (`folder_a/`, `folder_b/`):

```bash
uv run tools/benchmark/parity_efficient_loftr.py \
    --folder-a data/custom_pairs/folder_a \
    --folder-b data/custom_pairs/folder_b \
    --variant all \
    --max-dims 640 768 832 \
    --max-pairs 30
```

---

## Performance Benchmarks

Latency, throughput, and host process RAM deltas measured on the first pair of the [MegaDepth-1500](https://huggingface.co/SpatialHub/efficient-loftr-onnx/tree/main/0015_pairs) outdoor dataset (with native input resolutions of $1271 \times 953$ for image A and $1600 \times 1174$ for image B). Measurements reflect 5 unmeasured warmup iterations and 20 timed measurement iterations.

### CUDA Execution (`CUDAExecutionProvider`)

<table>
  <thead>
    <tr>
      <th>Variant</th>
      <th>Max Dim Limit</th>
      <th>Latency Mean (ms)</th>
      <th>Median (ms)</th>
      <th>P95 (ms)</th>
      <th>Throughput (Pairs/s)</th>
      <th>Host RAM Delta</th>
      <th>Peak VRAM Delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>full</code></td>
      <td><code>640</code></td>
      <td>85.42 +/- 5.27</td>
      <td>85.48</td>
      <td>96.85</td>
      <td><strong>11.7</strong></td>
      <td>+564.5 MB</td>
      <td>+1808.0 MB</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>205.73 +/- 40.02</td>
      <td>197.83</td>
      <td>265.53</td>
      <td><strong>4.9</strong></td>
      <td>+200.1 MB</td>
      <td>+4338.0 MB</td>
    </tr>
    <tr>
      <td><code>960</code></td>
      <td>358.43 +/- 86.24</td>
      <td>314.15</td>
      <td>541.61</td>
      <td><strong>2.8</strong></td>
      <td>+360.7 MB</td>
      <td>+6953.0 MB</td>
    </tr>
    <tr>
      <td><code>1024</code></td>
      <td>3560.52 +/- 334.86</td>
      <td>3437.02</td>
      <td>4224.72</td>
      <td><strong>0.3</strong></td>
      <td>+1859.1 MB</td>
      <td>+6953.0 MB</td>
    </tr>
    <tr>
      <td rowspan="4" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>opt</code></td>
      <td><code>640</code></td>
      <td>121.49 +/- 14.46</td>
      <td>122.42</td>
      <td>139.37</td>
      <td><strong>8.2</strong></td>
      <td>+101.2 MB</td>
      <td>+1626.0 MB</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>456.78 +/- 80.48</td>
      <td>503.46</td>
      <td>562.17</td>
      <td><strong>2.2</strong></td>
      <td>+207.7 MB</td>
      <td>+4220.0 MB</td>
    </tr>
    <tr>
      <td><code>960</code></td>
      <td>878.12 +/- 80.74</td>
      <td>879.32</td>
      <td>1009.08</td>
      <td><strong>1.1</strong></td>
      <td>+317.9 MB</td>
      <td>+5904.0 MB</td>
    </tr>
    <tr>
      <td><code>1024</code></td>
      <td>1453.20 +/- 133.25</td>
      <td>1463.95</td>
      <td>1660.77</td>
      <td><strong>0.7</strong></td>
      <td>+490.8 MB</td>
      <td>+6953.0 MB</td>
    </tr>
  </tbody>
</table>

### CPU Execution (`CPUExecutionProvider`)

<table>
  <thead>
    <tr>
      <th>Variant</th>
      <th>Max Dim Limit</th>
      <th>Latency Mean (ms)</th>
      <th>Median (ms)</th>
      <th>P95 (ms)</th>
      <th>Throughput (Pairs/s)</th>
      <th>Host RAM Delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>full</code></td>
      <td><code>640</code></td>
      <td>3085.78 +/- 333.01</td>
      <td>3018.24</td>
      <td>3690.02</td>
      <td><strong>0.3</strong></td>
      <td>+1175.2 MB</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>5789.84 +/- 268.09</td>
      <td>5704.12</td>
      <td>6189.51</td>
      <td><strong>0.2</strong></td>
      <td>+2731.2 MB</td>
    </tr>
    <tr>
      <td><code>960</code></td>
      <td>9123.22 +/- 450.85</td>
      <td>9049.17</td>
      <td>9671.42</td>
      <td><strong>0.1</strong></td>
      <td>+4065.2 MB</td>
    </tr>
    <tr>
      <td><code>1024</code></td>
      <td>11077.42 +/- 399.25</td>
      <td>11022.39</td>
      <td>11817.81</td>
      <td><strong>0.1</strong></td>
      <td>+3581.9 MB</td>
    </tr>
    <tr>
      <td rowspan="4" style="vertical-align: middle; text-align: center; font-weight: bold;"><code>opt</code></td>
      <td><code>640</code></td>
      <td>2833.60 +/- 120.07</td>
      <td>2821.49</td>
      <td>3020.90</td>
      <td><strong>0.4</strong></td>
      <td>+1332.2 MB</td>
    </tr>
    <tr>
      <td><code>832</code></td>
      <td>6070.13 +/- 1001.03</td>
      <td>5627.47</td>
      <td>8153.02</td>
      <td><strong>0.2</strong></td>
      <td>+1490.8 MB</td>
    </tr>
    <tr>
      <td><code>960</code></td>
      <td>8201.68 +/- 455.54</td>
      <td>8263.09</td>
      <td>8992.29</td>
      <td><strong>0.1</strong></td>
      <td>+3115.0 MB</td>
    </tr>
    <tr>
      <td><code>1024</code></td>
      <td>10058.98 +/- 647.43</td>
      <td>10141.71</td>
      <td>10939.23</td>
      <td><strong>0.1</strong></td>
      <td>+3940.5 MB</td>
    </tr>
  </tbody>
</table>

---

## SpatialHub Adapter API & Usage

```python
from spatialhub import EfficientLoFTR

# Initialize with 'opt' variant and CUDA acceleration
matcher = EfficientLoFTR(
    model_type="opt",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Execute matching between two images
result = matcher.match("image_a.jpg", "image_b.jpg", max_dim=1024)

# Render side-by-side visualization
result.visualize(top_k=50, save_path="matches.png")
```

---

## Tooling & Verification Commands

### Export ONNX

Export PyTorch checkpoint weights to standalone ONNX graphs:

```bash
uv run tools/export/export_efficient_loftr.py \
    --checkpoint weights/eloftr_outdoor.ckpt \
    --output-folder onnx_weight \
    --opset 17 \
    --device cpu
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--variant` | `str` | `"all"` | Model variant to export (`full`, `opt`, or `all`). |
| `--checkpoint` | `str` | `None` | Path to source `.ckpt` PyTorch weights file (downloaded if omitted). |
| `--output-folder` | `str` | `onnx_weight` | Destination directory for exported `.onnx` model files. |
| `--width` | `int` | `640` | Input image width in pixels (must be a multiple of 32). |
| `--height` | `int` | `480` | Input image height in pixels (must be a multiple of 32). |
| `--opset` | `int` | `17` | ONNX Operator Set version. |
| `--device` | `str` | `"cpu"` | Hardware device used during export tracing (`cpu` or `cuda`). |

### Parity Check

Validate numerical coordinate and confidence parity against PyTorch:

```bash
uv run tools/benchmark/parity_efficient_loftr.py \
    --folder-a upstream/efficient_loftr/data/0015_pairs/folder_a \
    --folder-b upstream/efficient_loftr/data/0015_pairs/folder_b \
    --variant all \
    --max-dims 640 768 832 \
    --max-pairs 30 \
    --coord-tol 0.01 \
    --conf-tol 0.05 \
    --output-file .profile/parity_eloftr.txt
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--folder-a` | `str` | *Required* | Path to first folder of paired evaluation images. |
| `--folder-b` | `str` | *Required* | Path to second folder of paired evaluation images. |
| `--variant` | `str` | `"all"` | Model variant to test (`full`, `opt`, or `all`). |
| `--max-dims` | `list[str]` | `["640", "1024"]` | Input spatial scale limits to evaluate. |
| `--max-pairs` | `int` | `None` | Maximum number of image pairs to evaluate. |
| `--coord-tol` | `float` | `0.01` | Keypoint coordinate tolerance threshold in pixels. |
| `--conf-tol` | `float` | `0.01` | Confidence score tolerance threshold. |
| `--checkpoint` | `str` | `None` | Optional path to custom PyTorch checkpoint file. |
| `--model-dir` | `str` | `None` | Directory containing local `.onnx` model files. |
| `--output-file` | `str` | `None` | File path to write markdown parity report. |

### Performance Profile

Benchmark execution latency, throughput, and memory deltas:

```bash
uv run tools/benchmark/profile_efficient_loftr.py \
    --variant all \
    --provider all \
    --image-a "upstream/efficient_loftr/data/0015_pairs/folder_a/0000.jpg" \
    --image-b "upstream/efficient_loftr/data/0015_pairs/folder_b/0000.jpg" \
    --max-dims 640 832 960 1024 \
    --warmup 5 \
    --iters 20 \
    --output-file .profile/profile_eloftr.txt
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--variant` | `str` | `"all"` | Model variant to profile (`full`, `opt`, or `all`). |
| `--provider` | `str` | `"all"` | Execution provider filter (`cuda`, `cpu`, or `all`). |
| `--image-a` | `str` | `None` | Path to first evaluation image. |
| `--image-b` | `str` | `None` | Path to second evaluation image. |
| `--resolutions` | `list[str]` | `["640x480"]` | Synthetic spatial resolutions if no images are given. |
| `--max-dims` | `list[str]` | `["640", "1024"]` | Input spatial scale limits to test. |
| `--warmup` | `int` | `5` | Number of unmeasured warmup iterations. |
| `--iters` | `int` | `20` | Number of timed measurement iterations. |
| `--model-dir` | `str` | `None` | Directory containing local `.onnx` model files. |
| `--output-file` | `str` | `None` | File path to write markdown summary table. |

---

## Returned Result Data Structure

Returns a [`MatchResult`](../../../docs/core-and-utils/structures/match_result.md) dataclass:

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image_a` | `str | Path | np.ndarray` | Input | First image reference or NumPy array. |
| `image_b` | `str | Path | np.ndarray` | Input | Second image reference or NumPy array. |
| `keypoints_a` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_a`. |
| `keypoints_b` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_b`. |
| `confidence` | `np.ndarray` | `(N,)` float32 | Match confidence scores in `[0.0, 1.0]`. |

