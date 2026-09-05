# DepthPredictionResult

`spatialhub.structures.DepthPredictionResult` encapsulates estimated metric or relative depth maps and camera parameters produced by depth estimation adapters such as [`DepthAnything3Adapter`](../../models/depthanything3.md).

```python
from spatialhub.structures import DepthPredictionResult
```

---

## Fields

| Field | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(N, H, W, 3)` or `(H, W, 3)` uint8 | Input RGB image array(s). |
| `depth` | `np.ndarray` | `(N, H, W)` float32 | Predicted depth map array in meters or relative scale. |
| `conf` | `np.ndarray \| None` | `(N, H, W)` float32 | Per-pixel prediction confidence, if provided by the model. |
| `intrinsics` | `np.ndarray \| None` | `(N, 3, 3)` float32 | Estimated camera intrinsic matrices `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. |
| `extrinsics` | `np.ndarray \| None` | `(N, 4, 4)` float32 | Estimated camera extrinsic transformation matrices `[R \| t]`. |
| `depth_type` | `str` | `"metric"` | Depth interpretation scale: `"metric"`, `"relative"`, `"inverse"`, or `"disparity"`. |

---

## Usage Example

```python
from spatialhub import DepthAnything3
from spatialhub.utils import reproject_depth_to_3d

estimator = DepthAnything3(model_variant="da3_metric_large")
result = estimator.predict("scene.png")

print(f"Depth shape: {result.depth.shape}")
print(f"Depth range: {result.depth.min():.2f}m to {result.depth.max():.2f}m")

# Reproject depth map to 3D point cloud using camera intrinsics
if result.intrinsics is not None:
    xyz_map = reproject_depth_to_3d(result.depth[0], result.intrinsics[0])
```
