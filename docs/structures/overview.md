# Data Structures API Reference

The `spatialhub.structures` module defines the standard dataclass return contracts used across all model adapters in SpatialHub. Consistent return structures allow perception pipelines to interoperate without custom adapter wrappers.

---

## 1. `MatchResult`

Returned by feature matching adapters (e.g., [`EfficientLoFTRAdapter`](../models/eloftr.md)).

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class MatchResult:
    image_a: str | Path | np.ndarray
    image_b: str | Path | np.ndarray
    keypoints_a: np.ndarray
    keypoints_b: np.ndarray
    confidence: np.ndarray
```

### Attributes

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image_a` | `str \| Path \| np.ndarray` | `(H_a, W_a, C)` or path | First input image reference or NumPy array. |
| `image_b` | `str \| Path \| np.ndarray` | `(H_b, W_b, C)` or path | Second input image reference or NumPy array. |
| `keypoints_a` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_a` pixel space. |
| `keypoints_b` | `np.ndarray` | `(N, 2)` float32 | Verified keypoint `[x, y]` coordinates in `image_b` pixel space. |
| `confidence` | `np.ndarray` | `(N,)` float32 | Match confidence scores ranging from `0.0` to `1.0`. |

### Methods

#### `visualize(conf_thresh=0.5, max_side=800, top_k=None, save_path=None)`
Renders a side-by-side visualization of matched keypoint pairs connected by confidence-colored lines.

```python
result.visualize(
    conf_thresh=0.5,
    max_side=800,
    top_k=50,
    save_path="matches.png"
)
```

- **`conf_thresh`** (`float`): Minimum confidence threshold for rendering matches. Default is `0.5`.
- **`max_side`** (`int`): Caps the maximum spatial dimension of the rendered visualization image. Default is `800`.
- **`top_k`** (`int | None`): Limits output display to the top $k$ highest confidence matches.
- **`save_path`** (`str | Path | None`): File path to save output visualization image. If `None`, returns image array without writing.

---

## 2. `DepthPredictionResult`

Returned by depth estimation adapters (e.g., [`DepthAnything3Adapter`](../models/depthanything3.md)).

```python
from dataclasses import dataclass
import numpy as np
from typing import Literal

@dataclass
class DepthPredictionResult:
    image: np.ndarray
    depth: np.ndarray
    conf: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    depth_type: Literal["metric", "relative", "inverse", "disparity"] = "metric"
```

### Attributes

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(N, H, W, 3)` or `(H, W, 3)` uint8 | Input RGB image array. |
| `depth` | `np.ndarray` | `(N, H, W)` float32 | Predicted depth maps in meters or relative scale. |
| `conf` | `np.ndarray \| None` | `(N, H, W)` float32 | Confidence maps indicating prediction certainty. |
| `intrinsics` | `np.ndarray \| None` | `(N, 3, 3)` float32 | Camera intrinsic parameters `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. |
| `extrinsics` | `np.ndarray \| None` | `(N, 4, 4)` float32 | Camera extrinsic transformation matrices `[R | t]`. |
| `depth_type` | `str` | N/A | Scale interpretation: `"metric"`, `"relative"`, `"inverse"`, or `"disparity"`. |

---

## 3. `FeatureExtractionResult`

Returned by feature extraction adapters (e.g., [`DINOv2Adapter`](../models/dinov2.md)).

```python
from dataclasses import dataclass
import numpy as np
from typing import Literal

@dataclass
class FeatureExtractionResult:
    images: np.ndarray
    features: np.ndarray
    embedding_type: Literal["global", "dense"] = "global"
    l2_normalized: bool = False
```

### Attributes

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `images` | `np.ndarray` | `(N, H, W, 3)` uint8 | Preprocessed input image batch. |
| `features` | `np.ndarray` | `(N, D)` or `(N, C, H, W)` float32 | Extracted feature embedding tensor (e.g., `D=1024` for ViT-L/14). |
| `embedding_type` | `str` | N/A | Representation scope: `"global"` (CLS token) or `"dense"` (patch tokens). |
| `l2_normalized` | `bool` | N/A | Flag indicating if embedding vectors are unit L2-normalized (`||v||_2 = 1.0`). |

---

## 4. `SegmentationResult`

Returned by object proposal, segmentation, and detection adapters (e.g., [`FastSAMAdapter`](../models/fastsam.md), [`SAMAdapter`](../models/sam.md), [`CNOSAdapter`](../models/cnos.md)).

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class SegmentationResult:
    image: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray | None = None
    class_names: list[str] | None = None
```

### Attributes

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | `(H, W, 3)` uint8 | Input RGB image array. |
| `boxes` | `np.ndarray` | `(N, 4)` float32 | Bounding box coordinates in `[x1, y1, x2, y2]` format. |
| `masks` | `np.ndarray` | `(N, H, W)` bool | Binary spatial segmentation masks. |
| `scores` | `np.ndarray` | `(N,)` float32 | Proposal detection or matching confidence scores. |
| `class_ids` | `np.ndarray \| None` | `(N,)` int | Numerical class identifiers (defaults to `0` for single-object matching). |
| `class_names` | `list[str] \| None` | Length `N` | String label list corresponding to detected classes. |

### Methods

#### `visualize_mask(save_path=None)`
Renders a colorized segmentation mask overlay on top of the original image with bounding box outlines.

```python
result.visualize_mask(save_path="segmentation.png")
```

- **`save_path`** (`str | Path | None`): Destination file path to save rendered PNG overlay image.
