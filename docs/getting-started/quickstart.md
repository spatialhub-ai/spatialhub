# Quickstart Guide

This guide provides technical usage examples across perception tasks in SpatialHub. All model adapters follow a unified initialization, inference execution, and result visualization pattern.

---

## 1. Feature Matching (EfficientLoFTR)

```python
from spatialhub import EfficientLoFTR

# Initialize matcher session
matcher = EfficientLoFTR(
    model_type="opt",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Match keypoints between image pair
result = matcher.match("image_a.jpg", "image_b.jpg", max_dim=1024)

# Access keypoints and confidence
print("Keypoints A shape:", result.keypoints_a.shape)
print("Keypoints B shape:", result.keypoints_b.shape)

# Render visualization overlay
result.visualize(top_k=50, save_path="matches.png")
```

---

## 2. Depth Estimation (Depth Anything 3)

```python
import cv2
from spatialhub import DepthAnything3

# Initialize depth estimator
estimator = DepthAnything3(
    model_name="da3_base",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Estimate depth maps
result = estimator.estimate_depth(images=["view1.png", "view2.png"])

# Render colorized depth map
colorized = estimator.visualize(result.depth[0])
cv2.imwrite("depth_output.png", cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
```

---

## 3. Image Embeddings (DINOv2)

```python
from spatialhub import DINOv2

# Initialize DINOv2 feature extractor
extractor = DINOv2(model_variant="vitl14")

# Extract global L2-normalized CLS token embedding
result = extractor.extract_features("object.png", l2_normalize=True)

print("Embedding shape:", result.features.shape)  # (1, 1024)
```

---

## 4. Proposal Segmentation (FastSAM)

```python
from spatialhub import FastSAM

# Initialize FastSAM proposal segmentor
segmentor = FastSAM(model_variant="x")

# Generate mask proposals
result = segmentor.generate_masks("scene.png", conf_threshold=0.3)

# Render colored mask overlay
result.visualize_mask(save_path="fastsam_output.png")
```

---

## 5. Automatic Mask Generation (SAM)

```python
from spatialhub import SAM

# Initialize SAM grid segmentor
segmentor = SAM(model_variant="vit_h")

# Grid-sample point prompts across image
result = segmentor.generate_masks("image.jpg", points_per_side=32)

# Render mask visualization
result.visualize_mask(save_path="sam_output.png")
```

---

## 6. CAD Zero-Shot Detection (CNOS)

```python
from spatialhub import CNOS, DINOv2, FastSAM

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize sub-adapters
descriptor = DINOv2(providers=providers)
segmentor = FastSAM(providers=providers)

# Initialize CNOS adapter with 3D CAD mesh file
cnos = CNOS(
    model_path="model.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
    providers=providers,
)

# Execute zero-shot detection
result = cnos.inference("scene.png", num_max_dets=3, conf_threshold=0.15)

# Render detection overlay
result.visualize_mask(save_path="cnos_output.png")
```
