# Quickstart Guide

This guide provides technical usage examples across perception tasks in SpatialHub. All model adapters follow a unified initialization, inference execution, and result visualization pattern.

---

## 1. Feature Matching (EfficientLoFTR)

```python
from spatialhub import EfficientLoFTR

# 1. Initialize matcher session
matcher = EfficientLoFTR(
    model_type="opt",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 2. Match keypoints between image pair
result = matcher.match("image_a.jpg", "image_b.jpg", max_dim=1024)

# 3. Access keypoints and confidence
print("Keypoints A shape:", result.keypoints_a.shape)
print("Keypoints B shape:", result.keypoints_b.shape)

# 4. Render visualization overlay
result.visualize(top_k=50, save_path="matches.png")
```

---

## 2. Depth Estimation (Depth Anything 3)

```python
import cv2
from spatialhub import DepthAnything3

# 1. Initialize depth estimator
estimator = DepthAnything3(
    model_name="da3_base",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 2. Estimate depth maps
result = estimator.estimate_depth(images=["view1.png", "view2.png"])

# 3. Render colorized depth map
colorized = estimator.visualize(result.depth[0])
cv2.imwrite("depth_output.png", cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
```

---

## 3. Image Embeddings (DINOv2)

```python
from spatialhub import DINOV2

# 1. Initialize DINOv2 feature extractor
extractor = DINOV2(model_variant="dinov2_vitl14")

# 2. Extract global L2-normalized CLS token embedding
result = extractor.extract_features("object.png", l2_normalize=True)

print("Embedding shape:", result.features.shape)  # (1, 1024)
```

---

## 4. Proposal Segmentation (FastSAM)

```python
from spatialhub import FastSAM

# 1. Initialize FastSAM proposal segmentor
segmentor = FastSAM(model_variant="FastSAM-x")

# 2. Generate mask proposals
result = segmentor.generate_masks("scene.png", conf_threshold=0.3)

# 3. Render colored mask overlay
result.visualize_mask(save_path="fastsam_output.png")
```

---

## 5. Automatic Mask Generation (SAM)

```python
from spatialhub import SAM

# 1. Initialize SAM grid segmentor
segmentor = SAM(model_variant="sam_vit_h")

# 2. Grid-sample point prompts across image
result = segmentor.generate_masks("image.jpg", points_per_side=32)

# 3. Render mask visualization
result.visualize_mask(save_path="sam_output.png")
```

---

## 6. CAD Zero-Shot Detection (CNOS)

```python
from spatialhub import CNOS, DINOV2, FastSAM

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 1. Initialize sub-adapters
descriptor = DINOV2(providers=providers)
segmentor = FastSAM(providers=providers)

# 2. Initialize CNOS adapter with 3D CAD mesh file
cnos = CNOS(
    model_path="model.ply",
    model_unit="mm",
    descriptor=descriptor,
    segmentor=segmentor,
    providers=providers,
)

# 3. Execute zero-shot detection
result = cnos.inference("scene.png", num_max_dets=3, conf_threshold=0.15)

# 4. Render detection overlay
result.visualize_mask(save_path="cnos_output.png")
```
