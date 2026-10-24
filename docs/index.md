# SpatialHub

SpatialHub is a Python library for composing deep vision models and geometric algorithms into spatial perception pipelines.

Building spatial perception systems typically requires integrating separate vision models—such as monocular depth estimation, instance segmentation, keypoint matching, and 6D object pose—with classical geometric algorithms. In practice, connecting these components requires reconciling conflicting coordinate systems, disparate tensor formats, and differing runtime dependencies across individual models.

SpatialHub addresses this by providing standardized geometric data structures, consistent coordinate conventions, and self-contained model inference sessions.

---

## Architecture

SpatialHub organizes vision models, data structures, and geometric utilities into a structured dataflow:

![SpatialHub Architecture](assets/arch_concept4_dataflow.svg)

### Input Data

Pipelines accept standard vision inputs as raw NumPy arrays, filepaths, or 3D mesh files:

- RGB images and sequential video frames (`uint8` arrays in RGB channel order).
- Camera intrinsic matrices ($K \in \mathbb{R}^{3 \times 3}$) and raw depth maps.
- 3D CAD meshes (`.ply`, `.obj`, `.stl`) for template rendering and model-based tracking.

### Inference

Models run through isolated ONNX Runtime sessions without requiring a PyTorch installation at runtime:

- **Hardware Acceleration**: Executes on available hardware providers (`CUDAExecutionProvider`, `TensorrtExecutionProvider`, or CPU fallback).
- **Weight Resolution**: Pretrained ONNX graphs are downloaded and cached locally from Hugging Face Hub on first run.
- **Vision Modalities**: Supports distinct perception tasks including monocular and multi-view depth estimation, instance segmentation, visual feature extraction, keypoint matching, and 6D object pose estimation. Each model produces predictions directly in the library's standardized data structures.

### Standardized Data Structures

Model outputs populate validated dataclasses defined in `spatialhub.structures` instead of unstructured arrays or vendor-specific dictionaries:

- **Pure NumPy Containers**: Structures hold validated NumPy arrays with zero framework locks.
- **Leading Batch Dimension**: All fields enforce an explicit leading batch dimension $(N, \dots)$ for consistent dimensional contracts.
- **Synchronized Slicing**: Indexing operations slice all parallel fields (masks, bounding boxes, scores, identifiers) simultaneously.
- **Coordinate Conventions**: All camera matrices, 3D coordinates, and $\mathrm{SE}(3)$ transformations strictly adhere to the OpenCV camera coordinate standard ($+X$ right, $+Y$ down, $+Z$ forward along the optical axis).

### Geometric & Rendering Utilities

SpatialHub provides functional utilities and hardware rendering support for spatial data processing:

- **Geometric & Spatial Utilities** (`spatialhub.utils`): Stateless operations for camera projection and 3D backprojection, $\mathrm{SE}(3)$ rigid transformation algebra, metric depth range mapping, and 3D bounding box and coordinate axis visual projections.
- **Hardware Offscreen Rendering** (`spatialhub.utils.moderngl`): Headless ModernGL contexts (EGL, OSMesa, WGL) supporting batched CAD mesh rasterization, multi-buffer generation (color, surface normals, depth), and fullscreen fragment shader filtering.

### Downstream Applications

The standardized data structures and utilities allow users to construct customized multi-stage vision pipelines, such as:

- **6D Object Pose Tracking**: Integrating CAD mesh registration and iterative pose refinement on video streams.
- **Zero-Shot CAD Detection**: Combining headless CAD viewpoint rendering with visual feature matching for CAD object localization.
- **Dense 3D Reconstruction**: Backprojecting monocular or multi-view depth maps into registered metric point clouds.
- **Two-View Geometry**: Establishing geometric point correspondences for relative camera pose estimation.

---

## Next Steps

- **[Installation](getting_started/installation.md)**: Hardware execution providers (CUDA, TensorRT, CPU) and optional rendering dependencies.
- **[Quickstart](getting_started/quickstart.md)**: End-to-end walkthrough running monocular depth estimation, instance segmentation, and 3D point cloud unprojection.
- **[Model Overview](models/overview.md)**: Catalog of supported model architectures and output contract specifications.
