# Installation Guide

SpatialHub requires **Python 3.12+**.

---

## PyPI Installation

Choose the appropriate installation command for your hardware and target tasks:

### Standard Installation

For CPU inference:
```bash
pip install "spatialhub[cpu]"
```

For NVIDIA GPU acceleration (CUDA / TensorRT):
```bash
pip install "spatialhub[gpu]"
```

### 3D CAD & Rendering Installation

For workflows requiring 3D CAD mesh loading and template rendering (e.g. FoundationPose, CNOS):

For CPU with rendering:
```bash
pip install "spatialhub[cpu,render]"
```

For GPU with rendering:
```bash
pip install "spatialhub[gpu,render]"
```

!!! note "Backend Conflict Warning"
    Do not install both `onnxruntime` and `onnxruntime-gpu` in the same Python virtual environment as their binary C++ namespaces conflict.

---

## Development Setup

SpatialHub uses `uv` for virtual environment management and project synchronization.

### Clone Repository
```bash
git clone https://github.com/spatialhub-ai/spatialhub.git
cd spatialhub
```

### Environment Synchronization
```bash
uv sync
```

### Environment Verification
```bash
uv run python -c "from spatialhub import EfficientLoFTR, DepthAnything3, CNOS, FastSAM, SAM, DINOv2; print('SpatialHub initialized successfully!')"
```

---

## Execution Providers Support Matrix

ONNX Runtime adapters accept execution provider configurations:

| Provider String | Target Hardware | Requirements |
| :--- | :--- | :--- |
| `"CPUExecutionProvider"` | CPU (Default) | Built-in default |
| `"CUDAExecutionProvider"` | NVIDIA GPUs | `onnxruntime-gpu`, CUDA toolkit |
| `"TensorrtExecutionProvider"` | NVIDIA TensorRT | TensorRT runtime |
| `"DirectMLExecutionProvider"` | Windows DirectX 12 GPUs | `onnxruntime-directml` |
