# Installation Guide

SpatialHub requires **Python 3.12+**.

---

## 1. Installation via PyPI

Install the lightweight CPU package directly from PyPI:

```bash
pip install spatialhub
```

### Optional Hardware Acceleration

For GPU hardware acceleration via CUDA:

```bash
pip install "spatialhub[gpu]"
```

---

## 2. Local Workspace & Development Setup

SpatialHub uses **`uv`** for virtual environment management and project synchronization.

### Step 1: Clone Repository
```bash
git clone https://github.com/pankajkaushik12/spatialhub.git
cd spatialhub
```

### Step 2: Install `uv` & Sync Dependencies
```bash
uv sync
```

### Step 3: Verify Environment
```bash
uv run python -c "from spatialhub import EfficientLoFTR, DepthAnything3, CNOS, FastSAM, SAM, DINOV2; print('SpatialHub initialized successfully!')"
```

---

## 3. Execution Providers Support Matrix

ONNX Runtime adapters accept execution provider configurations:

| Provider String | Target Hardware | Requirements |
| :--- | :--- | :--- |
| `"CPUExecutionProvider"` | CPU (Default) | Built-in default |
| `"CUDAExecutionProvider"` | NVIDIA GPUs | `onnxruntime-gpu`, CUDA toolkit |
| `"TensorrtExecutionProvider"` | NVIDIA TensorRT | TensorRT runtime |
| `"DirectMLExecutionProvider"` | Windows DirectX 12 GPUs | `onnxruntime-directml` |
