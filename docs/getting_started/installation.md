# Installation

SpatialHub requires Python 3.10 or higher.

---

## Environment Setup

Create an isolated virtual environment before installing packages.

### Using `uv`

```bash
# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or initialize a new project workspace:

```bash
uv init my_spatial_project
cd my_spatial_project
```

### Using standard `python` & `pip`

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## Package Installation

SpatialHub packages dependencies into modular extras:
- `cpu`: Installs `onnxruntime` for CPU execution.
- `gpu`: Installs `onnxruntime-gpu` for CUDA and TensorRT acceleration.
- `render`: Installs `moderngl`, `trimesh`, and `scipy` for 3D mesh template rasterization and fullscreen shaders.

Select the target configuration for your runtime:

### CPU Inference

Using `uv`:
```bash
uv add "spatialhub[cpu]"
```

Using `pip`:
```bash
pip install "spatialhub[cpu]"
```

### GPU Acceleration (CUDA / TensorRT)

Using `uv`:
```bash
uv add "spatialhub[gpu]"
```

Using `pip`:
```bash
pip install "spatialhub[gpu]"
```

### 3D CAD & ModernGL Rendering

For workflows requiring 3D CAD template rendering or GPU shader filtering (e.g., CNOS, FoundationPose):

**CPU with rendering:**
```bash
uv add "spatialhub[cpu,render]"
# or: pip install "spatialhub[cpu,render]"
```

**GPU with rendering:**
```bash
uv add "spatialhub[gpu,render]"
# or: pip install "spatialhub[gpu,render]"
```

For headless Linux servers or Docker containers requiring offscreen EGL or OSMesa driver configurations, see [Headless Contexts](../rendering/headless.md).

!!! note "Runtime Package Conflict"
    Do not install `cpu` and `gpu` extras in the same virtual environment. `onnxruntime` and `onnxruntime-gpu` share C++ binary namespaces and conflict if co-installed.

---

## Installing from Source

To contribute to SpatialHub or develop local pipeline extensions, clone the repository and set up an editable environment.

### Clone the Repository

```bash
git clone https://github.com/spatialhub-ai/spatialhub.git
cd spatialhub
```

### Using `uv`

Synchronize dependencies and development tools directly from `pyproject.toml` (`uv sync` installs the `dev` dependency group containing `pytest` and `psutil` by default):

```bash
# GPU development with 3D rendering
uv sync --extra gpu --extra render

# Or CPU-only development
uv sync --extra cpu --extra render
```

### Using `pip`

Create an activated virtual environment and install in editable mode (development tools from the PEP 735 `dev` group must be installed manually when using `pip`):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with extras
pip install -e ".[gpu,render]"

# Install development dependencies
pip install pytest psutil
```

### Running Tests

Verify the local development environment against the unit test suite:

```bash
uv run pytest
# or: pytest
```

---

## Verification

Verify the installation and inspect the available ONNX Runtime execution providers in your environment:

Using `python`:
```bash
python -c "import spatialhub as sh, onnxruntime as ort; print(f'SpatialHub installed: True (v{sh.__version__})'); print(f'Active providers:     {ort.get_available_providers()}')"
```

Using `uv`:
```bash
uv run python -c "import spatialhub as sh, onnxruntime as ort; print(f'SpatialHub installed: True (v{sh.__version__})'); print(f'Active providers:     {ort.get_available_providers()}')"
```

Expected output in a GPU-configured environment:
```text
SpatialHub installed: True (v<installed_version>)
Active providers:     ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Next Steps

Proceed to the [Quickstart Guide](quickstart.md) to execute your first perception pipeline.
