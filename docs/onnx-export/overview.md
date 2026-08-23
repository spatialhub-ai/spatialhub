# Reproducible ONNX Export Guide

Each model architecture in SpatialHub resides in an isolated submodule directory under `src/spatialhub/models/<model_name>/` containing its own `pyproject.toml` environment configuration. This design allows users to modify PyTorch source code, re-train models, or re-export custom ONNX graphs without polluting the lightweight runtime environment of `spatialhub`.

---

## Submodule Architecture Overview

```text
src/spatialhub/models/
├── efficient_loftr/
│   ├── pyproject.toml            # Export environment config
│   ├── export_onnx.py            # ONNX export script
│   └── src/                      # PyTorch source files
├── depth_anything_3/DepthAnything3/
│   ├── pyproject.toml            # Export environment config
│   ├── export_onnx.py            # ONNX export script
│   └── src/                      # PyTorch source files
├── dinov2/DINOv2/
│   ├── pyproject.toml            # Export environment config
│   ├── export_onnx.py            # ONNX export script
│   └── src/                      # PyTorch source files
├── fastsam/FastSAM/
│   ├── pyproject.toml            # Export environment config
│   ├── export_onnx.py            # ONNX export script
│   └── src/                      # PyTorch source files
├── sam/SAM/
│   ├── pyproject.toml            # Export environment config
│   ├── export_onnx.py            # ONNX export script
│   └── src/                      # PyTorch source files
└── cnos/CNOS/
    ├── pyproject.toml            # Export environment config
    ├── export_dinov2.py          # DINOv2 sub-module ONNX export script
    ├── export_fastsam.py         # FastSAM sub-module ONNX export script
    └── export_sam.py             # SAM sub-module ONNX export script
```

---

## General ONNX Export Workflow

### Step 1: Navigate to the Target Submodule Directory
Each model's export scripts and source code sit in its submodule folder beside its `pyproject.toml` file.

```bash
cd src/spatialhub/models/<model_name>/
# Or for nested submodules:
cd src/spatialhub/models/<model_name>/<SubmoduleName>/
```

### Step 2: Synchronize Virtual Environment
Use `uv` (or `pip`) to install the PyTorch export dependencies specified in that submodule's `pyproject.toml`:

```bash
uv sync
```

### Step 3: Modify PyTorch Source Code (Optional)
Modify PyTorch model layers, loss functions, attention operators, or forward pass wrappers inside the submodule's `src/` directory if custom behavior or alternative dynamic axes are required.

### Step 4: Execute Export Script
Run the submodule's `export_onnx.py` script. The export script loads PyTorch weights, traces the forward graph, applies dynamic axis rules, serializes the `.onnx` binary file, and performs ONNX checker validation.

```bash
uv run python export_onnx.py --checkpoint <path_to_ckpt> --output-path <destination_onnx>
```

---

## Model Export Commands Summary

| Model | Submodule Path | Export Command |
| :--- | :--- | :--- |
| **EfficientLoFTR** | `src/spatialhub/models/efficient_loftr` | `uv run python export_onnx.py --checkpoint weights/eloftr_outdoor.ckpt --output-path weights/eloftr_outdoor.onnx` |
| **Depth Anything 3** | `src/spatialhub/models/depth_anything_3/DepthAnything3` | `uv run python export_onnx.py --model-name depth-anything/DA3-BASE --onnx-path weights/da3_base.onnx` |
| **DINOv2** | `src/spatialhub/models/dinov2/DINOv2` | `uv run python export_onnx.py --model-name dinov2_vitl14 --output-folder ./onnx_model` |
| **FastSAM** | `src/spatialhub/models/fastsam/FastSAM` | `uv run python export_onnx.py --checkpoint FastSAM-x.pt --output-folder ./onnx_model` |
| **SAM** | `src/spatialhub/models/sam/SAM` | `uv run python export_onnx.py --model-type vit_h --out-encoder sam_image_encoder.onnx --out-decoder sam_mask_decoder.onnx` |
| **CNOS** | `src/spatialhub/models/cnos/CNOS` | `uv run python export_dinov2.py --model-name dinov2_vitl14 --output-folder ./pretrained` |
