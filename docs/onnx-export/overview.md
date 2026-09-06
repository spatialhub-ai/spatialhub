# ONNX Export Workflow

SpatialHub separates lightweight ONNX Runtime execution (`src/spatialhub/models/`) from standalone PyTorch export utilities (`tools/export/`) and upstream model research repositories (`upstream/`). This structure allows users to inspect or modify PyTorch architectures and export custom ONNX graphs without adding heavy deep learning dependencies to the runtime package.

---

## Directory Architecture Overview

```text
spatialhub/
├── tools/export/                       # Standalone ONNX export utilities
│   ├── export_depth_anything_3.py
│   ├── export_efficient_loftr.py
│   ├── export_dinov2.py
│   ├── export_fastsam.py
│   ├── export_sam.py
│   └── export_foundationpose.py
├── upstream/                           # Upstream PyTorch research submodules
│   ├── depth_anything_3/
│   ├── efficient_loftr/
│   ├── cnos/
│   └── foundationpose/
└── src/spatialhub/models/              # Lightweight ONNX Runtime inference packages
    ├── depth_anything_3/
    ├── efficient_loftr/
    ├── dinov2/
    ├── fastsam/
    ├── sam/
    ├── cnos/
    └── foundationpose/
```

---

## General Export Procedure

### Execute Export Script via `uv`

Each export script in `tools/export/` contains inline dependency metadata (PEP 723). Run the target script directly with `uv run`, which automatically resolves isolated PyTorch dependencies and executes the export:

```bash
uv run tools/export/export_<model>.py [options]
```

### Script Execution Flow

1. The script initializes the model architecture from `upstream/` or standard hubs.
2. Checkpoint weights are loaded into evaluation mode.
3. The forward graph is traced using `torch.onnx.export` with specified dynamic axes.
4. ONNX graph verification is performed via `onnx.checker.check_model`.
5. The exported `.onnx` file is written to the destination directory.

---

## Model Export Commands Summary

| Model | Source | Export Command |
| :--- | :--- | :--- |
| **EfficientLoFTR** | `upstream/efficient_loftr` | `uv run tools/export/export_efficient_loftr.py --checkpoint weights/eloftr_outdoor.ckpt --output-path weights/eloftr_outdoor.onnx` |
| **Depth Anything 3** | `upstream/depth_anything_3` | `uv run tools/export/export_depth_anything_3.py --model-name depth-anything/DA3-BASE --onnx-path weights/da3_base.onnx` |
| **DINOv2** | PyTorch Hub | `uv run tools/export/export_dinov2.py --model-name dinov2_vitl14 --output-folder ./weights` |
| **FastSAM** | Ultralytics | `uv run tools/export/export_fastsam.py --checkpoint FastSAM-x.pt --output-folder ./weights --imgsz 640` |
| **SAM** | Segment Anything | `uv run tools/export/export_sam.py --model-type vit_h --out-encoder ./weights/sam_image_encoder.onnx --out-decoder ./weights/sam_mask_decoder.onnx` |
| **CNOS** | Upstream / Hubs | `uv run tools/export/export_dinov2.py` & `uv run tools/export/export_fastsam.py` |
| **FoundationPose** | `upstream/foundationpose` | `uv run tools/export/export_foundationpose.py --weights-dir ./upstream/foundationpose/weights --output-folder ./weights` |

