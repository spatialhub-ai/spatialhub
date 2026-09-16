# ONNX Export Workflow

SpatialHub provides standalone PyTorch export utilities under `tools/export/` to convert research model architectures into self-contained ONNX model graphs.

---

## Standalone Execution via `uv`

Export scripts use [PEP 723](https://peps.python.org/pep-0723/) inline script metadata to declare isolated PyTorch and upstream dependencies. This allows models to be traced without installing PyTorch or CUDA training toolkits into your primary SpatialHub runtime environment:

```bash
uv run tools/export/export_<model>.py [options]
```

---

## Standard Export Pipeline

1. **Architecture Initialization**: Instantiates the model architecture from local research definitions or standard model hubs.
2. **Weight Restoration**: Loads PyTorch checkpoint parameters and sets layers to evaluation mode (`model.eval()`).
3. **Graph Tracing**: Executes `torch.onnx.export` with configured dynamic batch and spatial dimension axes.
4. **Graph Validation**: Verifies structural graph integrity using `onnx.checker.check_model`.
5. **Serialization**: Writes the optimized `.onnx` binary file to disk.

---

## Supported Export Scripts

| Model | Source Architecture | Export Script | Required Checkpoint / Source |
| :--- | :--- | :--- | :--- |
| **EfficientLoFTR** | `upstream/efficient_loftr` | `tools/export/export_efficient_loftr.py` | `eloftr_outdoor.ckpt` |
| **Depth Anything 3** | `upstream/depth_anything_3` | `tools/export/export_depth_anything_3.py` | Hugging Face Model ID / Weights |
| **DINOv2** | PyTorch Hub (`torchvision`) | `tools/export/export_dinov2.py` | PyTorch Hub model variant |
| **FastSAM** | Ultralytics | `tools/export/export_fastsam.py` | `FastSAM-x.pt` |
| **SAM** | Segment Anything | `tools/export/export_sam.py` | `sam_vit_h_4b8939.pth` |
| **FoundationPose** | `upstream/foundationpose` | `tools/export/export_foundationpose.py` | `weights/` directory |

---

## Common CLI Options

Most export utilities share the following standard command-line parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | `str` | *Contextual* | Path to source `.ckpt`, `.pt`, or `.pth` weights file. |
| `--output-path` | `str` | *Contextual* | Destination path or directory for exported `.onnx` files. |
| `--opset` | `int` | `17` | ONNX Operator Set version. |
| `--device` | `str` | `"cpu"` | Hardware device used for graph tracing (`cpu` or `cuda`). |
