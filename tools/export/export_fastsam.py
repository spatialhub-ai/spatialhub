# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
#     "ultralytics>=8.4.126",
#     "onnxslim==0.1.96",
# ]
# ///

import argparse
import logging
import os
from pathlib import Path
import shutil
import numpy as np

import onnx
import onnxruntime as ort

import torch

import urllib.request
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt"

def ensure_checkpoint(checkpoint_path: str | Path | None) -> Path:
    """Verify that FastSAM PyTorch checkpoint exists, downloading if missing.

    Args:
        checkpoint_path: Local path to FastSAM checkpoint file.

    Returns:
        Path: Path to verified checkpoint file.
    """
    
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        logger.info("Checkpoint exists at %s. Skipping download.", checkpoint_path)
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("FastSAM checkpoint missing. Downloading to %s...", checkpoint_path)
    download_model(fastsam_url, checkpoint_path)

    return checkpoint_path


def check_onnx(onnx_path: str):
    """
    Validates the exported ONNX model graph.
    """
    print(f"Checking ONNX model integrity at {onnx_path}...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    model = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model=model)
        print("The ONNX graph is clean and valid!")
    except onnx.checker.ValidationError as e:
        raise RuntimeError(f"Graph validation failed: {e}") from e


def flatten_to_numpy(val):
    """Recursively converts nested tensors, lists, and tuples into a flat list of NumPy arrays."""
    tensors = []
    if isinstance(val, torch.Tensor):
        tensors.append(val.detach().cpu().numpy())
    elif isinstance(val, (tuple, list)):
        for item in val:
            tensors.extend(flatten_to_numpy(item))
    return tensors


def validate_onnx(checkpoint_path: str, onnx_path: str, imgsz: int = 1024, rtol: float = 1e-3, atol: float = 1e-3,):
    """Validates ONNX graph validity and compares outputs with PyTorch."""
    logging.info(f"Validating ONNX model at {onnx_path}...")

    # Structural check
    check_onnx(onnx_path)

    # PyTorch model inference
    fastsam_pt = YOLO(checkpoint_path)
    py_model = fastsam_pt.model
    py_model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)

    with torch.no_grad():
        py_outputs = py_model(dummy_input)
        py_numpy_outputs = flatten_to_numpy(py_outputs)

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    # Shape-matched numerical comparison
    all_matched = True
    output_labels = ["Boxes / Scores / Mask Coeffs", "Proto Masks"]

    for idx, ort_out in enumerate(ort_outputs):
        label = output_labels[idx] if idx < len(output_labels) else f"Output {idx}"
        
        # Locate corresponding PyTorch tensor by matching shape
        matching_py = [p for p in py_numpy_outputs if p.shape == ort_out.shape]
        
        if not matching_py:
            logging.error(f"Could not find matching PyTorch tensor for shape {ort_out.shape}")
            all_matched = False
            continue

        py_out = matching_py[0]
        diff = np.max(np.abs(py_out - ort_out))
        mean_diff = np.mean(np.abs(py_out - ort_out))
        match = np.allclose(py_out, ort_out, rtol=rtol, atol=atol)

        logging.info(f"   {label} {ort_out.shape} -> Max Diff: {diff:.6e} | Mean Diff: {mean_diff:.6e}")

        if not match:
            all_matched = False

    if all_matched:
        logging.info("PyTorch and ONNX outputs MATCH within tolerance!")
    else:
        logging.warning("Numerical difference exceeded standard tolerance.")


def download_model(url: str, output_path: str) -> None:

    logging.info(f"Downloading model weights to {output_path}...")
    
    urllib.request.urlretrieve(url, output_path)
    
    logging.info("Model download complete!")


def export_fastsam(
    checkpoint_path: str | Path,
    output_folder: str | Path,
    imgsz: int = 1024,
    opset: int = 17,
    dynamic: bool = True,
) -> Path:
    """Export FastSAM PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch FastSAM model file (.pt).
        output_folder: Destination directory for exported .onnx model file.
        imgsz: Image spatial size (default: 1024).
        opset: ONNX operator set version (default: 17).
        dynamic: Flag to export dynamic batch size axes.

    Returns:
        Path: Destination path of exported ONNX model file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_folder = Path(output_folder)

    logger.info("Loading FastSAM model from %s...", checkpoint_path)
    model = YOLO(str(checkpoint_path))

    logger.info("Exporting to ONNX (imgsz=%d, opset=%d, dynamic=%s)...", imgsz, opset, dynamic)
    exported_path_str = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=opset)
    exported_path = Path(exported_path_str)

    output_folder.mkdir(parents=True, exist_ok=True)
    target_path = output_folder / exported_path.name

    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    logger.info("FastSAM ONNX model saved to: %s", target_path)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastSAM PyTorch model to ONNX format.")
    parser.add_argument("--checkpoint", type=str, default="FastSAM/FastSAM-s.pt", help="Path to FastSAM-x.pt or FastSAM-s.pt checkpoint (auto-downloaded if missing).")
    parser.add_argument("--output-folder", type=str, default="./onnx_model", help="Target folder for exported ONNX model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image spatial dimension for segmentation (default: 640).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version (default: 17).")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Export with dynamic axes for batch dimension.")
    args = parser.parse_args()

    checkpoint_path = ensure_checkpoint(args.checkpoint)
    onnx_path = export_fastsam(
        checkpoint_path=checkpoint_path,
        output_folder=args.output_folder,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )

    validate_onnx(checkpoint_path=str(checkpoint_path), onnx_path=str(onnx_path), imgsz=args.imgsz)


if __name__ == "__main__":
    main()

